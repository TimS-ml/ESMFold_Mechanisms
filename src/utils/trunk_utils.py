"""
Protein Structure Analysis Utilities
====================================

Utilities for analyzing protein structures predicted by ESMFold, with a focus
on beta-hairpin detection and characterization.

Key functions:
    run_dssp_on_pdb: Run DSSP secondary structure assignment on a PDB file
    detect_hairpins: Detect beta-hairpins from DSSP output
    compute_handedness_from_structure: Determine hairpin handedness (Type I/II)
    get_CB_or_virtual: Get CB coordinates (or virtual CB for glycine)
    visualize_hairpin_handedness_from_cif_or_pdb: 3D visualization with py3Dmol
"""

import pandas as pd
import numpy as np  # used by get_CB_or_virtual / compute_handedness_from_structure below
import os
import requests
import torch
# from nnsight import NNsight
from transformers import AutoTokenizer, EsmForProteinFolding

from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37


from difflib import SequenceMatcher
import torch
import py3Dmol
from Bio import PDB
from Bio.PDB import DSSP
import tempfile
import warnings
import pandas as pd
def clean_pdb_string(pdb_string):
    """Remove non-standard records that confuse DSSP/mkdssp."""
    # PARENT (template/provenance) and REMARK 220 (experimental-technique)
    # lines are extra records some ESMFold/OpenFold PDB writers emit; mkdssp's
    # strict parser chokes on them, so they're stripped before DSSP sees the file.
    skip_prefixes = ("PARENT", "REMARK 220")
    return "\n".join(
        line for line in pdb_string.split("\n")
        if not line.startswith(skip_prefixes)
    )
# ---------- DSSP from outputs ----------
def run_dssp_from_outputs(outputs, model, batch_idx=0):
    """Save ESMFold output to a temp PDB and run DSSP once."""
    # Detach + clone so we don't hold references into the live autograd graph
    # and don't mutate the caller's `outputs` dict in place.
    detached_outputs = {
        k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
        for k, v in outputs.items()
    }

    # output_to_pdb() returns one PDB string per batch element; batch_idx
    # picks which structure in the batch to run DSSP on.
    pdb_str = model.output_to_pdb(detached_outputs)[batch_idx]
    pdb_str = clean_pdb_string(pdb_str)

    # delete=False: the file must still exist on disk after this `with` block
    # closes it, because DSSP (below) shells out to the external `mkdssp`
    # binary, which re-reads the PDB from its path rather than from memory.
    pdb_path = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False).name
    with open(pdb_path, "w") as f:
        f.write(pdb_str)

    parser = PDB.PDBParser(QUIET=True)  # QUIET suppresses minor format warnings
    structure = parser.get_structure("ESMFold", pdb_path)
    model0 = structure[0]

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB.DSSP")
            # DSSP needs both the parsed Model (for coordinate/residue access)
            # and the on-disk path (mkdssp reads the file itself).
            dssp = DSSP(model0, pdb_path, dssp="mkdssp")
    except Exception as e:
        warnings.warn(f"DSSP failed: {e}", RuntimeWarning)
        # Return a bare None (not a tuple) so detect_hairpins()'s `if dssp_df is None`
        # check below actually catches the failure. The success path returns a single
        # DataFrame, so the failure sentinel must match that arity.
        return None

    rows = []
    for key in dssp.keys():
        # DSSP dict keys are (chain_id, residue_id); residue_id follows
        # Biopython's standard (hetero_flag, resseq, insertion_code) format.
        chain_id, (hetatm_flag, resseq, icode) = key
        # dssp[key] is a tuple; index 1 = 1-letter amino acid, index 2 = the
        # raw DSSP secondary-structure code (later fields hold accessibility,
        # phi/psi, H-bond energies, etc., which we don't need here).
        aa, ss = dssp[key][1], dssp[key][2]
        # Collapse DSSP's 8-state code to 3 states: {H, G, I} are the three
        # helix flavors (alpha-, 3_10-, and pi-helix -- they differ only in
        # H-bond turn length, i->i+4/i+3/i+5) -> "H"; {E, B} are strand-like
        # (E = extended beta-ladder strand, B = isolated single-residue
        # bridge) -> "E"; everything else (turns, bends, coil/"-") -> "C".
        # This 3-state scheme is what detect_hairpins() scans for strands.
        simp = "E" if ss in ["E", "B"] else ("H" if ss in ["H", "I", "G"] else "C")
        rows.append((chain_id, resseq, aa, ss, simp))
    df = pd.DataFrame(rows, columns=["Chain", "ResNum", "AA", "SecStruct", "SimpleSS"])
    return df

def run_dssp_on_pdb(pdb_path):
    """
    Parse a PDB file, run DSSP, and return:
      - structure: Bio.PDB structure object
      - dssp_df: DataFrame with columns:
          [Chain, ResNum, AA, SecStruct, SimpleSS]
    """
    from Bio.PDB import PDBParser, DSSP
    import warnings
    import pandas as pd

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", pdb_path)
    model0 = structure[0]

    try:
        # DSSP requires both structure object + path to the PDB file
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB.DSSP")
            dssp = DSSP(model0, pdb_path, dssp="mkdssp")
    except Exception as e:
        warnings.warn(f"DSSP failed: {e}", RuntimeWarning)
        return None, None

    rows = []
    for key in dssp.keys():
        # key = (chain_id, residue_id), residue_id = (hetero_flag, resseq, icode)
        # -- Biopython's standard PDB residue-ID tuple format.
        chain_id, (hetflag, resseq, icode) = key
        aa = dssp[key][1]  # 1-letter amino acid
        ss = dssp[key][2]  # DSSP annotation (H, E, C, etc.)

        # DSSP's 8-state code collapses to 3 states here: {H, G, I} are the
        # three helix types (alpha-, 3_10-, and pi-helix -- differing only in
        # H-bond turn length, i->i+4/i+3/i+5) -> "H"; {E, B} are strand-like
        # (E = extended ladder strand, B = isolated single-residue bridge)
        # -> "E"; remaining codes (T=turn, S=bend, "-"/" "=coil) -> "C".
        # Simple 3-state mapping: E (beta), H (helix), C (coil)
        simp = (
            "E" if ss in ["E", "B"]
            else "H" if ss in ["H", "I", "G"]
            else "C"
        )

        rows.append((chain_id, resseq, aa, ss, simp))

    df = pd.DataFrame(rows, columns=["Chain", "ResNum", "AA", "SecStruct", "SimpleSS"])
    return structure, df


# ------------------------------------------------------------
# 2. Detect β-hairpins from DSSP (two adjacent beta strands)
# ------------------------------------------------------------
def detect_hairpins(outputs, model, min_len=2, max_loop=5):

    """
    Detect β-hairpins defined as:
      - Two β-strand segments ('E' in SimpleSS) of length >= min_len
      - Separated by a loop of length in [0, max_loop]
    Returns:
      hairpin_detected: bool
      hairpins: list of tuples
        (chain_id, s1_start_idx, s1_end_idx, s2_start_idx, s2_end_idx)
    """

    dssp_df = run_dssp_from_outputs(outputs, model)
    # run_dssp_from_outputs() returns a bare None on DSSP failure, so this check
    # catches it and returns gracefully instead of crashing on the None below.
    if dssp_df is None:
        print("❌ DSSP failed.")
        return None, None

    hairpins = []

    for chain_id in dssp_df["Chain"].unique():
        cdf = dssp_df[dssp_df["Chain"] == chain_id].reset_index(drop=True)

        strands = []
        start = None

        # Find continuous β-strand segments in SimpleSS
        # Run-length scan: `start` marks the first residue of the current run
        # of "E" labels; a run is only kept (appended to `strands`) once it
        # breaks AND was at least min_len long, otherwise it's discarded as
        # too short to count as a real strand.
        for i, row in cdf.iterrows():
            if row["SimpleSS"] == "E":
                if start is None:
                    start = i
            else:
                if start is not None and i - start >= min_len:
                    # i-1 because `i` is the first non-"E" residue, i.e. one
                    # past the end of the strand.
                    strands.append((start, i - 1))
                start = None

        # Handle strand that reaches to the end
        # (the loop above only closes a run when it hits a break; a run still
        # open when residues run out needs to be closed off here instead)
        if start is not None and len(cdf) - start >= min_len:
            strands.append((start, len(cdf) - 1))

        # Pair neighboring strands into hairpins
        for i in range(len(strands) - 1):
            s1_start, s1_end = strands[i]
            s2_start, s2_end = strands[i + 1]
            # Residues strictly between the two strands (0 if sequence-adjacent).
            loop_len = s2_start - s1_end - 1

            # A hairpin requires a *short* connecting loop -- this is what
            # distinguishes a hairpin (two strands joined by a tight turn)
            # from two unrelated strands that happen to both be beta but are
            # far apart in sequence.
            if 0 <= loop_len <= max_loop:
                hairpins.append((chain_id, s1_start, s1_end, s2_start, s2_end))

    hairpin_detected = (len(hairpins) > 0)
    return hairpin_detected, hairpins


# ------------------------------------------------------------
# 3. Virtual CB helper (for GLY / missing CB)
# ------------------------------------------------------------
def get_CB_or_virtual(residue):
    """
    Return CB coordinate if present.
    If CB is missing (e.g., Gly), construct a virtual CB from N, CA, C.
    Uses the standard OpenFold/ESMFold virtual CB construction.
    """
    # Glycine has no side chain beyond the alpha carbon (its "R group" is a
    # single H), so it has no CB atom in the PDB; every other standard amino
    # acid does. The virtual-CB construction below gives glycine a consistent
    # stand-in side-chain-direction vector so downstream code (e.g. handedness)
    # doesn't need a per-residue-type special case.
    if "CB" in residue:
        return residue["CB"].coord

    # Need N, CA, C to build virtual CB
    N = residue["N"].coord
    CA = residue["CA"].coord
    C = residue["C"].coord

    b = CA - N
    c = C - CA

    # Normalize
    # NOTE: possible bug -- `np` (numpy) is used from here on but is never
    # imported anywhere in this file (only torch/pandas/etc. are imported
    # above); this raises NameError: name 'np' is not defined the first time
    # a residue is missing its CB atom (e.g. any glycine).
    b = b / np.linalg.norm(b)
    c = c / np.linalg.norm(c)

    # Virtual CB direction (OpenFold coefficients)
    # Fixed empirical coefficients (OpenFold/AlphaFold idealized backbone
    # geometry) that reconstruct the Cb direction from the local backbone
    # frame: a linear combination of the N->CA and CA->C bond vectors plus
    # their cross product (the out-of-plane component, which encodes the
    # L-amino-acid chirality).
    v = -0.58273431 * b + 0.56802827 * c + 0.54067466 * np.cross(b, c)
    v = v / np.linalg.norm(v)

    CB = CA + 1.522 * v  # CA–CB bond length ~1.522 Å
    return CB


# ------------------------------------------------------------
# 4. Compute handedness from structure (PDB/CIF only)
# ------------------------------------------------------------
def compute_handedness_from_structure(structure, chain_df, s1_end_idx, s2_start_idx, eps=1e-8):
    """
    Compute hairpin handedness using:
      - residue at end of strand 1 (s1_end_idx)
      - residue at start of strand 2 (s2_start_idx)
    from the given structure and chain_df (subset of dssp_df for one chain).

    triple = (u x v) · n
      u = C1 - N1       (strand direction)
      v = CA2 - CA1     (bridge vector)
      n = CB1 - CA1     (side-chain normal / virtual CB if needed)
    """
    model = structure[0]
    chain_id = chain_df.loc[0, "Chain"]
    chain = model[chain_id]

    res1 = int(chain_df.loc[s1_end_idx, "ResNum"])
    res2 = int(chain_df.loc[s2_start_idx, "ResNum"])

    # Handle possible missing residues gracefully
    try:
        # Biopython residue IDs are (hetero_flag, seqnum, icode) tuples; " "
        # (space) means a standard amino acid with no insertion code.
        r1 = chain[(" ", res1, " ")]
        r2 = chain[(" ", res2, " ")]
    except KeyError as e:
        raise KeyError(f"Residue not found in structure for ResNum {e}") from e

    N1 = r1["N"].coord
    CA1 = r1["CA"].coord
    C1 = r1["C"].coord
    CB1 = get_CB_or_virtual(r1)
    CA2 = r2["CA"].coord

    u = C1 - N1
    v = CA2 - CA1
    n = CB1 - CA1

    # Handedness = sign of the scalar triple product (u x v) . n. `u` anchors
    # the local backbone/strand direction at the last residue of strand 1;
    # `v` is the vector that bridges across the turn to the first residue of
    # strand 2; `n` points from Ca1 toward its side chain (Cb), giving a fixed
    # chirality reference. u x v is normal to the plane spanned by the strand
    # and the bridge, so dotting with n is positive when the bridge crosses to
    # the same side of the strand as the side chain, and negative on the
    # opposite side -- these two cases correspond to the two classic
    # cross-strand turn geometries (mirror images of each other, commonly
    # described as Type I vs Type II / "left-handed" vs "right-handed"
    # hairpin turns).
    triple = np.dot(np.cross(u, v), n)
    # Dividing by the product of vector norms removes the (largely
    # bond-length-driven) magnitude of u/v/n so `mag` mainly reflects the
    # angular/chirality relationship rather than raw distances; eps guards
    # against division by zero if any vector were degenerate.
    denom = (np.linalg.norm(u) * np.linalg.norm(v) * np.linalg.norm(n)) + eps
    mag = triple / denom

    return np.sign(mag), mag


# ------------------------------------------------------------
# 5. Get vectors for visualization (u, v, n, anchor atoms)
# ------------------------------------------------------------
def get_handedness_vectors_from_structure(structure, chain_df, s1_end_idx, s2_start_idx):
    """
    Build vectors u, v, n and anchor points from the structure.
    Returns a dict with:
      - "u", "v", "n": {start, end}
      - "points": {N1, CA1, C1, CB1, CA2}
    """
    model = structure[0]
    chain_id = chain_df.loc[0, "Chain"]
    chain = model[chain_id]

    res1 = int(chain_df.loc[s1_end_idx, "ResNum"])
    res2 = int(chain_df.loc[s2_start_idx, "ResNum"])

    r1 = chain[(" ", res1, " ")]
    r2 = chain[(" ", res2, " ")]

    N1 = r1["N"].coord
    CA1 = r1["CA"].coord
    C1 = r1["C"].coord
    CB1 = get_CB_or_virtual(r1)  # falls back to a constructed virtual CB for Gly
    CA2 = r2["CA"].coord

    # Same u/v/n definitions as compute_handedness_from_structure (kept in
    # sync manually) -- this function only builds arrows for the viewer, it
    # does not recompute the handedness sign itself.
    u = C1 - N1
    v = CA2 - CA1
    n = CB1 - CA1

    # Purely cosmetic: stretches the (bond-length-scale) vectors so they
    # render as visible arrows in py3Dmol; doesn't affect any handedness math.
    scale = 2.0

    arrows = {
        "u": {"start": N1, "end": N1 + scale * u},
        "v": {"start": CA1, "end": CA1 + scale * v},
        "n": {"start": CA1, "end": CA1 + scale * n},
        "points": {
            "N1": N1,
            "CA1": CA1,
            "C1": C1,
            "CB1": CB1,
            "CA2": CA2,
        },
    }
    return arrows


# ------------------------------------------------------------
# 6. Main visualization function
# ------------------------------------------------------------
def visualize_hairpin_handedness_from_cif_or_pdb(
    cif_or_pdb: str,
    is_cif: bool,
    true_hairpin_seq: str,
    width: int = 600,
    height: int = 600,
):
    """
    From a CIF file:
      - run DSSP
      - detect β-hairpins
      - pick the best-matching hairpin to `true_hairpin_seq`
      - compute handedness from structure
      - build a py3Dmol view with:
          * strands & loop colored
          * u, v, n vectors
          * anchor atoms

    Returns:
      metric_dict, py3Dmol_view

    metric_dict keys:
      - handedness (sign)
      - magnitude (float)
      - similarity (seq similarity to true_hairpin_seq)
      - matched_sequence (full hairpin sequence)
      - hairpin_indices: (chain_id, s1s, s1e, s2s, s2e)
    """
    # --- run DSSP ---
    if is_cif:
        # NOTE: possible bug -- run_dssp_on_cif() is called here but is not
        # defined or imported anywhere in this module (only run_dssp_on_pdb
        # is defined below); calling this with is_cif=True raises
        # NameError: name 'run_dssp_on_cif' is not defined.
        structure, dssp_df = run_dssp_on_cif(cif_or_pdb)
        if dssp_df is None:
            print("❌ DSSP failed.")
            return None, None
    else:
        structure, dssp_df = run_dssp_on_pdb(cif_or_pdb)

    # NOTE: possible bug -- detect_hairpins() is defined above as
    # detect_hairpins(outputs, model, min_len=2, max_loop=5) (it internally
    # re-runs DSSP from raw model outputs) and returns a (bool, list) tuple.
    # Calling it here with a single `dssp_df` argument would raise TypeError
    # (missing required 'model' argument); even if the call were fixed, the
    # `len(hairpins) == 0` check and the `for chain_id, ... in hairpins`
    # unpacking below both assume `hairpins` is already the plain list of
    # hairpin tuples, not the (bool, list) pair detect_hairpins() returns.
    hairpins = detect_hairpins(dssp_df)
    if len(hairpins) == 0:
        print("❌ No β-hairpins detected")
        return None, None

    # Pick best matching hairpin by sequence similarity
    best = {
        "similarity": -1.0,  # lower than any possible ratio() below, so the
                              # first candidate always replaces this sentinel
        "hairpin": None,
        "chain_df": None,
        "handed": None,
        "magnitude": None,
        "matched_sequence": None,
    }

    # A structure/chain can contain multiple candidate hairpins; DSSP-based
    # detection alone can't tell which one corresponds to the hairpin of
    # interest, so we score every candidate against the known true sequence
    # and keep the closest match.
    for chain_id, s1s, s1e, s2s, s2e in hairpins:
        cdf = dssp_df[dssp_df["Chain"] == chain_id].reset_index(drop=True)

        # hairpin sequence from strand1 start to strand2 end
        hairpin_seq = "".join(cdf.loc[s1s:s2e, "AA"].tolist())
        # difflib ratio(): normalized (0-1) fuzzy string-similarity score, used
        # instead of exact matching to tolerate small differences (e.g.
        # off-by-one strand-boundary detection).
        sim = SequenceMatcher(None, true_hairpin_seq, hairpin_seq).ratio()

        hand, mag = compute_handedness_from_structure(structure, cdf, s1e, s2s)

        if sim > best["similarity"]:
            best.update({
                "hairpin": (chain_id, s1s, s1e, s2s, s2e),
                "chain_df": cdf,
                "handed": hand,
                "magnitude": mag,
                "matched_sequence": hairpin_seq,
                "similarity": sim,
            })

    if best["hairpin"] is None:
        print("❌ Failed to pick a hairpin")
        return None, None

    chain_id, s1s, s1e, s2s, s2e = best["hairpin"]
    cdf = best["chain_df"]

    # --- vectors for visualization ---
    arrows = get_handedness_vectors_from_structure(structure, cdf, s1e, s2s)

    # --- convert CIF → PDB string for py3Dmol visualization ---
    # py3Dmol's addModel(..., "pdb") below only accepts PDB-format text, so
    # the (possibly CIF-parsed) Structure object is re-serialized to PDB and
    # read back in, regardless of whether the original input was CIF or PDB.
    # PDBIO is accessed via the `PDB` module namespace imported at the top
    # (`from Bio import PDB`); `PDBIO` on its own is not imported into scope.
    io = PDB.PDBIO()
    io.set_structure(structure)
    tmp_pdb = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False).name
    io.save(tmp_pdb)
    with open(tmp_pdb, "r") as f:
        pdb_str = f.read()

    # --- py3Dmol viewer ---
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_str, "pdb")
    view.setStyle({"cartoon": {"color": "lightgrey"}})

    # Color strands + loop
    # s1s/s1e/s2s/s2e are positional (0-indexed, DataFrame-row) indices into
    # cdf, not PDB residue numbers, so look up the actual ResNum values for
    # py3Dmol's `resi` selector; +1 on each end index because slicing is
    # exclusive of the stop but s1e/s2e are meant to be inclusive.
    resnums = cdf["ResNum"].tolist()
    s1_res = resnums[s1s:s1e + 1]
    s2_res = resnums[s2s:s2e + 1]
    loop_res = resnums[s1e + 1:s2s]  # residues strictly between the two strands

    for r in s1_res:
        view.setStyle({"resi": str(r)}, {"cartoon": {"color": "blue"}})
    for r in s2_res:
        view.setStyle({"resi": str(r)}, {"cartoon": {"color": "red"}})
    for r in loop_res:
        view.setStyle({"resi": str(r)}, {"cartoon": {"color": "orange"}})

    # Add arrows
    # (coordinates come back as numpy float32 scalars from Biopython; py3Dmol's
    # JS-facing API needs plain JSON-serializable floats, hence float(...))
    def add_arrow(start, end, color):
        """Draw a single py3Dmol arrow from `start` to `end` in the given color."""
        view.addArrow({
            "start": {"x": float(start[0]), "y": float(start[1]), "z": float(start[2])},
            "end":   {"x": float(end[0]),   "y": float(end[1]),   "z": float(end[2])},
            "color": color,
            "radius": 0.25,
        })

    # This blue/green/yellow legend is for the u/v/n geometry vectors only,
    # independent of the blue/red strand-cartoon coloring set above.
    add_arrow(arrows["u"]["start"], arrows["u"]["end"], "blue")
    add_arrow(arrows["v"]["start"], arrows["v"]["end"], "green")
    add_arrow(arrows["n"]["start"], arrows["n"]["end"], "yellow")

    # Anchor atoms
    for name, p in arrows["points"].items():
        view.addSphere({
            "center": {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
            "radius": 0.6,
            "color": "white",
        })

    view.zoomTo()

    metric = {
        "handedness": best["handed"],
        "magnitude": best["magnitude"],
        "similarity": best["similarity"],
        "matched_sequence": best["matched_sequence"],
        "hairpin_indices": (chain_id, s1s, s1e, s2s, s2e),
    }

    return metric, view
