#!/usr/bin/env python
"""
export_viz_bundle.py -- Export one ESMFold forward pass into a compact JSON
bundle for the llm-viz "ESMFold" visualization (sibling repo ../llm-viz).

What it does
------------
Runs facebook/esmfold_v1 on a single sequence, captures intermediate
activations with the repo's existing hook API (src/utils/representation_utils),
and writes a small JSON file the browser front-end can load directly.

The full activation tensors are far too big to ship to a browser
(esm: 36x L x 2560, s: 48x L x 1024, z: 48x L x L x 128). So we export
*summaries* that are visually meaningful:

  - esm.layer_res_norm : per (layer, residue) L2 norm            [36][L]
  - trunk.s_res_norm   : per (block, residue) L2 norm of s       [48][L]
  - trunk.s_slab       : downsampled s features for a few blocks [nb][L][ds]
  - trunk.z_norm       : per block, L x L norm over 128 channels [48][L][L]
  - trunk.pair_bias    : z->s attention bias (mean over heads)   [nb][L][L]
  - ipa.block_res_norm : per structure-module block, per residue [8][L]
  - structure          : pdb string, plddt, pae, ptm, CA coords

Usage
-----
    conda activate protein_folding
    cd ESM_Internal
    PYTHONPATH=src python scripts/export_viz_bundle.py \
        --name villin_hp36 \
        --sequence MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF

Defaults to villin HP36 (36 residues; small, fast, real fold) and writes to
../llm-viz/public/esmfold/data/<name>.json plus refreshes manifest.json.
"""

import argparse
import json
import os
import sys
import datetime
from pathlib import Path

import numpy as np
import torch

# --- make `import src.*` work regardless of cwd -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.model_utils import load_esmfold  # noqa: E402
from src.utils.representation_utils import (  # noqa: E402
    CollectedRepresentations,
    ESMEncoderHooks,
    TrunkHooks,
    IPAHooks,
)

# GB1 beta-hairpin peptide: the ESM_Internal steering-demo input.
# Strand1 EWTYD, turn DATK, Strand2 TFTVT (16 residues).
DEFAULT_SEQUENCE = "GEWTYDDATKTFTVTE"
DEFAULT_NAME = "gb1_hairpin"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _f(t: torch.Tensor) -> torch.Tensor:
    """Detach -> cpu -> float32 (fp8/bf16 activations have no numpy dtype)."""
    return t.detach().to("cpu").float()


def rnd(a, n=4):
    """Round a numpy array/scalar and return plain python lists/floats."""
    arr = np.asarray(a, dtype=np.float64)
    arr = np.round(arr, n)
    return arr.tolist()


def align_esm_to_residues(t: torch.Tensor, L: int) -> torch.Tensor:
    """ESM hidden states carry a leading BOS token (add_special_tokens=False
    still lets ESMFold prepend BOS internally), so the sequence axis is L+1.
    Drop the leading token(s) and keep the last L positions."""
    s = t.shape[1]
    if s == L:
        return t[:, :L]
    return t[:, s - L:]  # drop leading BOS (and any extra) -> last L residues


# ---------------------------------------------------------------------------
# collection: one forward pass with all the hooks we need
# ---------------------------------------------------------------------------
def collect(model, tokenizer, device, sequence, num_recycles):
    collector = CollectedRepresentations()

    esm_hooks = ESMEncoderHooks(model.esm, collector)
    trunk_hooks = TrunkHooks(model.trunk, collector)
    ipa_hooks = IPAHooks(model.trunk.structure_module, collector)

    esm_hooks.register()
    # collect_pair2seq=True gives us the z->s attention bias [1,L,L,heads]
    trunk_hooks.register(collect_s=True, collect_z=True, collect_pair2seq=True)
    ipa_hooks.register()
    ipa_hooks.reset()

    try:
        with torch.no_grad():
            inputs = tokenizer(
                sequence, return_tensors="pt", add_special_tokens=False
            ).to(device)
            outputs = model(**inputs, num_recycles=num_recycles)
    finally:
        esm_hooks.remove()
        trunk_hooks.remove()
        ipa_hooks.remove()

    return outputs, collector


# ---------------------------------------------------------------------------
# turn collected tensors into the compact JSON bundle
# ---------------------------------------------------------------------------
def build_bundle(name, sequence, outputs, collector, model, num_recycles,
                 slab_blocks, slab_ds, bias_blocks):
    L = len(sequence)

    # ---- ESM-2 backbone: per (layer, residue) activation magnitude ----------
    esm_ids = sorted(collector.esm_layers.keys())
    layer_res_norm = []
    for i in esm_ids:
        t = align_esm_to_residues(_f(collector.esm_layers[i]), L)[0]  # [L, 2560]
        layer_res_norm.append(rnd(t.norm(dim=-1).numpy()))

    combine_weights = None
    w = getattr(model, "esm_s_combine", None)
    if w is not None:
        try:
            combine_weights = rnd(_f(w).softmax(0).flatten().numpy())
        except Exception:
            combine_weights = None

    # ---- folding trunk: s magnitude + downsampled s slab + z pair norm ------
    blk_ids = sorted(collector.s_blocks.keys())
    s_res_norm = []
    for i in blk_ids:
        s = _f(collector.s_blocks[i])[0]  # [L, 1024]
        s_res_norm.append(rnd(s.norm(dim=-1).numpy()))

    s_slab = None
    if slab_blocks:
        want = [b for b in slab_blocks if b in collector.s_blocks]
        data = []
        for b in want:
            s = _f(collector.s_blocks[b])[0]  # [L, 1024]
            ds = s[:, ::max(1, s.shape[1] // slab_ds)][:, :slab_ds]  # [L, ds]
            data.append(rnd(ds.numpy(), 3))
        s_slab = {"blocks": want, "feat": slab_ds, "data": data}

    z_ids = sorted(collector.z_blocks.keys())
    z_norm = []
    for i in z_ids:
        z = _f(collector.z_blocks[i])[0]  # [L, L, 128]
        z_norm.append(rnd(z.norm(dim=-1).numpy(), 3))

    pair_bias = None
    if bias_blocks and collector.pair2seq_biases:
        want = [b for b in bias_blocks if b in collector.pair2seq_biases]
        data = []
        for b in want:
            pb = _f(collector.pair2seq_biases[b])[0]  # [L, L, heads]
            data.append(rnd(pb.mean(dim=-1).numpy(), 3))
        pair_bias = {"blocks": want, "data": data}

    # ---- structure module IPA magnitude ------------------------------------
    ipa_ids = sorted(collector.ipa_outputs.keys())
    ipa_res_norm = [rnd(_f(collector.ipa_outputs[i])[0].norm(dim=-1).numpy())
                    for i in ipa_ids] or None

    # ---- final structure ---------------------------------------------------
    pdb = model.output_to_pdb(outputs)[0]

    # outputs.plddt is per-atom [B, L, 37] (atom37). Take the CA atom
    # (atom37 index 1) as the per-residue confidence -- matches the CA
    # b-factor written into the PDB. Values are in [0, 1].
    p = _f(outputs.plddt)
    if p.ndim == 3:        # [B, L, 37]
        p = p[0, :, 1]     # CA
    elif p.ndim == 2:      # [B, L] or [L, 37]
        p = p[0] if p.shape[0] == 1 else p[:, 1]
    plddt = rnd(p.numpy())

    ptm = None
    if getattr(outputs, "ptm", None) is not None:
        ptm = float(outputs.ptm.item())

    pae = None
    pae_t = getattr(outputs, "predicted_aligned_error", None)
    if pae_t is not None:
        pae = rnd(_f(pae_t)[0].numpy(), 2)

    # CA coords from atom14 (index 1 = CA), last refinement iteration
    ca_xyz = None
    pos = getattr(outputs, "positions", None)
    if pos is not None:
        ca = _f(pos)[-1, 0, :, 1, :]  # [L, 3]
        ca_xyz = rnd(ca.numpy(), 3)

    # infer head count for metadata
    try:
        heads_seq = int(model.trunk.blocks[0].seq_attention.num_heads)
    except Exception:
        heads_seq = None

    return {
        "meta": {
            "name": name,
            "sequence": sequence,
            "length": L,
            "source": "esmfold_v1",
            "num_recycles": num_recycles,
            "dims": {"esm": 2560, "s": 1024, "z": 128, "ipa": 384},
            "counts": {
                "esm_layers": len(esm_ids),
                "trunk_blocks": len(blk_ids),
                "ipa_blocks": len(ipa_ids),
                "heads_seq": heads_seq,
            },
            "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "esm": {"layer_res_norm": layer_res_norm, "combine_weights": combine_weights},
        "trunk": {
            "s_res_norm": s_res_norm,
            "s_slab": s_slab,
            "z_norm": z_norm,
            "pair_bias": pair_bias,
        },
        "ipa": {"block_res_norm": ipa_res_norm},
        "structure": {
            "pdb": pdb,
            "plddt": plddt,
            "pae": pae,
            "ptm": ptm,
            "ca_xyz": ca_xyz,
        },
    }


def refresh_manifest(data_dir: Path):
    entries = []
    for p in sorted(data_dir.glob("*.json")):
        if p.name == "manifest.json":
            continue
        try:
            with open(p) as fh:
                meta = json.load(fh).get("meta", {})
            entries.append({
                "name": meta.get("name", p.stem),
                "file": p.name,
                "length": meta.get("length"),
                "source": meta.get("source"),
            })
        except Exception:
            continue
    with open(data_dir / "manifest.json", "w") as fh:
        json.dump({"proteins": entries}, fh, indent=2)
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=DEFAULT_NAME)
    ap.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="fp32",
                    help="fp32 (default, robust) | esm_bf16 | esm_fp8 | esm_half")
    ap.add_argument("--num-recycles", type=int, default=0)
    ap.add_argument("--slab-blocks", type=int, nargs="*",
                    default=[0, 8, 16, 24, 32, 40, 47],
                    help="trunk blocks to export a downsampled s feature slab for")
    ap.add_argument("--slab-ds", type=int, default=64,
                    help="number of downsampled s features in the slab")
    ap.add_argument("--bias-blocks", type=int, nargs="*",
                    default=[0, 8, 16, 24, 32, 40, 47],
                    help="trunk blocks to export the z->s attention bias for")
    ap.add_argument("--out", default=None,
                    help="output dir (default ../llm-viz/public/esmfold/data)")
    args = ap.parse_args()

    seq = args.sequence.strip().upper()
    L = len(seq)
    print(f"[export] {args.name}: L={L}  precision={args.precision}  "
          f"recycles={args.num_recycles}")

    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        print(f"[export] CUDA unavailable, using {device}")

    model, tokenizer = load_esmfold(device, precision=args.precision)

    print("[export] running forward pass + hooks ...")
    outputs, collector = collect(model, tokenizer, device, seq, args.num_recycles)

    print("[export] building bundle ...")
    bundle = build_bundle(
        args.name, seq, outputs, collector, model, args.num_recycles,
        args.slab_blocks, args.slab_ds, args.bias_blocks,
    )

    out_dir = Path(args.out) if args.out else (
        REPO_ROOT.parent / "llm-viz" / "public" / "esmfold" / "data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name}.json"
    with open(out_path, "w") as fh:
        json.dump(bundle, fh, separators=(",", ":"))
    size_kb = out_path.stat().st_size / 1024
    print(f"[export] wrote {out_path}  ({size_kb:.0f} KB)")

    entries = refresh_manifest(out_dir)
    print(f"[export] manifest: {[e['name'] for e in entries]}")
    m = bundle["meta"]["counts"]
    print(f"[export] done. esm_layers={m['esm_layers']} trunk_blocks={m['trunk_blocks']} "
          f"ipa_blocks={m['ipa_blocks']} ptm={bundle['structure']['ptm']}")


if __name__ == "__main__":
    main()
