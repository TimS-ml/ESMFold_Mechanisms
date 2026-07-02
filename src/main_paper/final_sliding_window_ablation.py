"""
Sliding Window Patching Ablation
================================

Systematically ablates the contribution of sequence (s) vs pairwise (z)
representations by patching with sliding windows across trunk blocks.

Tests three patch modes:
- sequence: Patch only s representation
- pairwise: Patch only z representation
- combined: Patch both s and z

This reveals that pairwise representations carry the critical structural
information, with sequence representations playing a supporting role.

Usage:
    python final_sliding_window_ablation.py \
        --ablation_csv successful_cases.csv \
        --n_sequence_cases 400 \
        --n_pairwise_cases 200 \
        --output_dir results/
"""

import argparse
import os
import types
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

# Suppress BioPython DSSP warnings
warnings.filterwarnings("ignore", message=".*mmCIF.*")
warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB.DSSP")

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from transformers import EsmForProteinFolding, AutoTokenizer
from transformers.models.esm.modeling_esmfold import (
    categorical_lddt,
    EsmFoldingTrunk,
    EsmForProteinFoldingOutput
)
from transformers.models.esm.openfold_utils import (
    compute_predicted_aligned_error,
    compute_tm,
    make_atom14_masks,
    to_pdb,
)
from transformers.utils import ContextManagers, ModelOutput
from src.utils.trunk_utils import detect_hairpins
from src.utils.model_utils import load_esmfold
# ============================================================================
# Alpha Helix Content Calculation
# ============================================================================

def compute_alpha_helix_content(pdb_string: str) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    """
    Compute the percentage of residues in alpha helix from a PDB string.
    Returns (helix_count, total_count, helix_percentage) or (None, None, None) if DSSP fails.
    
    Requires: trunk_utils.py with run_dssp_on_pdb function
    """
    import tempfile
    
    # Guarded/lazy import so this module can still be imported in environments
    # without the DSSP-related utilities installed (helix computation is optional).
    try:
        from src.utils.trunk_utils import run_dssp_on_pdb
    except ImportError:
        print("Warning: utils.trunk_utils not found, cannot compute helix content")
        return None, None, None
    
    # DSSP needs a real file path, so the in-memory PDB string is written to a temp
    # file first. delete=False keeps the file on disk after this `with` block exits
    # (required on some platforms to let DSSP reopen it); it is removed manually below.
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode='w') as f:
        f.write(pdb_string)
        pdb_path = f.name
    
    try:
        structure, dssp_df = run_dssp_on_pdb(pdb_path)
        if dssp_df is None:
            return None, None, None
        
        # Count helix residues (H, G, I are helix types in DSSP)
        # SimpleSS == "H" covers all helix types
        total_residues = len(dssp_df)
        helix_residues = len(dssp_df[dssp_df["SimpleSS"] == "H"])
        helix_percentage = (helix_residues / total_residues * 100) if total_residues > 0 else 0
        
        return helix_residues, total_residues, helix_percentage
    
    except Exception as e:
        print(f"DSSP failed: {e}")
        return None, None, None
    finally:
        # Clean up temp file
        # Runs regardless of whether DSSP succeeded or raised above; the bare
        # except suppresses cleanup errors (e.g. file already gone) so they
        # don't mask/replace whatever exception was being handled.
        import os
        try:
            os.unlink(pdb_path)
        except:
            pass


# ============================================================================
# Custom Output Class with Block Representations
# ============================================================================

@dataclass
class NewEsmForProteinFoldingOutput(ModelOutput):
    """Extended output class that includes per-block representations."""
    # All fields below mirror `EsmForProteinFoldingOutput` from transformers exactly,
    # EXCEPT `s_s_list`/`s_z_list` at the bottom, which are new: they hold the
    # sequence/pairwise representation after every trunk block (not just the final
    # one), populated by the monkey-patched trunk forwards below.
    frames: Optional[torch.FloatTensor] = None
    sidechain_frames: Optional[torch.FloatTensor] = None
    unnormalized_angles: Optional[torch.FloatTensor] = None
    angles: Optional[torch.FloatTensor] = None
    positions: Optional[torch.FloatTensor] = None
    states: Optional[torch.FloatTensor] = None
    s_s: Optional[torch.FloatTensor] = None
    s_z: Optional[torch.FloatTensor] = None
    distogram_logits: Optional[torch.FloatTensor] = None
    lm_logits: Optional[torch.FloatTensor] = None
    aatype: Optional[torch.FloatTensor] = None
    atom14_atom_exists: Optional[torch.FloatTensor] = None
    residx_atom14_to_atom37: Optional[torch.FloatTensor] = None
    residx_atom37_to_atom14: Optional[torch.FloatTensor] = None
    atom37_atom_exists: Optional[torch.FloatTensor] = None
    residue_index: Optional[torch.FloatTensor] = None
    lddt_head: Optional[torch.FloatTensor] = None
    plddt: Optional[torch.FloatTensor] = None
    ptm_logits: Optional[torch.FloatTensor] = None
    ptm: Optional[torch.FloatTensor] = None
    aligned_confidence_probs: Optional[torch.FloatTensor] = None
    predicted_aligned_error: Optional[torch.FloatTensor] = None
    max_predicted_aligned_error: Optional[torch.FloatTensor] = None
    s_s_list: Optional[List[torch.FloatTensor]] = None
    s_z_list: Optional[List[torch.FloatTensor]] = None


# ============================================================================
# Monkey-patched Forward Functions
# ============================================================================
# The experiments below need to change what happens INSIDE a trunk block's
# forward pass (e.g. skip one specific cross-stream term, or splice in donor
# activations mid-block) -- not just read/modify tensors flowing in and out of
# it. Standard PyTorch forward hooks (`register_forward_hook`) only see a
# module's final input/output, not its internal control flow, so they can't
# express "skip this one line of math but run everything else normally".
# Instead, this file replaces the bound `forward` method on specific
# `nn.Module` instances at runtime ("monkey-patching"):
#
#     block.forward = types.MethodType(new_forward_fn, block)
#
# `types.MethodType(fn, obj)` binds a plain function `fn(self, ...)` to the
# instance `obj`, giving a bound method equivalent to `obj.fn(...)`. Assigning
# it to `obj.forward` creates an INSTANCE attribute that shadows the class's
# own `forward` method. Since `nn.Module.__call__` internally dispatches to
# `self.forward(...)`, calling `obj(...)` afterwards runs `new_forward_fn`
# instead of the original class method -- with no change to `obj`'s
# parameters/weights, and no effect on any other (unpatched) instance of the
# same class. Below, `collect_block_representations`/`patch_s_representations_in_trunk`
# stand in for `EsmFoldingTrunk.forward`, `return_block_representations`/
# `high_forward_pass` stand in for `EsmForProteinFolding.forward`, and
# `ablate_bridges` (further down) stands in for
# `EsmFoldTriangularSelfAttentionBlock.forward` -- each written as a plain
# function taking `self` as its first argument so it can be bound this way.
# ============================================================================

def collect_block_representations(self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles):
    """
    Modified trunk forward that collects representations from each block.
    """
    device = seq_feats.device
    s_s_0 = seq_feats  # recycle-0 (original) sequence repr, kept separate so
    s_z_0 = pair_feats  # each recycle iteration below can add it back in fresh

    if no_recycles is None:
        no_recycles = self.config.max_recycles
    else:
        if no_recycles < 0:
            raise ValueError("Number of recycles must not be negative.")
        no_recycles += 1  # first "recycle" is just the initial forward pass

    def trunk_iter(s, z, residx, mask):
        """Run every trunk block once (in order), recording each block's (s, z) output."""
        z = z + self.pairwise_positional_embedding(residx, mask=mask)
        s_s_list = []
        s_z_list = []
        for block in self.blocks:
            s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size)
            # Unlike the stock EsmFoldingTrunk.forward (which only keeps the
            # final s/z), snapshot every block's output here -- this is exactly
            # the per-block "donor" activation history later grafted into a
            # different sequence's forward pass at a matching block index.
            s_s_list.append(s)
            s_z_list.append(z)
        return s, z, s_s_list, s_z_list

    s_s = s_s_0
    s_z = s_z_0

    recycle_s = torch.zeros_like(s_s)
    recycle_z = torch.zeros_like(s_z)
    recycle_bins = torch.zeros(*s_z.shape[:-1], device=device, dtype=torch.int64)

    for recycle_idx in range(no_recycles):
        with ContextManagers([] if recycle_idx == no_recycles - 1 else [torch.no_grad()]):
            recycle_s = self.recycle_s_norm(recycle_s.detach()).to(device)
            recycle_z = self.recycle_z_norm(recycle_z.detach()).to(device)
            recycle_z += self.recycle_disto(recycle_bins.detach()).to(device)

            # NOTE: sequence_list/pairwise_list are overwritten every recycle
            # iteration, so only the LAST recycle's per-block snapshots survive
            # past this loop. Harmless here since every caller in this script
            # passes num_recycles=0 (-> no_recycles=1, a single iteration).
            s_s, s_z, sequence_list, pairwise_list = trunk_iter(
                s_s_0 + recycle_s, s_z_0 + recycle_z, residx, mask
            )

            structure = self.structure_module(
                {"single": self.trunk2sm_s(s_s), "pair": self.trunk2sm_z(s_z)},
                true_aa,
                mask.float(),
            )

            recycle_s = s_s
            recycle_z = s_z
            # 3.375 / 21.375 (Angstroms) are the min/max Cbeta-Cbeta distance-bin
            # edges used to discretize the predicted coordinates into
            # `self.recycle_bins` (=15) bins for the next recycle's distance
            # conditioning -- identical constants to the stock trunk.
            recycle_bins = EsmFoldingTrunk.distogram(
                structure["positions"][-1][:, :, :3],
                3.375,
                21.375,
                self.recycle_bins,
            )

    structure["s_s"] = s_s
    structure["s_z"] = s_z
    # Attach the per-block lists onto the returned dict so they survive being
    # wrapped into NewEsmForProteinFoldingOutput by the caller (return_block_representations).
    structure['s_s_list'] = sequence_list
    structure['s_z_list'] = pairwise_list

    return structure


def return_block_representations(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    masking_pattern: Optional[torch.Tensor] = None,
    num_recycles: Optional[int] = None,
    output_hidden_states: Optional[bool] = False,
) -> NewEsmForProteinFoldingOutput:
    """Modified forward that returns block-level representations."""
    # This is a near-verbatim clone of the stock EsmForProteinFolding.forward
    # (ESM backbone -> s/z init -> trunk -> output heads). The only functional
    # differences from stock are: (1) "s_s_list"/"s_z_list" are kept in the
    # filtered structure dict below, and (2) the result is wrapped in
    # NewEsmForProteinFoldingOutput instead of EsmForProteinFoldingOutput.
    # Everything else must stay identical so this still IS a normal forward
    # pass, just with extra per-block activations exposed. Must be paired with
    # `model.trunk.forward = types.MethodType(collect_block_representations, ...)`,
    # since that's the trunk implementation that actually produces s_s_list/s_z_list.
    cfg = self.config.esmfold_config

    aa = input_ids
    B = aa.shape[0]
    L = aa.shape[1]
    device = input_ids.device
    
    if attention_mask is None:
        attention_mask = torch.ones_like(aa, device=device)
    if position_ids is None:
        position_ids = torch.arange(L, device=device).expand_as(input_ids)

    esmaa = self.af2_idx_to_esm_idx(aa, attention_mask)

    if masking_pattern is not None:
        masked_aa, esmaa, mlm_targets = self.bert_mask(aa, esmaa, attention_mask, masking_pattern)
    else:
        masked_aa = aa
        mlm_targets = None

    esm_s = self.compute_language_model_representations(esmaa)
    esm_s = esm_s.to(self.esm_s_combine.dtype)  # cast ESM backbone output back to the trunk's (fp32) dtype

    if cfg.esm_ablate_sequence:
        # Stock ESMFold config flag that zeroes the WHOLE ESM sequence signal --
        # a coarser, model-level ablation unrelated to this file's own
        # block-level bridge ablation (ablate_bridges) below.
        esm_s = esm_s * 0

    esm_s = esm_s.detach()  # ESM backbone is frozen; stop gradients (no-op at inference)
    # Learned per-layer softmax weights linearly combine the stacked per-layer
    # ESM-2 hidden states into a single c_s-width sequence representation.
    esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
    s_s_0 = self.esm_s_mlp(esm_s)
    # Pairwise repr always starts at all-zeros: ESM only supplies a per-residue
    # signal, so the trunk blocks are what first build structure into z from s.
    s_z_0 = s_s_0.new_zeros(B, L, L, cfg.trunk.pairwise_state_dim)

    if self.config.esmfold_config.embed_aa:
        s_s_0 += self.embedding(masked_aa)

    structure: dict = self.trunk(
        s_s_0, s_z_0, aa, position_ids, attention_mask, no_recycles=num_recycles
    )
    
    # Keep only the keys downstream code needs -- same filtering as stock code,
    # with "s_s_list"/"s_z_list" added so the per-block history survives.
    structure = {
        k: v for k, v in structure.items()
        if k in [
            "s_z", "s_s", "frames", "sidechain_frames", "unnormalized_angles",
            "angles", "positions", "states", "s_s_list", "s_z_list"
        ]
    }

    if mlm_targets:
        structure["mlm_targets"] = mlm_targets

    # --- From here down, identical to stock EsmForProteinFolding.forward: ---
    # distogram/LM/pTM/pLDDT output heads applied to the trunk's final s_s/s_z.
    disto_logits = self.distogram_head(structure["s_z"])
    disto_logits = (disto_logits + disto_logits.transpose(1, 2)) / 2
    structure["distogram_logits"] = disto_logits

    lm_logits = self.lm_head(structure["s_s"])
    structure["lm_logits"] = lm_logits

    structure["aatype"] = aa
    make_atom14_masks(structure)
    
    for k in ["atom14_atom_exists", "atom37_atom_exists"]:
        structure[k] *= attention_mask.unsqueeze(-1)
    structure["residue_index"] = position_ids

    lddt_head = self.lddt_head(structure["states"]).reshape(
        structure["states"].shape[0], B, L, -1, self.lddt_bins
    )
    structure["lddt_head"] = lddt_head
    plddt = categorical_lddt(lddt_head[-1], bins=self.lddt_bins)
    structure["plddt"] = plddt

    ptm_logits = self.ptm_head(structure["s_z"])
    structure["ptm_logits"] = ptm_logits
    structure["ptm"] = compute_tm(ptm_logits, max_bin=31, no_bins=self.distogram_bins)
    structure.update(compute_predicted_aligned_error(ptm_logits, max_bin=31, no_bins=self.distogram_bins))

    return NewEsmForProteinFoldingOutput(**structure)


#Patch block

# `ablate_bridges` is a drop-in replacement for the stock
# `EsmFoldTriangularSelfAttentionBlock.forward` (same validation + math as the
# original, copied below) with two added boolean switches that let a caller
# selectively CUT one or both of the two places where information crosses
# between the sequence stream (s) and the pairwise stream (z) inside this one
# block, while leaving each stream's own internal computation untouched:
#   - ablate_pair_to_seq=True: skip `pair_to_sequence(pairwise_state)` (the
#     z-derived attention bias normally fed into s's self-attention). s's
#     self-attention this block then sees NO information from z at all.
#   - ablate_seq_to_pair=True: skip adding `sequence_to_pair(sequence_state)`
#     into z, leaving z completely unchanged by s this block.
# s's self-attention/MLP and z's triangular multiplicative updates/triangular
# attention/MLP (each purely intra-stream, not dependent on the other stream)
# still run normally regardless of these flags -- only the two CROSS-stream
# coupling terms are affected. With both flags at their default (False), this
# function is numerically identical to the original block forward, which is
# what makes it safe to monkey-patch onto EVERY block up front (see
# `run_ablation_experiment` below) and only "activate" ablation selectively,
# per call, via these extra kwargs (see `patch_s_representations_in_trunk`).
def ablate_bridges(self, sequence_state, pairwise_state, mask=None, chunk_size=None, ablate_pair_to_seq = False, ablate_seq_to_pair = False, **__kwargs):
        """
        Inputs:
          sequence_state: B x L x sequence_state_dim pairwise_state: B x L x L x pairwise_state_dim mask: B x L boolean
          tensor of valid positions

        Output:
          sequence_state: B x L x sequence_state_dim pairwise_state: B x L x L x pairwise_state_dim
        """
        if len(sequence_state.shape) != 3:
            raise ValueError(f"`sequence_state` should be a 3d-tensor, got {len(sequence_state.shape)} dims.")
        if len(pairwise_state.shape) != 4:
            raise ValueError(f"`pairwise_state` should be a 4d-tensor, got {len(pairwise_state.shape)} dims.")
        if mask is not None and len(mask.shape) != 2:
            raise ValueError(f"`mask` should be a 2d-tensor, got {len(mask.shape)} dims.")

        batch_dim, seq_dim, sequence_state_dim = sequence_state.shape
        pairwise_state_dim = pairwise_state.shape[3]

        if sequence_state_dim != self.config.sequence_state_dim:
            raise ValueError(
                "`sequence_state` last dimension should be equal to `self.sequence_state_dim`. Got "
                f"{sequence_state_dim} != {self.config.sequence_state_dim}."
            )
        if pairwise_state_dim != self.config.pairwise_state_dim:
            raise ValueError(
                "`pairwise_state` last dimension should be equal to `self.pairwise_state_dim`. Got "
                f"{pairwise_state_dim} != {self.config.pairwise_state_dim}."
            )
        if batch_dim != pairwise_state.shape[0]:
            raise ValueError(
                f"`sequence_state` and `pairwise_state` have inconsistent batch size: {batch_dim} != "
                f"{pairwise_state.shape[0]}."
            )
        if seq_dim != pairwise_state.shape[1] or seq_dim != pairwise_state.shape[2]:
            raise ValueError(
                f"`sequence_state` and `pairwise_state` have inconsistent sequence length: {seq_dim} != "
                f"{pairwise_state.shape[1]} or {pairwise_state.shape[2]}."
            )

        if ablate_pair_to_seq:
            # Cut the z -> s bridge for this block: no pairwise-derived bias
            # reaches the sequence self-attention below.
            # Update sequence state
            bias = None
        else:
            bias = self.pair_to_sequence(pairwise_state)

        # Self attention with bias + mlp.
        y = self.layernorm_1(sequence_state)
        y, _ = self.seq_attention(y, mask=mask, bias=bias)
        sequence_state = sequence_state + self.drop(y)
        sequence_state = self.mlp_seq(sequence_state)

        # Update pairwise state
        if ablate_seq_to_pair:
            # Cut the s -> z bridge for this block: z passes through this
            # block completely unchanged by s (skip the additive update).
            pairwise_state = pairwise_state
        else:
            pairwise_state = pairwise_state + self.sequence_to_pair(sequence_state)

        # Axial attention with triangular bias.
        tri_mask = mask.unsqueeze(2) * mask.unsqueeze(1) if mask is not None else None
        pairwise_state = pairwise_state + self.row_drop(self.tri_mul_out(pairwise_state, mask=tri_mask))
        pairwise_state = pairwise_state + self.col_drop(self.tri_mul_in(pairwise_state, mask=tri_mask))
        pairwise_state = pairwise_state + self.row_drop(
            self.tri_att_start(pairwise_state, mask=tri_mask, chunk_size=chunk_size)
        )
        pairwise_state = pairwise_state + self.col_drop(
            self.tri_att_end(pairwise_state, mask=tri_mask, chunk_size=chunk_size)
        )

        # MLP over pairs.
        pairwise_state = self.mlp_pair(pairwise_state)

        return sequence_state, pairwise_state
def patch_s_representations_in_trunk(
    self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles,
    donor_s_s_list, donor_s_z_list, target_start, target_end, target_block, patch_mode, donor_hairpin_start, pairwise_mask, ablate_pair_to_seq, ablate_seq_to_pair, ablate_block_indices
):
    """
    Modified trunk forward that patches representations at a specific block.
    
    Args:
        ablate_block_indices: List of block indices where ablation should be applied (can be None or empty)
    """
    # Replacement for EsmFoldingTrunk.forward used during the actual sliding-
    # window-ablation inference passes. Combines two independent interventions
    # in one trunk pass: (1) donor-activation patching at exactly one block
    # (target_block), and (2) bridge ablation (via ablate_bridges kwargs)
    # across a contiguous window of blocks (ablate_block_indices).
    device = seq_feats.device
    s_s_0 = seq_feats
    s_z_0 = pair_feats
    
    # Convert ablate_block_indices to a set for O(1) lookup
    ablate_blocks_set = set(ablate_block_indices) if ablate_block_indices else set()

    if no_recycles is None:
        no_recycles = self.config.max_recycles
    else:
        if no_recycles < 0:
            raise ValueError("Number of recycles must not be negative.")
        no_recycles += 1

    def apply_patch(block_idx, s_s, s_z):
        """Overwrite the target's s and/or z at `block_idx` with donor activations, restricted to the patch region/mask."""
        if patch_mode in ('both', 'sequence'):
            # donor_s_s_list[block_idx] was pre-sliced to ONLY the donor's
            # hairpin-region columns when it was cached (see run_ablation_experiment),
            # so its length along dim=1 is exactly donor_hairpin_end - donor_hairpin_start.
            # The assert enforces a position-for-position copy: target patch
            # region and donor hairpin must have the same residue count.
            donor_block_s_repr = donor_s_s_list[block_idx].to(s_s.device, dtype=s_s.dtype)
            donor_len = donor_block_s_repr.shape[1]
            target_len = target_end - target_start
            assert donor_len == target_len, f"Donor length mismatch: {donor_len} != {target_len}"
            # In-place overwrite of only the target region's columns; positions
            # outside [target_start:target_end) keep this block's own (possibly ablated) output.
            s_s[:, target_start:target_end, :] = donor_block_s_repr
            
        if patch_mode in ('both', 'pairwise'):
            # Unlike the sequence donor list, donor_s_z_list was cached WITHOUT
            # slicing (full L_donor x L_donor grid) -- see run_ablation_experiment --
            # so absolute donor coordinates must be recovered via the offset below,
            # rather than indexed directly like the sequence patch above.
            donor_z = donor_s_z_list[block_idx].to(s_z.device, dtype=s_z.dtype)
            # Local var; shadows (but is unrelated to) the outer attention `mask`
            # argument of patch_s_representations_in_trunk -- scoped to this closure only.
            mask = pairwise_mask.to(s_z.device)

            #apply mask: for each True position in mask, copy from corresponding donor representation
            # Plain Python double loop over up to target_len^2 cells -- simple
            # and correct, not vectorized, since patch masks here are small
            # (single hairpin regions), not full L x L grids.
            for t_i in range(mask.shape[0]):
                for t_j in range(mask.shape[1]):
                    if mask[t_i, t_j]:
                        #map target coords to donor coords via relative position
                        # Re-base each target index through its own hairpin-start
                        # offset into the donor's index space (target_start and
                        # donor_hairpin_start both mark "start of hairpin region"
                        # in their respective sequences). create_pairwise_mask's
                        # transport range is bounded by min(donor side, target
                        # side) on each side specifically so d_i/d_j below always
                        # stay within the donor tensor's valid index range.
                        d_i = t_i - target_start + donor_hairpin_start
                        d_j = t_j - target_start + donor_hairpin_start
                        s_z[:, t_i, t_j, :] = donor_z[:, d_i, d_j, :]
    
        return s_s, s_z

    def trunk_iter(s, z, residx, mask):
        """Run all trunk blocks once, ablating bridges on `ablate_blocks_set` blocks and patching donor activations in at `target_block`."""
        z = z + self.pairwise_positional_embedding(residx, mask=mask)
        s_s_list = []
        s_z_list = []
        for block_idx, block in enumerate(self.blocks):
            if block_idx in ablate_blocks_set:
                # Only blocks inside the requested sliding window get the
                # ablate_* kwargs; every other block below calls with the
                # defaults (both False), so ablate_bridges behaves exactly
                # like the un-ablated original there.
                s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size, ablate_pair_to_seq=ablate_pair_to_seq, ablate_seq_to_pair=ablate_seq_to_pair)
            else:
                s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size)
            if block_idx == target_block:
                # Patching happens AFTER this block's own (possibly-ablated)
                # computation, so donor activations REPLACE this block's output
                # before it becomes the input to the NEXT block -- i.e. "run
                # normally/ablated through target_block, splice in the donor's
                # version of the patch region, then let the rest of the trunk
                # propagate that forward."
                s, z = apply_patch(block_idx, s, z)
            s_s_list.append(s)
            s_z_list.append(z)
        return s, z, s_s_list, s_z_list

    # Recycle loop: identical mechanics to collect_block_representations above
    # (recycle_s/recycle_z/recycle_bins zero-initialized, only the final
    # recycle's per-block lists survive, num_recycles=0 everywhere in this
    # script so this always runs exactly once) -- see comments there.
    s_s = s_s_0
    s_z = s_z_0

    recycle_s = torch.zeros_like(s_s)
    recycle_z = torch.zeros_like(s_z)
    recycle_bins = torch.zeros(*s_z.shape[:-1], device=device, dtype=torch.int64)

    for recycle_idx in range(no_recycles):
        with ContextManagers([] if recycle_idx == no_recycles - 1 else [torch.no_grad()]):
            recycle_s = self.recycle_s_norm(recycle_s.detach()).to(device)
            recycle_z = self.recycle_z_norm(recycle_z.detach()).to(device)
            recycle_z += self.recycle_disto(recycle_bins.detach()).to(device)

            s_s, s_z, sequence_list, pairwise_list = trunk_iter(
                s_s_0 + recycle_s, s_z_0 + recycle_z, residx, mask
            )

            structure = self.structure_module(
                {"single": self.trunk2sm_s(s_s), "pair": self.trunk2sm_z(s_z)},
                true_aa,
                mask.float(),
            )

            recycle_s = s_s
            recycle_z = s_z
            recycle_bins = EsmFoldingTrunk.distogram(
                structure["positions"][-1][:, :, :3],
                3.375,
                21.375,
                self.recycle_bins,
            )

    structure["s_s"] = s_s
    structure["s_z"] = s_z
    structure['s_s_list'] = sequence_list
    structure['s_z_list'] = pairwise_list

    return structure

def high_forward_pass(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    masking_pattern: Optional[torch.Tensor] = None,
    num_recycles: Optional[int] = None,
    output_hidden_states: Optional[bool] = False,
    donor_s_s_list=None,
    donor_s_z_list=None,
    target_start=None,
    target_end=None,
    target_block=None,
    patch_mode=None,
    donor_hairpin_start=None,
    pairwise_mask=None,
    ablate_seq_to_pair=None,
    ablate_pair_to_seq=None,
    ablate_block_indices=None,
) -> NewEsmForProteinFoldingOutput:
    """Modified forward that applies patching during inference.
    
    Args:
        ablate_block_indices: List of block indices where ablation should be applied
    """
    # Body is identical to return_block_representations above (see comments
    # there) except that all the donor/patch/ablation kwargs are forwarded
    # into self.trunk(...) below. Must be paired with
    # `model.trunk.forward = types.MethodType(patch_s_representations_in_trunk, ...)`,
    # since that's the only trunk forward that accepts these extra kwargs.
    cfg = self.config.esmfold_config

    aa = input_ids
    B = aa.shape[0]
    L = aa.shape[1]
    device = input_ids.device
    
    if attention_mask is None:
        attention_mask = torch.ones_like(aa, device=device)
    if position_ids is None:
        position_ids = torch.arange(L, device=device).expand_as(input_ids)

    esmaa = self.af2_idx_to_esm_idx(aa, attention_mask)

    if masking_pattern is not None:
        masked_aa, esmaa, mlm_targets = self.bert_mask(aa, esmaa, attention_mask, masking_pattern)
    else:
        masked_aa = aa
        mlm_targets = None

    esm_s = self.compute_language_model_representations(esmaa)
    esm_s = esm_s.to(self.esm_s_combine.dtype)

    if cfg.esm_ablate_sequence:
        esm_s = esm_s * 0

    esm_s = esm_s.detach()
    esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
    s_s_0 = self.esm_s_mlp(esm_s)
    s_z_0 = s_s_0.new_zeros(B, L, L, cfg.trunk.pairwise_state_dim)

    if self.config.esmfold_config.embed_aa:
        s_s_0 += self.embedding(masked_aa)

    # Forward all patch/ablation parameters straight through to
    # patch_s_representations_in_trunk (bound as self.trunk.forward).
    structure: dict = self.trunk(
        s_s_0, s_z_0, aa, position_ids, attention_mask, no_recycles=num_recycles,
        donor_s_s_list=donor_s_s_list, donor_s_z_list=donor_s_z_list,
        target_start=target_start, target_end=target_end,
        target_block=target_block, patch_mode=patch_mode,
        donor_hairpin_start=donor_hairpin_start, pairwise_mask=pairwise_mask,
        ablate_pair_to_seq=ablate_pair_to_seq, ablate_seq_to_pair=ablate_seq_to_pair, 
        ablate_block_indices=ablate_block_indices
    )
    
    structure = {
        k: v for k, v in structure.items()
        if k in [
            "s_z", "s_s", "frames", "sidechain_frames", "unnormalized_angles",
            "angles", "positions", "states", "s_s_list", "s_z_list"
        ]
    }

    if mlm_targets:
        structure["mlm_targets"] = mlm_targets

    disto_logits = self.distogram_head(structure["s_z"])
    disto_logits = (disto_logits + disto_logits.transpose(1, 2)) / 2
    structure["distogram_logits"] = disto_logits

    lm_logits = self.lm_head(structure["s_s"])
    structure["lm_logits"] = lm_logits

    structure["aatype"] = aa
    make_atom14_masks(structure)
    
    for k in ["atom14_atom_exists", "atom37_atom_exists"]:
        structure[k] *= attention_mask.unsqueeze(-1)
    structure["residue_index"] = position_ids

    lddt_head = self.lddt_head(structure["states"]).reshape(
        structure["states"].shape[0], B, L, -1, self.lddt_bins
    )
    structure["lddt_head"] = lddt_head
    plddt = categorical_lddt(lddt_head[-1], bins=self.lddt_bins)
    structure["plddt"] = plddt

    ptm_logits = self.ptm_head(structure["s_z"])
    structure["ptm_logits"] = ptm_logits
    structure["ptm"] = compute_tm(ptm_logits, max_bin=31, no_bins=self.distogram_bins)
    structure.update(compute_predicted_aligned_error(ptm_logits, max_bin=31, no_bins=self.distogram_bins))

    return NewEsmForProteinFoldingOutput(**structure)

# ============================================================================
# Single Block Experiment Runner
# ============================================================================
def create_pairwise_mask(
    donor_hairpin_start: int,
    donor_hairpin_end: int,
    donor_len: int,
    target_start: int,
    target_end: int,
    target_len: int,
    mode: str,
) -> torch.Tensor:
    """Create the pairwise patch mask."""
    # Boolean L x L grid (L = target_len) marking which (i, j) pairwise cells
    # get overwritten with donor activations in apply_patch, above.
    patch_mask = torch.zeros(target_len, target_len, dtype=torch.bool)
    
    if mode == "intra":
        # Only pairs where BOTH residues fall inside the patch/hairpin region
        # (contacts within the new hairpin itself).
        patch_mask[target_start:target_end, target_start:target_end] = True
        
    elif mode in ("touch", "hole"):
        # Compute transportable range
        # How far outside the patch region (toward the N-terminus) the patch
        # can safely extend, bounded by whichever sequence -- donor or target --
        # has LESS room on that side. This guarantees the d_i/d_j offset
        # mapping in apply_patch always lands inside the donor's valid index range.
        donor_left_extent = donor_hairpin_start
        target_left_extent = target_start
        left_extent = min(donor_left_extent, target_left_extent)
        
        # Same idea for the C-terminal side.
        donor_right_extent = donor_len - donor_hairpin_end
        target_right_extent = target_len - target_end
        right_extent = min(donor_right_extent, target_right_extent)
        
        transport_start = target_start - left_extent
        transport_end = target_end + right_extent
        
        # Create cross
        # "+"-shaped region of the L x L grid: (hairpin rows x transport-range
        # cols) plus its transpose (transport-range rows x hairpin cols) --
        # i.e. every pairwise interaction between the hairpin and its extended
        # neighborhood, in both directions (z is not symmetric).
        patch_mask[target_start:target_end, transport_start:transport_end] = True
        patch_mask[transport_start:transport_end, target_start:target_end] = True
        
        if mode == "hole":
            # "hole" = same cross as "touch" but with the intra-hairpin block
            # (the "intra" mode's region) carved back out -- isolates
            # hairpin-to-neighborhood contacts while excluding the hairpin's
            # contacts with itself.
            # Cut out intra-hairpin region
            patch_mask[target_start:target_end, target_start:target_end] = False
    
    return patch_mask


def visualize_pairwise_mask(
    patch_mask: torch.Tensor,
    target_start: int,
    target_end: int,
    mode: str,
    save_path: str,
    figsize: Tuple[int, int] = (6, 5),
):
    """Save visualization of pairwise mask."""
    fig, ax = plt.subplots(figsize=figsize)
    # Render the bool mask as a 0/1 heatmap; origin='upper' keeps row 0 at the
    # top, matching the usual (i, j) convention for viewing an L x L pairwise grid.
    ax.imshow(patch_mask.numpy(), cmap='Greys', vmin=0, vmax=1, origin='upper')
    ax.set_xlabel('Residue Position (Target)')
    ax.set_ylabel('Residue Position (Target)')
    ax.set_title(f'Pairwise Patch Mask ({mode})\nHairpin region: [{target_start}:{target_end}]')
    
    from matplotlib.patches import Rectangle
    # The -0.5 offset accounts for imshow cell centers sitting at integer
    # coordinates, so the rectangle aligns with cell boundaries (rather than
    # being off by half a cell) to outline the patch/hairpin region.
    rect = Rectangle((target_start - 0.5, target_start - 0.5),
                      target_end - target_start,
                      target_end - target_start,
                      linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved patch visualization to {save_path}")

def run_single_block_experiment_fast(
    model,
    tokenizer,
    device,
    target_seq: str,
    donor_s_blocks: List[torch.Tensor],
    donor_z_blocks: List[torch.Tensor],
    donor_hairpin_start: int,
    target_start: int,
    target_end: int,
    pairwise_mask: torch.Tensor,
    patch_mode: str,
    patch_mask_mode: str,
    num_blocks: int,
    save_pdbs: bool = False,
    pdb_dir: Optional[str] = None,
    case_id: Optional[str] = None,
    compute_helix: bool = False,
    orig_helix_data: Tuple = (None, None, None),
) -> List[Dict[str, Any]]:
    """
    Run patching experiment - assumes donor representations are already extracted.
    """
    # NOTE: this function is not called anywhere in this file's own main()/
    # run_ablation_experiment() pipeline (which uses run_single_ablation_experiment
    # instead) -- it appears to be a leftover/reusable helper carried over from
    # an earlier single-block (non-sliding-window) patching experiment.
    results = []
    orig_helix_count, orig_total, orig_helix_pct = orig_helix_data
    
    for block_idx in range(num_blocks):
        # Re-apply the model+trunk forward monkey-patch every iteration (cheap
        # bound-method assignment); model.forward and model.trunk.forward must
        # always be swapped together since high_forward_pass only works when
        # paired with patch_s_representations_in_trunk (see comments above).
        model.forward = types.MethodType(high_forward_pass, model)
        model.trunk.forward = types.MethodType(patch_s_representations_in_trunk, model.trunk)
        
        with torch.no_grad():
            target_inputs = tokenizer(
                target_seq, return_tensors='pt', add_special_tokens=False
            ).to(device)
            
            patched_outputs = model(
                **target_inputs,
                num_recycles=0,
                donor_s_s_list=donor_s_blocks,
                donor_s_z_list=donor_z_blocks,
                target_start=target_start,
                target_end=target_end,
                target_block=block_idx,
                patch_mode=patch_mode,
                donor_hairpin_start=donor_hairpin_start,
                pairwise_mask=pairwise_mask,
            )
        
        # Clean up outputs
        # Strip the extra per-block lists (not fields of the stock output
        # class) before repackaging into EsmForProteinFoldingOutput, which
        # detect_hairpins/model.output_to_pdb below expect.
        patched_outputs_dict = dict(patched_outputs)
        patched_outputs_dict.pop("s_s_list", None)
        patched_outputs_dict.pop("s_z_list", None)
        clean_outputs = EsmForProteinFoldingOutput(**patched_outputs_dict)
        
        # Check for hairpin using detect_hairpins
        # Runs DSSP-based secondary-structure detection on the predicted
        # structure; only the boolean is needed here, so the hairpin
        # coordinate list (2nd return value) is discarded.
        hairpin_found, _ = detect_hairpins(clean_outputs, model)
        
        result = {
            "block_idx": block_idx,
            "patch_mode": patch_mode,
            "patch_mask_mode": patch_mask_mode,
            "hairpin_found": hairpin_found,
        }
        
        # Helix analysis
        if save_pdbs or compute_helix:
            pdb_string = model.output_to_pdb(clean_outputs)[0]
            
            if compute_helix:
                patched_helix_count, patched_total, patched_helix_pct = compute_alpha_helix_content(pdb_string)
                
                if orig_helix_pct is not None and patched_helix_pct is not None:
                    helix_absolute_change = patched_helix_pct - orig_helix_pct
                    helix_relative_change = ((patched_helix_pct - orig_helix_pct) / orig_helix_pct * 100) if orig_helix_pct > 0 else None
                else:
                    helix_absolute_change = None
                    helix_relative_change = None
                
                result.update({
                    "original_helix_count": orig_helix_count,
                    "original_total_residues": orig_total,
                    "original_helix_pct": orig_helix_pct,
                    "patched_helix_count": patched_helix_count,
                    "patched_total_residues": patched_total,
                    "patched_helix_pct": patched_helix_pct,
                    "helix_absolute_change": helix_absolute_change,
                    "helix_relative_change": helix_relative_change,
                })
            
            if save_pdbs and pdb_dir:
                pdb_filename = f"{case_id}_block{block_idx}_{patch_mode}_{patch_mask_mode}.pdb"
                pdb_path = os.path.join(pdb_dir, pdb_filename)
                with open(pdb_path, 'w') as f:
                    f.write(pdb_string)
        
        results.append(result)
        # Release this pass's freed activation memory back to the CUDA
        # allocator so many repeated forward passes (one per block here) don't
        # accumulate fragmented/reserved-but-unused GPU memory.
        torch.cuda.empty_cache()
    
    return results


# ============================================================================
# Main Experiment Runner
# ============================================================================

def parse_patch_region(patch_region_str: str) -> Tuple[int, int]:
    """Parse target_patch_region string like '(11, 27)' to tuple."""
    import ast
    # literal_eval safely parses a literal tuple/list/etc from a string
    # without executing arbitrary code, unlike a plain eval().
    return ast.literal_eval(patch_region_str)


def run_single_ablation_experiment(
    model,
    tokenizer,
    device,
    target_seq: str,
    donor_s_blocks: List[torch.Tensor],
    donor_z_blocks: List[torch.Tensor],
    donor_hairpin_start: int,
    target_start: int,
    target_end: int,
    pairwise_mask: torch.Tensor,
    patch_mode: str,
    patch_mask_mode: str,
    patch_block_idx: int,
    num_blocks: int,
    case_id: Optional[str] = None,
    window_sizes: List[int] = [3, 5, 10, 15],
) -> List[Dict[str, Any]]:
    """
    Run sliding window ablation experiment for a single successful patching case.
    
    For a given patch applied at patch_block_idx, test ablating contiguous windows
    of bridges starting at the patch block and sliding forward.
    
    Args:
        window_sizes: List of window sizes to test (default: [3, 5, 10, 15])
    """
    results = []
    
    # The 3 (ablate_pair_to_seq, ablate_seq_to_pair) combinations actually
    # tested per window: cut the z->s bridge only, the s->z bridge only, or
    # both simultaneously, for every block in the current sliding window.
    ablation_types = [
        ("pair2seq", True, False),
        ("seq2pair", False, True),
        ("both", True, True),
    ]
    
    # Calculate total iterations for this case
    # Precompute the total number of (window_size, window_start, ablation_type)
    # combinations purely to size the progress bar below; not used elsewhere.
    total_iters = 0
    for window_size in window_sizes:
        max_start = num_blocks - window_size
        n_windows = max(0, max_start - patch_block_idx + 1)
        total_iters += n_windows * len(ablation_types)
    
    # Inner progress bar
    pbar = tqdm(total=total_iters, desc="  Windows", position=1, leave=False)
    
    for window_size in window_sizes:
        # Sliding window starts at patch_block_idx and moves forward
        # Window can start at patch_block_idx up to (num_blocks - window_size)
        # max_start is the last valid start so the window [start, start+size)
        # never runs past the final block.
        max_start = num_blocks - window_size
        
        for window_start in range(patch_block_idx, max_start + 1):
            # The window is only ever placed AT or AFTER patch_block_idx:
            # ablating bridges before the donor patch is applied couldn't
            # affect whether the patched information survives downstream, so
            # only "how far after the patch does ablation still destroy the
            # hairpin" is tested here.
            window_end = window_start + window_size  # exclusive
            ablate_block_indices = list(range(window_start, window_end))
            
            for ablation_name, ablate_pair_to_seq, ablate_seq_to_pair in ablation_types:
                # Re-apply the paired model/trunk monkey-patch before every
                # single forward pass (see the "Monkey-patched Forward
                # Functions" section above for why model.forward and
                # model.trunk.forward must always be swapped as a matched pair).
                model.forward = types.MethodType(high_forward_pass, model)
                model.trunk.forward = types.MethodType(patch_s_representations_in_trunk, model.trunk)
                
                with torch.no_grad():
                    target_inputs = tokenizer(
                        target_seq, return_tensors='pt', add_special_tokens=False
                    ).to(device)
                    
                    # target_block is always patch_block_idx here (the block
                    # where the original single-block patching experiment
                    # succeeded) -- only the ablation window/type varies across
                    # this sweep. num_recycles=0 keeps each of the many sweep
                    # iterations to a single, directly-comparable trunk pass.
                    patched_outputs = model(
                        **target_inputs,
                        num_recycles=0,
                        donor_s_s_list=donor_s_blocks,
                        donor_s_z_list=donor_z_blocks,
                        target_start=target_start,
                        target_end=target_end,
                        target_block=patch_block_idx,
                        patch_mode=patch_mode,
                        donor_hairpin_start=donor_hairpin_start,
                        pairwise_mask=pairwise_mask,
                        ablate_pair_to_seq=ablate_pair_to_seq,
                        ablate_seq_to_pair=ablate_seq_to_pair,
                        ablate_block_indices=ablate_block_indices,
                    )
                
                # Clean up outputs
                patched_outputs_dict = dict(patched_outputs)
                patched_outputs_dict.pop("s_s_list", None)
                patched_outputs_dict.pop("s_z_list", None)
                clean_outputs = EsmForProteinFoldingOutput(**patched_outputs_dict)
                
                # Check for hairpin using detect_hairpins
                hairpin_found, _ = detect_hairpins(clean_outputs, model)
                
                result = {
                    "patch_block_idx": patch_block_idx,
                    "window_size": window_size,
                    "window_start": window_start,
                    "window_end": window_end,
                    "ablate_block_indices": str(ablate_block_indices),
                    "ablation_type": ablation_name,
                    "ablate_pair_to_seq": ablate_pair_to_seq,
                    "ablate_seq_to_pair": ablate_seq_to_pair,
                    "hairpin_found": hairpin_found,
                }
                
                results.append(result)
                pbar.update(1)
                # This is the hottest loop in the whole sweep (window_sizes x
                # window_starts x 3 ablation_types forward passes per case) --
                # empty_cache() here keeps GPU memory from fragmenting/growing
                # across the many repeated forward passes.
                torch.cuda.empty_cache()
    
    pbar.close()
    return results


def run_ablation_experiment(
    results_csv_path: str,
    n_sequence_cases: int,
    n_pairwise_cases: int,
    output_dir: str,
    device: Optional[str] = None,
    cache_flush_interval: int = 20,
    window_sizes: List[int] = [3, 5, 10, 15],
) -> pd.DataFrame:
    """
    Run sliding window ablation experiments on successful patching cases.
    
    Args:
        results_csv_path: Path to block_patching_successes.csv with successful cases
        n_sequence_cases: Number of sequence patching cases to run
        n_pairwise_cases: Number of pairwise patching cases to run
        output_dir: Directory for outputs
        device: Device to use (auto-detected if None)
        cache_flush_interval: Flush donor cache and save results every N cases (default: 20)
        window_sizes: List of window sizes to test (default: [3, 5, 10, 15])
        
    Returns:
        DataFrame with ablation results
    """
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    
    # Device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model via the shared helper (handles device placement + precision
    # auto-detection; see src/utils/model_utils.py).
    model, tokenizer = load_esmfold(device)

    # Monkey-patch the block forward to support ablation
    # One-time, PERMANENT patch applied to every block right after loading and
    # never reverted. This is safe to leave in place because ablate_bridges
    # called with its default kwargs (both False) is behaviorally identical to
    # the original block forward -- so blocks never targeted by a later
    # ablate_block_indices window are completely unaffected.
    for block in model.trunk.blocks:
        block.forward = types.MethodType(ablate_bridges, block)
    
    # Load results and filter to successful cases
    print(f"Loading results from {results_csv_path}...")
    df = pd.read_csv(results_csv_path)
    
    # Filter to only successful hairpin formations
    # Ablation is only informative on cases that already produced a hairpin in
    # the earlier single-block patching experiment -- there'd be nothing to
    # (potentially) destroy otherwise. .copy() avoids pandas chained-indexing
    # issues on the filtered view before the .head()/concat calls below.
    successful_df = df[df['hairpin_found'] == True].copy()
    print(f"Found {len(successful_df)} successful cases out of {len(df)} total")
    
    # Separate by patch mode and take specified number from each
    # .head(n) takes the FIRST n rows in CSV order (not a random sample) --
    # deterministic/reproducible, but whatever ordering bias the source CSV has.
    sequence_cases = successful_df[successful_df['patch_mode'] == 'sequence'].head(n_sequence_cases)
    pairwise_cases = successful_df[successful_df['patch_mode'] == 'pairwise'].head(n_pairwise_cases)
    
    print(f"Selected {len(sequence_cases)} sequence patching cases (requested: {n_sequence_cases})")
    print(f"Selected {len(pairwise_cases)} pairwise patching cases (requested: {n_pairwise_cases})")
    
    cases_to_run = pd.concat([sequence_cases, pairwise_cases], ignore_index=True)
    print(f"Total cases to run: {len(cases_to_run)}")
    
    # Parent columns to preserve (updated for new CSV format)
    # Identifying/provenance columns copied from the source single-block-patching
    # CSV onto every ablation-sweep result row below, so each output row can be
    # traced back to its originating target/donor/patch-location case.
    parent_columns = [
        "case_idx", "target_name", "target_sequence", "target_length",
        "loop_idx", "loop_start", "loop_end", "loop_length", "loop_sequence",
        "target_patch_start", "target_patch_end", "patch_length",
        "donor_pdb", "donor_sequence", "donor_length",
        "donor_hairpin_start", "donor_hairpin_end", "donor_hairpin_length", "donor_hairpin_sequence",
        "patch_mode", "patch_mask_mode", "block_idx",
    ]
    
    all_results = []
    
    # Cache for donor representations (keyed by donor_sequence)
    donor_cache = {}
    cases_processed = 0
    
    # Paths for saving
    results_path = os.path.join(output_dir, "ablation_results.csv")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Main progress bar
    pbar = tqdm(total=len(cases_to_run), desc="Cases", position=0)
    
    skipped_cases = []
    
    for row_idx, row in cases_to_run.iterrows():
        case_id = f"ablate_case_{row_idx}"
        # Broad catch-all around the whole per-case body: this sweep can run
        # for hours over hundreds of cases, so one bad case (DSSP failure,
        # malformed CSV row, shape mismatch, ...) shouldn't abort the entire
        # run -- it's logged to skipped_cases and reported at the end instead.
        try:
        
            # Extract info from parent row (updated for new CSV format)
            target_seq = row['target_sequence']
            donor_seq = row['donor_sequence']
            patch_mode = row['patch_mode']
            patch_mask_mode = row['patch_mask_mode']
            patch_block_idx = int(row['block_idx'])
        
            # Get patch region from CSV
            target_start = int(row['target_patch_start'])
            target_end = int(row['target_patch_end'])
        
            # Get donor hairpin locations from CSV
            donor_hairpin_start = int(row['donor_hairpin_start'])
            donor_hairpin_end = int(row['donor_hairpin_end'])
        
            # Get parent metadata
            parent_meta = {col: row[col] for col in parent_columns if col in row.index}
        
            # Update progress bar description
            pbar.set_description(f"[{patch_mode[:3]}] {row['target_name'][:10]}<-{row['donor_pdb'][:6]}")
        
            # Get or compute donor representations
            # Cache MISS: monkey-patch to the "collection" pair
            # (return_block_representations / collect_block_representations)
            # to run one forward pass over the donor sequence and record every
            # block's s/z.
            if donor_seq not in donor_cache:
                model.forward = types.MethodType(return_block_representations, model)
                model.trunk.forward = types.MethodType(collect_block_representations, model.trunk)
            
                with torch.no_grad():
                    donor_inputs = tokenizer(donor_seq, return_tensors='pt', add_special_tokens=False).to(device)
                    donor_outputs = model(**donor_inputs, num_recycles=0)
            
                # Extract and cache
                # Only keep the donor's hairpin-region COLUMNS of s, since that's
                # the only part ever grafted into the target sequence's s
                # (see apply_patch's "sequence" branch above).
                donor_s_blocks = []
                for block_repr in donor_outputs.s_s_list:
                    donor_s_blocks.append(
                        block_repr[:, donor_hairpin_start:donor_hairpin_end, :].detach().cpu()
                    )
            
                # z is cached in FULL (no slicing) because "touch"/"hole" patch
                # modes can reach beyond the hairpin region itself -- apply_patch
                # recovers the right donor coordinates via an index offset instead.
                donor_z_blocks = []
                for block_repr in donor_outputs.s_z_list:
                    donor_z_blocks.append(block_repr.detach().cpu())
            
                donor_cache[donor_seq] = {
                    'donor_s_blocks': donor_s_blocks,
                    'donor_z_blocks': donor_z_blocks,
                }
            
                del donor_outputs
                torch.cuda.empty_cache()
            else:
                # Cache HIT: reuse the previously-extracted donor activations.
                cached = donor_cache[donor_seq]
                donor_s_blocks = cached['donor_s_blocks']
                donor_z_blocks = cached['donor_z_blocks']

            num_blocks = len(donor_s_blocks)
        
            # Create pairwise mask
            pairwise_mask = create_pairwise_mask(
                donor_hairpin_start=donor_hairpin_start,
                donor_hairpin_end=donor_hairpin_end,
                donor_len=len(donor_seq),
                target_start=target_start,
                target_end=target_end,
                target_len=len(target_seq),
                mode=patch_mask_mode,
            )
        
            # Run ablation experiment
            results = run_single_ablation_experiment(
                model=model,
                tokenizer=tokenizer,
                device=device,
                target_seq=target_seq,
                donor_s_blocks=donor_s_blocks,
                donor_z_blocks=donor_z_blocks,
                donor_hairpin_start=donor_hairpin_start,
                target_start=target_start,
                target_end=target_end,
                pairwise_mask=pairwise_mask,
                patch_mode=patch_mode,
                patch_mask_mode=patch_mask_mode,
                patch_block_idx=patch_block_idx,
                num_blocks=num_blocks,
                case_id=case_id,
                window_sizes=window_sizes,
            )
        
            # Add parent metadata to each result
            for r in results:
                r.update(parent_meta)
        
            all_results.extend(results)
            cases_processed += 1
            pbar.update(1)
        
            # Periodically save results, generate plots, and flush cache
            # Checkpoint every cache_flush_interval cases: a durability +
            # memory safeguard (partial results survive a crash; donor_cache
            # -- one activation tensor per trunk block per distinct donor --
            # is bounded rather than growing for the whole run), not something
            # that changes the science.
            if cases_processed % cache_flush_interval == 0:
                pbar.write(f"\n{'='*60}")
                pbar.write(f"Checkpoint at {cases_processed} cases")
                pbar.write(f"{'='*60}")
            
                # Save current results
                results_df = pd.DataFrame(all_results)
                results_df.to_csv(results_path, index=False)
                pbar.write(f"Saved {len(results_df)} results to {results_path}")
            
                # Generate plots
                # Plotting is best-effort/non-critical: if it fails (e.g. no
                # data yet for some window/patch_mode combo), the sweep keeps
                # running rather than crashing over a cosmetic plot failure.
                try:
                    from sliding_window_plotting import generate_ablation_plots, print_ablation_summary
                    print_ablation_summary(results_df)
                    generate_ablation_plots(results_df, plots_dir)
                except Exception as e:
                    pbar.write(f"Warning: Plot generation failed: {e}")
            
                # Flush donor cache
                pbar.write(f"Flushing donor cache ({len(donor_cache)} entries)")
                donor_cache.clear()
                torch.cuda.empty_cache()
            
                pbar.write(f"{'='*60}\n")
        except Exception as e:
            pbar.write(f"WARNING: Skipping case {row_idx} ({case_id}): {e}")
            skipped_cases.append({'row_idx': row_idx, 'error': str(e)})
            pbar.update(1)
            continue
    
    if skipped_cases:
        print(f"\nSkipped {len(skipped_cases)} cases due to errors:")
        for sc in skipped_cases:
            print(f"  Row {sc['row_idx']}: {sc['error']}")
    
    pbar.close()
    
    # Final save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_path, index=False)
    print(f"\nFinal results saved to {results_path}")
    
    # Final plots
    try:
        from sliding_window_plotting import generate_ablation_plots, print_ablation_summary
        print_ablation_summary(results_df)
        generate_ablation_plots(results_df, plots_dir)
    except Exception as e:
        print(f"Warning: Final plot generation failed: {e}")
    
    return results_df


# Add to main entry point
def main():
    """CLI entry point: run the sliding-window bridge-ablation experiment sweep."""
    parser = argparse.ArgumentParser(
        description="Run ESMFold block patching experiments"
    )
    parser.add_argument(
        "--ablation_csv", type=str, default='data/single_block_patching_successes.csv',
        help="Path to successful_cases.csv"
    )
    # NOTE: possible bug -- args.patch_modes is parsed but never read anywhere
    # below (run_ablation_experiment doesn't take a patch_modes parameter);
    # this flag currently has no effect.
    parser.add_argument(
        "--patch_modes",
        nargs="+",
        default=["sequence", "pairwise"],
        help="Patch modes"
    )
    parser.add_argument(
        "--output_dir", type=str, default="sliding_window_ablation",
        help="Output directory (default: ./results)"
    )
    # NOTE: possible bug -- args.save_pdbs / args.compute_helix are parsed but
    # never passed into run_ablation_experiment (which has no such
    # parameters); PDB saving and helix-content computation only exist on the
    # unused run_single_block_experiment_fast path, not this sweep's actual
    # entry point below.
    parser.add_argument(
        "--save_pdbs", action="store_true",
        help="Save PDB structures"
    )
    parser.add_argument(
        "--compute_helix", action="store_true",
        help="Compute alpha helix content (requires DSSP via trunk_utils)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (default: auto-detect)"
    )
    # NOTE: possible bug -- args.skip_experiments is parsed but never checked
    # anywhere below; there is no plots-only code path currently wired up.
    parser.add_argument(
        "--skip_experiments", action="store_true",
        help="Skip experiments and only generate plots from existing results"
    )
    parser.add_argument(
        "--n_sequence_cases", type=int, default=400,
        help="Number of sequence patching cases to run (default: 400)"
    )
    parser.add_argument(
        "--n_pairwise_cases", type=int, default=200,
        help="Number of pairwise patching cases to run (default: 200)"
    )
    parser.add_argument(
        "--cache_flush_interval", type=int, default=20,
        help="Flush donor cache every N cases to prevent memory buildup (default: 20)"
    )
    # NOTE: possible bug -- the actual default is [15], but the help text says
    # "(default: 3 5 10 15)", which matches run_ablation_experiment's/
    # run_single_ablation_experiment's own default parameter instead. A user
    # relying on `--help` here would be misled about what runs if they don't
    # pass --window_sizes explicitly.
    parser.add_argument(
        "--window_sizes", type=int, nargs="+", default=[15],
        help="Window sizes for sliding window ablation (default: 3 5 10 15)"
    )
    
    args = parser.parse_args()
    
    # =========================================================================
    # ABLATION EXPERIMENT
    # =========================================================================
    
    # NOTE: possible bug -- args.ablation_csv always has a truthy default
    # string, so this `or` fallback to os.path.join(args.output_dir, ...) is
    # effectively unreachable in normal usage (only triggers if a user
    # explicitly passes --ablation_csv "").
    ablation_csv = args.ablation_csv or os.path.join(args.output_dir, "block_patching_successes.csv")
    if not os.path.exists(ablation_csv):
        print(f"Error: No results found at {ablation_csv}")
        return
    
    ablation_output_dir = os.path.join(args.output_dir, "ablation")
    
    results_df = run_ablation_experiment(
        results_csv_path=ablation_csv,
        n_sequence_cases=args.n_sequence_cases,
        n_pairwise_cases=args.n_pairwise_cases,
        output_dir=ablation_output_dir,
        device=args.device,
        cache_flush_interval=args.cache_flush_interval,
        window_sizes=args.window_sizes,
    )
    
    print(f"\nAblation experiment complete!")
    print(f"Results: {os.path.join(ablation_output_dir, 'ablation_results.csv')}")
    print(f"Plots: {os.path.join(ablation_output_dir, 'plots')}")
    return


if __name__ == "__main__":
    main()