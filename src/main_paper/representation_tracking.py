#!/usr/bin/env python
"""
Representation Flow Analysis with Baseline Freezing and Zero Ablation (Refactored)
==================================================================================

Analyzes how representations flow through ESMFold's trunk blocks, with support for:
1. Collecting baseline representations, pair2seq biases, and seq2pair updates
2. Patching donor representations into target sequences
3. Intervening on information flow (freeze to baseline or zero ablation)

Uses hooks and context managers for clean representation collection and intervention.

Usage:
    python representation_flow_analysis.py \
        --dataset next_experiment.parquet \
        --output results/ \
        --intervention_windows early_10 late_10
"""

import argparse
import os
import warnings
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from transformers import AutoTokenizer, EsmForProteinFolding

from src.utils.model_utils import load_esmfold  # shared ESMFold loader (handles precision/device setup)

# Suppress benign HF/openfold warnings about mmCIF-related code paths that are
# unrelated to the representation-patching analysis performed in this script.
warnings.filterwarnings("ignore", message=".*mmCIF.*")


# ============================================================================
# CONSTANTS
# ============================================================================

NUM_BLOCKS = 48  # ESMFold's folding trunk has 48 sequence/pairwise update blocks

# The 5 conditions swept over for every patched case: an unmodified baseline
# (no intervention), plus freeze/zero ablation crossed with the two
# information pathways (seq2pair, pair2seq) that connect s and z within a block.
# 'freeze' replaces a pathway's output with its pre-patch baseline value;
# 'zero' replaces it with zeros (full ablation of that pathway's contribution).
INTERVENTION_CONDITIONS = [
    (None, None),  # No intervention (baseline)
    ('freeze', 'seq2pair'),
    ('freeze', 'pair2seq'),
    ('zero', 'seq2pair'),
    ('zero', 'pair2seq'),
]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CollectedRepresentations:
    """Container for collected representations from ESMFold trunk blocks."""
    # NOTE: this is a local, file-scoped copy of a same-named dataclass defined
    # in src/utils/representation_utils.py (with a different set of fields).
    # The two are intentionally kept separate; do not merge/refactor them.
    s_blocks: Dict[int, torch.Tensor] = field(default_factory=dict)  # block_idx -> s tensor [1, L, dim]
    z_blocks: Dict[int, torch.Tensor] = field(default_factory=dict)  # block_idx -> z tensor [1, L, L, dim]
    pair2seq_bias_blocks: Dict[int, torch.Tensor] = field(default_factory=dict)  # block_idx -> pair_to_sequence output
    seq2pair_update_blocks: Dict[int, torch.Tensor] = field(default_factory=dict)  # block_idx -> sequence_to_pair output
    
    def clear(self):
        """Clear all collected representations (empties each dict in place)."""
        self.s_blocks.clear()
        self.z_blocks.clear()
        self.pair2seq_bias_blocks.clear()
        self.seq2pair_update_blocks.clear()


# ============================================================================
# HOOK MANAGERS
# ============================================================================

class TrunkCollectionHooks:
    """
    Collect trunk block outputs and intermediate values via hooks.
    
    Collects:
    - s (sequence representation) after each block
    - z (pairwise representation) after each block
    - pair2seq bias (output of pair_to_sequence projection)
    - seq2pair update (output of sequence_to_pair projection)
    
    Usage:
        collector = CollectedRepresentations()
        with TrunkCollectionHooks(model.trunk, collector) as hooks:
            outputs = model(**inputs)
    """
    
    def __init__(
        self,
        trunk: nn.Module,
        collector: CollectedRepresentations,
        blocks: Any = 'all',
        collect_s: bool = True,
        collect_z: bool = True,
        collect_pair2seq: bool = True,
        collect_seq2pair: bool = True,
    ):
        self.trunk = trunk
        self.collector = collector
        self.blocks = blocks
        self.collect_s = collect_s
        self.collect_z = collect_z
        self.collect_pair2seq = collect_pair2seq
        self.collect_seq2pair = collect_seq2pair
        self.handles: List = []
    
    def register(self):
        """Register hooks on trunk blocks and their submodules."""
        blocks = self.blocks
        if blocks == 'all':
            blocks = range(len(self.trunk.blocks))
        
        for idx in blocks:
            block = self.trunk.blocks[idx]
            
            # Hook for collecting s and z after the full block
            if self.collect_s or self.collect_z:
                # make_block_hook is a closure factory: it binds block_idx (and
                # the do_s/do_z flags) by value at registration time. Without
                # this indirection, all hooks would share the same `idx`
                # variable from the enclosing loop (Python closures capture
                # variables by reference, not by value), so every hook would
                # incorrectly read whatever `idx` happened to be after the
                # loop finished.
                def make_block_hook(block_idx, do_s, do_z):
                    def hook(module, inputs, outputs):
                        # Trunk block forward returns a (s, z) tuple.
                        s, z = outputs
                        if do_s:
                            # .detach() drops autograd history; .cpu() frees GPU
                            # memory since these are just being stored for later
                            # offline analysis, not used in a backward pass.
                            self.collector.s_blocks[block_idx] = s.detach().cpu()
                        if do_z:
                            self.collector.z_blocks[block_idx] = z.detach().cpu()
                    return hook
                
                # register_forward_hook fires after block.forward() runs; since
                # this hook returns None, it only observes the output and does
                # not alter it (contrast with TrunkPatchingHooks/TrunkInterventionHooks below).
                handle = block.register_forward_hook(
                    make_block_hook(idx, self.collect_s, self.collect_z)
                )
                self.handles.append(handle)
            
            # Hook for collecting pair2seq bias (output of pair_to_sequence)
            if self.collect_pair2seq:
                def make_pair2seq_hook(block_idx):
                    def hook(module, inputs, outputs):
                        self.collector.pair2seq_bias_blocks[block_idx] = outputs.detach().cpu()
                    return hook
                
                handle = block.pair_to_sequence.register_forward_hook(
                    make_pair2seq_hook(idx)
                )
                self.handles.append(handle)
            
            # Hook for collecting seq2pair update (output of sequence_to_pair)
            if self.collect_seq2pair:
                def make_seq2pair_hook(block_idx):
                    def hook(module, inputs, outputs):
                        self.collector.seq2pair_update_blocks[block_idx] = outputs.detach().cpu()
                    return hook
                
                handle = block.sequence_to_pair.register_forward_hook(
                    make_seq2pair_hook(idx)
                )
                self.handles.append(handle)
    
    def remove(self):
        """Remove all registered hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()
    
    # Context manager protocol: register() on entry, remove() on exit (always,
    # even if the forward pass inside the `with` block raises), so hooks never
    # leak onto the model beyond the intended scope. Same pattern is repeated
    # for the other Hooks classes below.
    def __enter__(self):
        self.register()
        return self
    
    def __exit__(self, *args):
        self.remove()


class TrunkPatchingHooks:
    """
    Apply representation patches after specific blocks via hooks.
    
    Patches donor representations into target positions after the specified block.
    
    Usage:
        with TrunkPatchingHooks(model.trunk, patch_block=0, donor_s=..., donor_z=...) as hooks:
            outputs = model(**inputs)
    """
    
    def __init__(
        self,
        trunk: nn.Module,
        patch_block: int,
        patch_mode: str,  # 'sequence', 'pairwise', or 'both'
        donor_s: Optional[torch.Tensor] = None,  # [1, region_len, dim]
        donor_z: Optional[torch.Tensor] = None,  # [1, L_donor, L_donor, dim]
        target_start: int = 0,
        target_end: int = 0,
        donor_hairpin_start: int = 0,
        pairwise_mask: Optional[torch.Tensor] = None,
    ):
        self.trunk = trunk
        self.patch_block = patch_block
        self.patch_mode = patch_mode
        self.donor_s = donor_s
        self.donor_z = donor_z
        self.target_start = target_start
        self.target_end = target_end
        self.donor_hairpin_start = donor_hairpin_start
        self.pairwise_mask = pairwise_mask
        self.handles: List = []
    
    def register(self):
        """Register post-hook on the patch block."""
        block = self.trunk.blocks[self.patch_block]
        
        def patch_hook(module, inputs, outputs):
            s, z = outputs
            
            # Patch sequence representation
            if self.patch_mode in ('both', 'sequence') and self.donor_s is not None:
                # Move donor tensor onto the same device/dtype as the live
                # activation before assigning into it.
                donor_repr = self.donor_s.to(s.device, dtype=s.dtype)
                # .clone() so we don't mutate the original `s` tensor in place
                # (it may be referenced elsewhere, e.g. by other hooks or the
                # model's own residual connections).
                s = s.clone()
                s[:, self.target_start:self.target_end, :] = donor_repr
            
            # Patch pairwise representation
            if self.patch_mode in ('both', 'pairwise') and self.donor_z is not None:
                donor_z = self.donor_z.to(z.device, dtype=z.dtype)
                z = z.clone()
                
                if self.pairwise_mask is not None:
                    mask_tensor = self.pairwise_mask.to(z.device)
                    # Get the (i, j) target-space coordinates where the mask is True.
                    indices = torch.where(mask_tensor)
                    
                    # Per-pair loop (rather than a vectorized assignment) because
                    # each target index must be remapped into donor-space and
                    # individually bounds-checked (the donor hairpin region can
                    # be a different length/position than the target's patch
                    # region, so some mapped indices may fall outside donor_z).
                    for idx in range(len(indices[0])):
                        t_i, t_j = indices[0][idx].item(), indices[1][idx].item()
                        # Shift target-space index into donor-space using the
                        # same offset convention as the rest of this module:
                        # donor_idx = target_idx - target_start + donor_hairpin_start.
                        d_i = t_i - self.target_start + self.donor_hairpin_start
                        d_j = t_j - self.target_start + self.donor_hairpin_start
                        if 0 <= d_i < donor_z.shape[1] and 0 <= d_j < donor_z.shape[2]:
                            z[:, t_i, t_j, :] = donor_z[:, d_i, d_j, :]
            
            # Returning a value from a forward hook replaces the module's
            # output for the rest of the forward pass (this is what makes the
            # patch actually take effect on subsequent blocks).
            return (s, z)
        
        handle = block.register_forward_hook(patch_hook)
        self.handles.append(handle)
    
    def remove(self):
        """Remove all registered hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()
    
    def __enter__(self):
        self.register()
        return self
    
    def __exit__(self, *args):
        self.remove()


class TrunkInterventionHooks:
    """
    Apply interventions (freeze or zero) on seq2pair or pair2seq pathways.
    
    For 'freeze': Replace the output with pre-computed baseline values
    For 'zero': Replace the output with zeros
    
    Usage:
        with TrunkInterventionHooks(
            model.trunk,
            intervention_type='freeze',
            intervention_pathway='seq2pair',
            start_block=0, end_block=9,
            baseline_values=baseline_seq2pair_list
        ):
            outputs = model(**inputs)
    """
    
    def __init__(
        self,
        trunk: nn.Module,
        intervention_type: str,  # 'freeze' or 'zero'
        intervention_pathway: str,  # 'seq2pair' or 'pair2seq'
        start_block: int,
        end_block: int,
        baseline_values: Optional[List[torch.Tensor]] = None,  # For freeze
    ):
        self.trunk = trunk
        self.intervention_type = intervention_type
        self.intervention_pathway = intervention_pathway
        self.start_block = start_block
        self.end_block = end_block
        self.baseline_values = baseline_values
        self.handles: List = []
    
    def register(self):
        """Register hooks to intervene on the specified pathway."""
        # end_block is inclusive, hence the +1.
        for block_idx in range(self.start_block, self.end_block + 1):
            if block_idx >= len(self.trunk.blocks):
                # Guard against a window that extends past the last block.
                continue
            
            block = self.trunk.blocks[block_idx]
            
            if self.intervention_pathway == 'pair2seq':
                # Intervene on pair_to_sequence output
                # baseline_vals is expected to be a dense, NUM_BLOCKS-length list
                # indexed by block (see baseline_pair2seq_list/baseline_seq2pair_list
                # in run_experiment below), not a sparse dict.
                def make_pair2seq_intervention(idx, int_type, baseline_vals):
                    def hook(module, inputs, outputs):
                        if int_type == 'freeze':
                            # Replace this block's pathway output with the value
                            # it had during the earlier (unpatched) baseline
                            # forward pass -- severs any causal influence the
                            # patched upstream representations would otherwise
                            # have had on this specific pathway at this block.
                            return baseline_vals[idx].to(outputs.device, dtype=outputs.dtype)
                        elif int_type == 'zero':
                            # Full ablation: this pathway contributes nothing
                            # at this block.
                            return torch.zeros_like(outputs)
                        # If int_type is neither 'freeze' nor 'zero', falls through
                        # and implicitly returns None, which leaves the module's
                        # original output unchanged (register_forward_hook semantics).
                    return hook
                
                handle = block.pair_to_sequence.register_forward_hook(
                    make_pair2seq_intervention(block_idx, self.intervention_type, self.baseline_values)
                )
                self.handles.append(handle)
            
            elif self.intervention_pathway == 'seq2pair':
                # Intervene on sequence_to_pair output
                # (mirrors the pair2seq branch above, applied to the other pathway)
                def make_seq2pair_intervention(idx, int_type, baseline_vals):
                    def hook(module, inputs, outputs):
                        if int_type == 'freeze':
                            return baseline_vals[idx].to(outputs.device, dtype=outputs.dtype)
                        elif int_type == 'zero':
                            return torch.zeros_like(outputs)
                    return hook
                
                handle = block.sequence_to_pair.register_forward_hook(
                    make_seq2pair_intervention(block_idx, self.intervention_type, self.baseline_values)
                )
                self.handles.append(handle)
    
    def remove(self):
        """Remove all registered hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()
    
    def __enter__(self):
        self.register()
        return self
    
    def __exit__(self, *args):
        self.remove()


# ============================================================================
# CONTEXT MANAGERS
# ============================================================================

@contextmanager
def collect_all_representations(
    model: EsmForProteinFolding,
    collect_s: bool = True,
    collect_z: bool = True,
    collect_pair2seq: bool = True,
    collect_seq2pair: bool = True,
):
    """
    Context manager for collecting all representations during forward pass.
    
    Yields:
        CollectedRepresentations with s_blocks, z_blocks, pair2seq_bias_blocks, seq2pair_update_blocks
    """
    collector = CollectedRepresentations()
    hooks = TrunkCollectionHooks(
        model.trunk, collector,
        collect_s=collect_s,
        collect_z=collect_z,
        collect_pair2seq=collect_pair2seq,
        collect_seq2pair=collect_seq2pair,
    )
    hooks.register()
    try:
        # Yield the (still-empty) collector; the caller runs the model's
        # forward pass inside the `with` block, and the hooks populate the
        # collector's dicts as a side effect during that call.
        yield collector
    finally:
        # try/finally ensures hooks are removed even if the forward pass
        # inside the `with` block raises (e.g. OOM).
        hooks.remove()


@contextmanager
def apply_patch(
    model: EsmForProteinFolding,
    patch_block: int,
    patch_mode: str,
    donor_s: Optional[torch.Tensor] = None,
    donor_z: Optional[torch.Tensor] = None,
    target_start: int = 0,
    target_end: int = 0,
    donor_hairpin_start: int = 0,
    pairwise_mask: Optional[torch.Tensor] = None,
):
    """Context manager for applying representation patches."""
    hooks = TrunkPatchingHooks(
        model.trunk,
        patch_block=patch_block,
        patch_mode=patch_mode,
        donor_s=donor_s,
        donor_z=donor_z,
        target_start=target_start,
        target_end=target_end,
        donor_hairpin_start=donor_hairpin_start,
        pairwise_mask=pairwise_mask,
    )
    hooks.register()
    try:
        # Nothing to yield -- this context manager's only effect is that the
        # patch hook is active for the duration of the `with` block.
        yield
    finally:
        hooks.remove()


@contextmanager
def apply_intervention(
    model: EsmForProteinFolding,
    intervention_type: str,
    intervention_pathway: str,
    start_block: int,
    end_block: int,
    baseline_values: Optional[List[torch.Tensor]] = None,
):
    """Context manager for applying freeze or zero interventions."""
    # (same register/try-yield/finally-remove pattern as collect_all_representations and apply_patch above)
    hooks = TrunkInterventionHooks(
        model.trunk,
        intervention_type=intervention_type,
        intervention_pathway=intervention_pathway,
        start_block=start_block,
        end_block=end_block,
        baseline_values=baseline_values,
    )
    hooks.register()
    try:
        yield
    finally:
        hooks.remove()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_pairwise_mask(
    donor_hairpin_start: int, donor_hairpin_end: int, donor_len: int,
    target_start: int, target_end: int, target_len: int, mode: str = 'touch',
) -> torch.Tensor:
    """Create pairwise mask for target sequence."""
    # NOTE: possible bug -- donor_hairpin_start, donor_hairpin_end, and donor_len
    # are accepted as parameters but never referenced below; only the target_*
    # args are used to build the mask. If donor-side masking was intended here,
    # it is not implemented.
    # mode='intra': pairs where BOTH i and j fall inside the target patch region.
    # mode='touch': pairs where i OR j falls inside the target patch region
    # (includes cross pairs between the patched region and the rest of the protein).
    mask = torch.zeros(target_len, target_len, dtype=torch.bool)
    if mode == 'intra':
        mask[target_start:target_end, target_start:target_end] = True
    elif mode == 'touch':
        mask[target_start:target_end, :] = True
        mask[:, target_start:target_end] = True
    return mask


def compute_representation_metrics(
    baseline_s, baseline_z, donor_s, donor_z, patched_s, patched_z,
    target_start, target_end, donor_hairpin_start, touch_mask, intra_mask,
) -> Dict[str, float]:
    """Compute interpolation and similarity metrics."""
    results = {}
    
    def cosine_sim(a, b):
        """Cosine similarity between two tensors, flattened to 1D vectors."""
        # .float() upcasts in case inputs are ever lower precision; flatten()
        # collapses any tensor shape into a single vector for the comparison.
        a_flat, b_flat = a.flatten().float(), b.flatten().float()
        if a_flat.numel() == 0:
            # Guard for an empty selection (e.g. a mask with no True entries).
            return float('nan')
        return torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()
    
    def interpolation_coefficient(baseline, donor, patched):
        """
        Scalar alpha describing where `patched` landed along the line from
        `baseline` to `donor`: alpha=0 means patched == baseline (patch had no
        effect), alpha=1 means patched == donor (patch fully took), and
        values outside [0, 1] indicate overshoot or movement in the opposite
        direction. Computed as the projection of the observed movement
        (patched - baseline) onto the baseline->donor direction vector,
        normalized by that direction's squared norm.
        """
        b, d, p = baseline.flatten().float(), donor.flatten().float(), patched.flatten().float()
        direction = d - b
        direction_norm_sq = torch.sum(direction ** 2)
        if direction_norm_sq == 0:
            # baseline == donor exactly: "distance toward donor" is undefined.
            return float('nan')
        movement = p - b
        return (torch.sum(movement * direction) / direction_norm_sq).item()
    
    def extract_masked_pairwise(baseline_z, donor_z, patched_z, mask, target_start, donor_hairpin_start):
        """
        Gather the z-vectors at the (i, j) pairs where `mask` is True, mapping
        each target-space index into donor-space (using the same offset
        convention as the patch hook) and dropping any pair whose donor-space
        index falls outside donor_z's bounds.
        """
        indices = torch.where(mask)
        t_i_indices, t_j_indices = indices[0], indices[1]
        if len(t_i_indices) == 0:
            # e.g. the 'pw_cross' mask can be empty when touch == intra.
            return None, None, None
        d_i_indices = t_i_indices - target_start + donor_hairpin_start
        d_j_indices = t_j_indices - target_start + donor_hairpin_start
        # The donor hairpin region may differ in length/position from the
        # target's patch region, so some mapped indices can land outside
        # donor_z -- filter those out rather than indexing out of bounds.
        valid_mask = (
            (d_i_indices >= 0) & (d_i_indices < donor_z.shape[1]) &
            (d_j_indices >= 0) & (d_j_indices < donor_z.shape[2])
        )
        if not valid_mask.any():
            return None, None, None
        t_i_valid, t_j_valid = t_i_indices[valid_mask], t_j_indices[valid_mask]
        d_i_valid, d_j_valid = d_i_indices[valid_mask], d_j_indices[valid_mask]
        # Index batch 0 (single-sequence forward passes throughout this script).
        return (baseline_z[0, t_i_valid, t_j_valid, :],
                donor_z[0, d_i_valid, d_j_valid, :],
                patched_z[0, t_i_valid, t_j_valid, :])
    
    def compute_pairwise_metrics(baseline_masked, donor_masked, patched_masked, prefix):
        """Cosine-similarity and interpolation-alpha metrics for one mask,
        with `prefix`-named keys; returns None-valued metrics if the mask
        selected no valid (in-bounds) pairs, keeping the output schema
        consistent across cases so all result rows share the same columns."""
        metrics = {}
        if baseline_masked is None:
            metrics[f'{prefix}_cos_baseline_donor'] = None
            metrics[f'{prefix}_cos_donor_patched'] = None
            metrics[f'{prefix}_interp_alpha'] = None
            return metrics
        metrics[f'{prefix}_cos_baseline_donor'] = cosine_sim(baseline_masked, donor_masked)
        metrics[f'{prefix}_cos_donor_patched'] = cosine_sim(donor_masked, patched_masked)
        metrics[f'{prefix}_interp_alpha'] = interpolation_coefficient(baseline_masked, donor_masked, patched_masked)
        return metrics
    
    # Sequence metrics
    results['seq_cos_baseline_donor'] = cosine_sim(baseline_s, donor_s)
    results['seq_cos_donor_patched'] = cosine_sim(donor_s, patched_s)
    results['seq_interp_alpha'] = interpolation_coefficient(baseline_s, donor_s, patched_s)
    
    # Pairwise metrics with different masks
    # (touch = any pair touching the patch region, intra = pairs fully inside
    # it, cross = touch-but-not-intra, i.e. "boundary" pairs between the
    # patched region and the rest of the protein)
    for mask, prefix in [(touch_mask, 'pw_touch'), (intra_mask, 'pw_intra'), (touch_mask & ~intra_mask, 'pw_cross')]:
        b, d, p = extract_masked_pairwise(baseline_z, donor_z, patched_z, mask, target_start, donor_hairpin_start)
        results.update(compute_pairwise_metrics(b, d, p, prefix))
    
    return results


def get_intervention_windows(patch_block: int, num_blocks: int = NUM_BLOCKS) -> List[Tuple[str, int, int]]:
    """Get intervention windows: early_10, late_10, full_remaining."""
    windows = [
        # 10-block window starting right at the patch point (capped at the last block).
        ('early_10', patch_block, min(patch_block + 9, num_blocks - 1)),
        # Last 10 blocks of the trunk (38 = num_blocks - 10); if patch_block is
        # already past 38, start the window at patch_block instead so it never
        # begins before the patch itself.
        ('late_10', max(patch_block, 38), num_blocks - 1),
        # Everything from the patch point through the end of the trunk.
        ('full_remaining', patch_block, num_blocks - 1),
    ]
    # Drop any degenerate/empty windows (start > end).
    return [(name, start, end) for name, start, end in windows if start <= end]


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment(
    df: pd.DataFrame,
    model: EsmForProteinFolding,
    tokenizer,
    device: str,
    output_dir: str,
    save_every: int = 10,
) -> pd.DataFrame:
    """Run the representation flow analysis experiment."""
    all_results = []
    
    # Distinct (patch_mode, patch_mask_mode, block_idx) combinations, used only
    # to drive the outer progress-reporting loop below; each combination may
    # still have multiple donor/target case rows in `df`, which are re-selected
    # into config_df just below.
    unique_configs = df[['patch_mode', 'patch_mask_mode', 'block_idx']].drop_duplicates()
    print(f"\nUnique patch configurations: {len(unique_configs)}")
    
    cases_processed = 0
    
    for _, config_row in unique_configs.iterrows():
        patch_mode = config_row['patch_mode']
        patch_mask_mode = config_row['patch_mask_mode']
        patch_block = int(config_row['block_idx'])
        
        config_df = df[
            (df['patch_mode'] == patch_mode) &
            (df['patch_mask_mode'] == patch_mask_mode) &
            (df['block_idx'] == patch_block)
        ]
        
        print(f"\n{'='*60}")
        print(f"Config: {patch_mode} @ block {patch_block}, {patch_mask_mode} mask")
        print(f"Cases: {len(config_df)}")
        print(f"{'='*60}")
        
        for idx, row in tqdm(config_df.iterrows(), total=len(config_df), desc="Cases"):
            target_seq = row['target_sequence']
            donor_seq = row['donor_sequence']
            target_start = int(row['target_patch_start'])
            target_end = int(row['target_patch_end'])
            donor_hairpin_start = int(row['donor_hairpin_start'])
            donor_hairpin_end = int(row['donor_hairpin_end'])
            
            touch_mask = create_pairwise_mask(
                donor_hairpin_start, donor_hairpin_end, len(donor_seq),
                target_start, target_end, len(target_seq), mode='touch',
            )
            intra_mask = create_pairwise_mask(
                donor_hairpin_start, donor_hairpin_end, len(donor_seq),
                target_start, target_end, len(target_seq), mode='intra',
            )
            patch_mask = touch_mask if patch_mask_mode == 'touch' else intra_mask
            
            # ================================================================
            # Step 1: Collect baseline representations
            # ================================================================
            with torch.no_grad():
                # add_special_tokens=False: ESMFold expects raw residue tokens
                # only (no BOS/EOS/CLS special tokens).
                target_inputs = tokenizer(target_seq, return_tensors='pt', add_special_tokens=False).to(device)
                
                # num_recycles=0: disable the structure module's recycling loop
                # so the trunk runs exactly once. This matters because the
                # collection hooks store one tensor per block_idx in a dict --
                # a second recycling pass would silently overwrite the first
                # pass's captured activations, which would break the "baseline"
                # captured here.
                with collect_all_representations(model) as baseline_collector:
                    _ = model(**target_inputs, num_recycles=0)  # output itself unused; only the hook side effects matter
            
            # Slice s down to just the target's patch region ([1, L, dim] -> [1, region_len, dim]).
            # Later compared elementwise against donor_s_blocks/patched_s_blocks,
            # which are trimmed to the corresponding donor-hairpin/target-patch
            # spans -- this assumes those two regions have matching lengths.
            baseline_s_blocks = {k: v[:, target_start:target_end, :] for k, v in baseline_collector.s_blocks.items()}
            # z is NOT sliced to a rectangular region (unlike s) since pairwise
            # metrics gather specific (i, j) entries via extract_masked_pairwise
            # rather than a contiguous span.
            baseline_z_blocks = dict(baseline_collector.z_blocks)
            # Convert from a (possibly sparse) dict into a dense NUM_BLOCKS-length
            # list via .get(i) (missing indices become None), so later code can
            # index directly by block_idx (see baseline_vals[idx] in TrunkInterventionHooks).
            baseline_pair2seq_list = [baseline_collector.pair2seq_bias_blocks.get(i) for i in range(NUM_BLOCKS)]
            baseline_seq2pair_list = [baseline_collector.seq2pair_update_blocks.get(i) for i in range(NUM_BLOCKS)]
            
            # ================================================================
            # Step 2: Collect donor representations
            # ================================================================
            with torch.no_grad():
                donor_inputs = tokenizer(donor_seq, return_tensors='pt', add_special_tokens=False).to(device)
                
                # pair2seq/seq2pair hooks skipped here: those intermediate
                # values are only needed as "freeze" targets for the TARGET
                # forward pass (Step 3), not for the donor.
                with collect_all_representations(model, collect_pair2seq=False, collect_seq2pair=False) as donor_collector:
                    _ = model(**donor_inputs, num_recycles=0)
            
            donor_s_blocks = {k: v[:, donor_hairpin_start:donor_hairpin_end, :] for k, v in donor_collector.s_blocks.items()}
            donor_z_blocks = dict(donor_collector.z_blocks)
            
            # ================================================================
            # Step 3: Run patching with various intervention conditions
            # ================================================================
            # NOTE: possible bug -- window_name is the outer loop and condition
            # (intervention_type/intervention_pathway) is the inner loop, but the
            # intervention_type=None ("no intervention") condition's forward pass
            # and metrics do not actually depend on intervention_start/end at all
            # (those are only used inside apply_intervention, which isn't entered
            # in the `else` branch below). So the identical baseline case is
            # recomputed and stored once per window_name (3x redundant compute,
            # and 3 duplicate 'none'-condition rows per case in the saved
            # results) instead of once per case.
            for window_name, intervention_start, intervention_end in get_intervention_windows(patch_block):
                for intervention_type, intervention_pathway in INTERVENTION_CONDITIONS:
                    # Create condition name
                    if intervention_type is None:
                        condition_name = 'none'
                    else:
                        condition_name = f'{intervention_type}_{intervention_pathway}'
                    
                    with torch.no_grad():
                        # Get donor representations for this patch block
                        # (this is what actually gets spliced into the target's
                        # forward pass by the patch hook, as opposed to the donor
                        # representations at other blocks, which are only used for metrics)
                        donor_s_for_patch = donor_s_blocks[patch_block].to(device)
                        donor_z_for_patch = donor_z_blocks[patch_block].to(device)
                        
                        # Set up context managers
                        # (constructed but not yet entered, so they can be
                        # combined into a single `with` statement below)
                        patch_ctx = apply_patch(
                            model,
                            patch_block=patch_block,
                            patch_mode=patch_mode,
                            donor_s=donor_s_for_patch,
                            donor_z=donor_z_for_patch,
                            target_start=target_start,
                            target_end=target_end,
                            donor_hairpin_start=donor_hairpin_start,
                            pairwise_mask=patch_mask,
                        )
                        
                        # Build nested context managers
                        if intervention_type is not None:
                            # Freeze to whichever pathway's baseline values match
                            # the pathway being intervened on.
                            baseline_vals = (baseline_seq2pair_list if intervention_pathway == 'seq2pair' 
                                           else baseline_pair2seq_list)
                            
                            intervention_ctx = apply_intervention(
                                model,
                                intervention_type=intervention_type,
                                intervention_pathway=intervention_pathway,
                                start_block=intervention_start,
                                end_block=intervention_end,
                                baseline_values=baseline_vals,
                            )
                            
                            # All three context managers active simultaneously for
                            # this one forward pass: apply the patch at patch_block,
                            # apply the freeze/zero intervention over the window,
                            # and collect s/z for every block of the patched run.
                            with patch_ctx, intervention_ctx, collect_all_representations(
                                model, collect_pair2seq=False, collect_seq2pair=False
                            ) as patched_collector:
                                _ = model(**target_inputs, num_recycles=0)
                        else:
                            # Baseline/no-intervention condition: patch only, no
                            # freeze/zero applied downstream.
                            with patch_ctx, collect_all_representations(
                                model, collect_pair2seq=False, collect_seq2pair=False
                            ) as patched_collector:
                                _ = model(**target_inputs, num_recycles=0)
                    
                    patched_s_blocks = {k: v[:, target_start:target_end, :] for k, v in patched_collector.s_blocks.items()}
                    patched_z_blocks = dict(patched_collector.z_blocks)
                    
                    # Compute metrics for each observation block
                    # (every block, not just the patch block or the intervention
                    # window, to see how the intervention's effect propagates or
                    # decays across the rest of the trunk's depth)
                    for obs_block in range(NUM_BLOCKS):
                        metrics = compute_representation_metrics(
                            baseline_s=baseline_s_blocks[obs_block],
                            baseline_z=baseline_z_blocks[obs_block],
                            donor_s=donor_s_blocks[obs_block],
                            donor_z=donor_z_blocks[obs_block],
                            patched_s=patched_s_blocks[obs_block],
                            patched_z=patched_z_blocks[obs_block],
                            target_start=target_start,
                            target_end=target_end,
                            donor_hairpin_start=donor_hairpin_start,
                            touch_mask=touch_mask,
                            intra_mask=intra_mask,
                        )
                        
                        # One result row per (case, window, condition, observation_block)
                        # combination -- this is the granularity of the final dataframe.
                        result = {
                            'patch_mode': patch_mode,
                            'patch_mask_mode': patch_mask_mode,
                            'patch_block': patch_block,
                            'condition': condition_name,
                            'intervention_type': intervention_type if intervention_type else 'none',
                            'intervention_pathway': intervention_pathway if intervention_pathway else 'none',
                            'window_name': window_name,
                            'intervention_start': intervention_start,
                            'intervention_end': intervention_end,
                            'observation_block': obs_block,
                            **metrics,
                        }
                        all_results.append(result)
            
            cases_processed += 1
            if cases_processed % save_every == 0:
                # Periodic checkpoint so a crash/interruption during a long sweep
                # doesn't lose all progress.
                pd.DataFrame(all_results).to_parquet(
                    os.path.join(output_dir, 'intervention_experiment_results_checkpoint.parquet'),
                    index=False
                )
                print(f"\n[Checkpoint] Saved {len(all_results)} results")
            
            # Free cached GPU memory between cases -- this loop runs many forward
            # passes per case (across windows x conditions), which can otherwise
            # accumulate/fragment GPU memory over a long-running sweep.
            torch.cuda.empty_cache()
    
    results_df = pd.DataFrame(all_results)
    results_df.to_parquet(os.path.join(output_dir, 'intervention_experiment_results.parquet'), index=False)
    print(f"\nSaved {len(results_df)} rows")
    
    return results_df


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_results(results_df: pd.DataFrame, output_dir: str):
    """Generate all analysis plots."""
    print("\nGenerating plots...")
    
    # 2x3 grid of (patch_block, patch_mode, patch_mask_mode) combos to plot,
    # one row for an early patch point (block 0) and one for a late patch
    # point (block 27), each shown for sequence/touch, pairwise/touch, and
    # pairwise/intra.
    conditions_grid = [
        [(0, 'sequence', 'touch'), (0, 'pairwise', 'touch'), (0, 'pairwise', 'intra')],
        [(27, 'sequence', 'touch'), (27, 'pairwise', 'touch'), (27, 'pairwise', 'intra')],
    ]
    
    # Consistent color/linestyle per condition across all subplots, so the
    # same condition is visually identifiable across every panel.
    colors = {
        'none': 'black',
        'freeze_seq2pair': 'blue',
        'freeze_pair2seq': 'cyan',
        'zero_seq2pair': 'red',
        'zero_pair2seq': 'orange',
    }
    linestyles = {
        'none': '-',
        'freeze_seq2pair': '--',
        'freeze_pair2seq': '--',
        'zero_seq2pair': ':',
        'zero_pair2seq': ':',
    }
    
    # Defensive filter: window_name is always one of early_10/late_10/full_remaining
    # in data produced by run_experiment (never the literal string 'none'), so
    # this has no practical effect on the current schema.
    window_names = [w for w in results_df['window_name'].unique() if w != 'none']
    
    for window_name in window_names:
        # Rows for this specific window, plus every baseline/no-intervention row
        # (condition == 'none') so the unpatched baseline curve is overlaid for
        # comparison in every per-window plot.
        window_df = results_df[
            (results_df['window_name'] == window_name) | 
            (results_df['condition'] == 'none')
        ].copy()
        
        # Used only to look up this window's intervention_start/end (for shading)
        # and to skip the window entirely if it has no data.
        sample_rows = results_df[results_df['window_name'] == window_name]
        if len(sample_rows) == 0:
            continue
        
        # Plot sequence alpha
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        for row_idx, row_conditions in enumerate(conditions_grid):
            for col_idx, (patch_block, patch_mode, patch_mask_mode) in enumerate(row_conditions):
                ax = axes[row_idx, col_idx]
                
                # Look up this panel's actual intervention window bounds from
                # the data (falls back to (patch_block, 47) if no rows exist,
                # just so axvspan below always has valid bounds).
                subset_sample = sample_rows[sample_rows['patch_block'] == patch_block]
                if len(subset_sample) > 0:
                    intervention_start = int(subset_sample.iloc[0]['intervention_start'])
                    intervention_end = int(subset_sample.iloc[0]['intervention_end'])
                else:
                    intervention_start, intervention_end = patch_block, 47
                
                for condition in colors.keys():
                    subset = window_df[
                        (window_df['patch_block'] == patch_block) &
                        (window_df['patch_mode'] == patch_mode) &
                        (window_df['patch_mask_mode'] == patch_mask_mode) &
                        (window_df['condition'] == condition)
                    ]
                    
                    if len(subset) == 0:
                        continue
                    
                    # Average this condition's metric across all cases sharing
                    # this (patch_block, patch_mode, patch_mask_mode, condition)
                    # combo, per observation_block, to get one mean trend curve.
                    avg = subset.groupby('observation_block').mean(numeric_only=True).reset_index()
                    ax.plot(avg['observation_block'], avg['seq_interp_alpha'],
                            label=condition, color=colors[condition],
                            linestyle=linestyles[condition], linewidth=2)
                
                ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)  # alpha=0 reference (no effect)
                ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)  # alpha=1 reference (full donor replacement)
                ax.axvline(x=patch_block, color='black', linestyle='--', linewidth=2)  # marks where the patch was applied
                ax.axvspan(intervention_start, intervention_end, alpha=0.1, color='yellow')  # shades the intervention window
                
                ax.set_xlabel('Observation Block', fontsize=11)
                ax.set_ylabel('Sequence α', fontsize=11)
                ax.set_title(f'Block {patch_block}, {patch_mode.upper()}, {patch_mask_mode}', fontsize=12)
                ax.legend(loc='best', fontsize=7)
                ax.set_ylim(-0.1, 1.2)
                ax.grid(alpha=0.3)
        
        plt.suptitle(f'SEQUENCE Interpolation α: Freezing vs Zero Ablation\n{window_name}', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sequence_alpha_{window_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot pairwise intra alpha
        # (mirrors the sequence-alpha block above, plotting
        # 'pw_intra_interp_alpha' instead of 'seq_interp_alpha')
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        for row_idx, row_conditions in enumerate(conditions_grid):
            for col_idx, (patch_block, patch_mode, patch_mask_mode) in enumerate(row_conditions):
                ax = axes[row_idx, col_idx]
                
                subset_sample = sample_rows[sample_rows['patch_block'] == patch_block]
                if len(subset_sample) > 0:
                    intervention_start = int(subset_sample.iloc[0]['intervention_start'])
                    intervention_end = int(subset_sample.iloc[0]['intervention_end'])
                else:
                    intervention_start, intervention_end = patch_block, 47
                
                for condition in colors.keys():
                    subset = window_df[
                        (window_df['patch_block'] == patch_block) &
                        (window_df['patch_mode'] == patch_mode) &
                        (window_df['patch_mask_mode'] == patch_mask_mode) &
                        (window_df['condition'] == condition)
                    ]
                    
                    if len(subset) == 0:
                        continue
                    
                    avg = subset.groupby('observation_block').mean(numeric_only=True).reset_index()
                    ax.plot(avg['observation_block'], avg['pw_intra_interp_alpha'],
                            label=condition, color=colors[condition],
                            linestyle=linestyles[condition], linewidth=2)
                
                ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
                ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
                ax.axvline(x=patch_block, color='black', linestyle='--', linewidth=2)
                ax.axvspan(intervention_start, intervention_end, alpha=0.1, color='yellow')
                
                ax.set_xlabel('Observation Block', fontsize=11)
                ax.set_ylabel('PW Intra α', fontsize=11)
                ax.set_title(f'Block {patch_block}, {patch_mode.upper()}, {patch_mask_mode}', fontsize=12)
                ax.legend(loc='best', fontsize=7)
                ax.set_ylim(-0.1, 1.2)
                ax.grid(alpha=0.3)
        
        plt.suptitle(f'PAIRWISE INTRA Interpolation α: Freezing vs Zero Ablation\n{window_name}', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pairwise_alpha_{window_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved plots to {output_dir}/")


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    """Parse command-line arguments for the representation flow analysis script."""
    parser = argparse.ArgumentParser(
        description='Representation flow analysis with freeze/zero interventions'
    )
    # NOTE: possible bug -- help text says "parquet dataset" but the default
    # path is a .csv and main() below loads it with pd.read_csv, not a parquet
    # reader; the help string appears to be a stale/copy-pasted description.
    parser.add_argument('--dataset', type=str, default='data/single_block_patching_successes.csv',
                        help='Path to input parquet dataset')
    parser.add_argument('--output', type=str, default='representation_analysis_results',
                        help='Output directory')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N cases')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--plot_only', action='store_true',
                        help='Only generate plots from existing results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def main():
    """Entry point: run the intervention experiment end-to-end (or, with
    --plot_only, just regenerate plots from a previously saved results file)."""
    args = parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.plot_only:
        # Skip loading the model/dataset and re-running the experiment entirely;
        # just reload previously saved results and regenerate plots from them.
        results_path = os.path.join(args.output, 'intervention_experiment_results.parquet')
        if os.path.exists(results_path):
            results_df = pd.read_parquet(results_path)
            print(f"Loaded {len(results_df)} rows from {results_path}")
            plot_results(results_df, args.output)
        else:
            print(f"No results found at {results_path}")
        return
    
    model, tokenizer = load_esmfold(device)  # shared loader; picks precision automatically (see src/utils/model_utils.py)
    
    print(f"\nLoading dataset from {args.dataset}...")
    df = pd.read_csv(args.dataset)
    print(f"Loaded {len(df)} rows")
    print("\nBreakdown:")
    print(df.groupby(['patch_mode', 'patch_mask_mode', 'block_idx']).size())
    
    results_df = run_experiment(
        df=df,
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=args.output,
        save_every=args.save_every,
    )
    
    if len(results_df) > 0:
        plot_results(results_df, args.output)
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == '__main__':
    main()