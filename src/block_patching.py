"""
Single-Block Activation Patching
================================

Identifies which ESMFold trunk blocks encode hairpin structural information by
patching activations one block at a time. For each of the 48 trunk blocks,
transplants the (s, z) representations from a hairpin-containing donor sequence
into a helical acceptor sequence and measures whether a hairpin forms.

This experiment reveals the temporal dynamics of structure formation: early blocks
(0-10) show the strongest patching effects, indicating that hairpin geometry is
established early in the folding trunk's forward pass.

Usage:
    python block_patching.py --parquet patching_dataset.parquet --output_dir results/
"""

import argparse
import os
import types
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import types

# Add project root (parent of src/) to path so `src.*` imports work without PYTHONPATH
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from transformers import EsmForProteinFolding, AutoTokenizer
# Note: categorical_lddt, compute_predicted_aligned_error, compute_tm, and
# make_atom14_masks are imported below but not directly used in this file
# (EsmFoldingTrunk and EsmForProteinFoldingOutput are used).
from transformers.models.esm.modeling_esmfold import (
    categorical_lddt,
    EsmFoldingTrunk,
    EsmForProteinFoldingOutput,
)
from transformers.models.esm.openfold_utils import (
    compute_predicted_aligned_error,
    compute_tm,
    make_atom14_masks,
)
from transformers.utils import ContextManagers

from src.utils.trunk_utils import detect_hairpins
# Collection classes imported from the shared utils module (unlike
# module_patching.py, which keeps its own local copies of these).
from src.utils.representation_utils import CollectedRepresentations, TrunkHooks
from src.utils.model_utils import load_esmfold  # shared model+tokenizer loader (handles device/precision setup)


# ============================================================================
# PART 1: COLLECTION CONVENIENCE FUNCTION
# ============================================================================

def run_and_collect(
    model,
    tokenizer,
    device: str,
    sequence: str,
    num_recycles: int = 0,
) -> Tuple[EsmForProteinFoldingOutput, CollectedRepresentations]:
    """
    Run model and collect trunk representations using hooks.
    """
    # Only trunk (s, z) hooks are needed here -- unlike module_patching.py's
    # run_and_collect, single-block patching never touches the ESM encoder or
    # structure-module IPA, so those collectors are skipped entirely.
    collector = CollectedRepresentations()
    
    trunk_hooks = TrunkHooks(model.trunk, collector)
    trunk_hooks.register(collect_s=True, collect_z=True)
    
    try:
        with torch.no_grad():
            # add_special_tokens=False: tokenizer's own CLS/EOS are omitted so
            # positions line up 1:1 with residue indices for the trunk's
            # mask/residx (see module_patching.py's run_and_collect for the
            # fuller note on ESMFold's separate internal BOS/EOS handling,
            # which doesn't apply here since only trunk, not ESM, is patched).
            inputs = tokenizer(sequence, return_tensors='pt', add_special_tokens=False).to(device)
            outputs = model(**inputs, num_recycles=num_recycles)
    finally:
        # Always remove hooks, even if the forward pass above raised.
        trunk_hooks.remove()
    
    return outputs, collector


# ============================================================================
# PART 2: MASKS
# ============================================================================

def create_pairwise_mask(
    donor_start: int,
    donor_end: int,
    donor_len: int,
    target_start: int,
    target_end: int,
    target_len: int,
    mode: str,
) -> torch.Tensor:
    """Create pairwise patch mask."""
    # Boolean mask over the target's [L, L] pairwise coordinate space.
    mask = torch.zeros(target_len, target_len, dtype=torch.bool)
    
    if mode == "intra":
        # Only patch within the hairpin/patch region itself.
        mask[target_start:target_end, target_start:target_end] = True
        
    elif mode in ("touch", "hole"):
        # How far the patched window can extend beyond the core region on
        # each side while staying within both the donor's and target's
        # sequence bounds (the smaller of the two limits keeps the coordinate
        # mapping back to the donor, below, valid).
        left_extent = min(donor_start, target_start)
        right_extent = min(donor_len - donor_end, target_len - target_end)
        
        transport_start = target_start - left_extent
        transport_end = target_end + right_extent
        
        # Cross ("+") pattern: rows = core patch region x cols = extended
        # window, unioned with its transpose -- selects (i, j) pairs where
        # either i or j is in the core region and the other spans the
        # flanking context (interactions between patch and surrounding
        # residues, not just interactions within the patch itself).
        mask[target_start:target_end, transport_start:transport_end] = True
        mask[transport_start:transport_end, target_start:target_end] = True
        
        if mode == "hole":
            # "hole" = the cross minus the "intra" square: isolates just the
            # core-to-context pairs, excluding core-to-core pairs.
            mask[target_start:target_end, target_start:target_end] = False
    
    return mask


# ============================================================================
# PART 3: INTERVENTION - SINGLE BLOCK TRUNK PATCHING
# ============================================================================

def make_trunk_single_block_patch_forward(
    donor_s_blocks: Dict[int, torch.Tensor],
    donor_z_blocks: Dict[int, torch.Tensor],
    target_start: int,
    target_end: int,
    donor_start: int,
    pairwise_mask: torch.Tensor,
    patch_mode: str,
    target_block: int,
):
    """
    Create a trunk forward that patches a SINGLE block with donor representations.
    
    Args:
        donor_s_blocks: Dict mapping block_idx -> [B, patch_len, D_s] sequence repr
        donor_z_blocks: Dict mapping block_idx -> [B, L, L, D_z] FULL pairwise repr
        target_start, target_end: Where to patch in target
        donor_start: Where donor region starts (for pairwise coordinate mapping)
        pairwise_mask: Boolean mask for which (i,j) pairs to patch
        patch_mode: 'sequence', 'pairwise', or 'both'
        target_block: Which block index to patch (only this block gets patched)
    
    Returns:
        A forward function to be bound to model.trunk
    """
    
    # Like module_patching.py's make_trunk_all_block_patch_forward, this whole
    # `forward` reproduces the original EsmFoldingTrunk.forward
    # (transformers.models.esm.modeling_esmfold) so the patched run stays
    # numerically identical to a normal forward pass except at the patched
    # positions. The one difference from the all-block version is
    # apply_patch's early-return below, which restricts the intervention to a
    # single block -- this is what lets the experiment isolate each block's
    # individual causal contribution instead of patching all 48 at once.
    def forward(self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles):
        # seq_feats: [B, L, c_s] sequence repr; pair_feats: [B, L, L, c_z] pairwise repr.
        device = seq_feats.device
        s_s_0, s_z_0 = seq_feats, pair_feats

        if no_recycles is None:
            no_recycles = self.config.max_recycles
        else:
            # First "recycle" is just the standard forward pass through the model.
            no_recycles += 1

        def apply_patch(block_idx, s, z):
            """Apply donor patch at this block (only if it's the target block)."""
            # Core single-block restriction: every other block index is a no-op,
            # so only `target_block`'s output ever gets overwritten.
            if block_idx != target_block:
                return s, z
            
            # Patch sequence
            # (overwrites s's [target_start:target_end] span with the donor's s
            # at this same block index; requires equal-length donor/target regions)
            if patch_mode in ('both', 'sequence') and block_idx in donor_s_blocks:
                donor_s = donor_s_blocks[block_idx].to(s.device, dtype=s.dtype)
                s[:, target_start:target_end, :] = donor_s
            
            # Patch pairwise
            # (for every (i, j) selected by pairwise_mask, defined in the
            # target's coordinate space, copies the donor's z value at the
            # corresponding donor-space coordinate, shifted by target_start -> donor_start)
            if patch_mode in ('both', 'pairwise') and block_idx in donor_z_blocks:
                donor_z = donor_z_blocks[block_idx].to(z.device, dtype=z.dtype)
                mask_dev = pairwise_mask.to(z.device)
                
                indices = torch.where(mask_dev)
                for i in range(len(indices[0])):
                    ti, tj = indices[0][i].item(), indices[1][i].item()
                    di = ti - target_start + donor_start
                    dj = tj - target_start + donor_start
                    # Skip if the mapped donor coordinate falls out of bounds.
                    if 0 <= di < donor_z.shape[1] and 0 <= dj < donor_z.shape[2]:
                        z[:, ti, tj, :] = donor_z[:, di, dj, :]
            
            return s, z

        def trunk_iter(s, z, residx, mask):
            z = z + self.pairwise_positional_embedding(residx, mask=mask)
            for block_idx, block in enumerate(self.blocks):
                s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size)
                # Intervention point: only fires for block_idx == target_block
                # (see apply_patch's early return above).
                s, z = apply_patch(block_idx, s, z)
            return s, z

        # Standard recycle loop (unmodified from the original trunk forward):
        # each recycle pass's final s/z and a coarse predicted-distance
        # histogram feed back additively into the next pass's inputs.
        s_s, s_z = s_s_0, s_z_0
        recycle_s = torch.zeros_like(s_s)
        recycle_z = torch.zeros_like(s_z)
        recycle_bins = torch.zeros(*s_z.shape[:-1], device=device, dtype=torch.int64)

        for recycle_idx in range(no_recycles):
            # Only the final recycle needs gradients; earlier ones are pure
            # feed-forward inputs to the next iteration.
            with ContextManagers([] if recycle_idx == no_recycles - 1 else [torch.no_grad()]):
                recycle_s = self.recycle_s_norm(recycle_s.detach()).to(device)
                recycle_z = self.recycle_z_norm(recycle_z.detach()).to(device)
                recycle_z += self.recycle_disto(recycle_bins.detach()).to(device)

                s_s, s_z = trunk_iter(s_s_0 + recycle_s, s_z_0 + recycle_z, residx, mask)

                structure = self.structure_module(
                    {"single": self.trunk2sm_s(s_s), "pair": self.trunk2sm_z(s_z)},
                    true_aa, mask.float(),
                )

                recycle_s, recycle_z = s_s, s_z
                # Bin predicted CB-CB distances (3.375-21.375 A, 15 bins) into a
                # coarse distogram fed back into recycle_z on the next iteration.
                recycle_bins = EsmFoldingTrunk.distogram(
                    structure["positions"][-1][:, :, :3],
                    3.375, 21.375, self.recycle_bins,
                )

        structure["s_s"] = s_s
        structure["s_z"] = s_z
        return structure
    
    return forward


@contextmanager
def patch_trunk_single_block(
    model,
    donor_s_blocks: Dict[int, torch.Tensor],
    donor_z_blocks: Dict[int, torch.Tensor],
    target_start: int,
    target_end: int,
    donor_start: int,
    pairwise_mask: torch.Tensor,
    patch_mode: str,
    target_block: int,
):
    """
    Context manager for trunk single-block patching.
    
    Usage:
        with patch_trunk_single_block(model, donor_s, donor_z, ..., target_block=5):
            outputs = model(**inputs)
    """
    # The intervention needs to happen *inside* the trunk's recycling loop
    # (between specific blocks), so a plain forward hook can't do it -- instead
    # we temporarily swap out model.trunk's bound forward method.
    original = model.trunk.forward
    
    patched_forward = make_trunk_single_block_patch_forward(
        donor_s_blocks, donor_z_blocks,
        target_start, target_end, donor_start,
        pairwise_mask, patch_mode, target_block,
    )
    # types.MethodType binds the plain function as a method of this specific
    # model.trunk instance so `self` resolves correctly inside it.
    model.trunk.forward = types.MethodType(patched_forward, model.trunk)
    
    try:
        # Enter: forward is already monkey-patched by this point.
        yield
    finally:
        # Exit (normal or exception): always restore the original forward.
        model.trunk.forward = original


# ============================================================================
# PART 4: ANALYSIS UTILITIES
# ============================================================================

def evaluate_hairpin(
    outputs: EsmForProteinFoldingOutput,
    model,
    target_start: int,
    target_end: int,
) -> Dict[str, Any]:
    """Evaluate hairpin formation and structure quality."""
    # Second return value (list of hairpin coordinate tuples) is discarded --
    # only presence/absence is needed for this metric.
    hairpin_found, _ = detect_hairpins(outputs, model)
    
    # outputs.plddt: [B, L] per-residue confidence in [0, 1]; [0] indexes the
    # single-sequence batch dim.
    plddt = outputs.plddt[0].cpu().numpy()
    ptm = outputs.ptm.item() if outputs.ptm is not None else None
    
    return {
        'hairpin_found': hairpin_found,
        'mean_plddt': float(plddt.mean()),
        'patch_region_plddt': float(plddt[target_start:target_end].mean()),
        'ptm': ptm,
    }


# ============================================================================
# PART 5: PLOTTING
# ============================================================================

def generate_summary_plots(results_df: pd.DataFrame, output_dir: str):
    """Generate summary plots from results."""
    # NOTE: possible bug -- this function is defined but never called anywhere
    # in this file; run_experiment_on_dataset() below calls
    # generate_basic_plots() directly (both for periodic flushes and the final
    # summary), bypassing this function entirely. Also, even if it were called,
    # the import below would always raise ImportError anyway: src/block_plotting.py
    # (as currently written) defines plot_formation_rate_lines/generate_all_plots/
    # print_summary_statistics, not plot_success_rates/plot_success_rates_grouped/
    # plot_plddt_comparison/plot_summary_table -- so the except-ImportError
    # fallback to generate_basic_plots() below would trigger unconditionally.
    try:
        # Bare `block_plotting` (not `src.block_plotting`) only resolves if the
        # script's own directory happens to be on sys.path.
        from block_plotting import (
            plot_success_rates,
            plot_success_rates_grouped,
            plot_plddt_comparison,
            plot_summary_table,
        )
        
        output_path = Path(output_dir)
        
        plot_success_rates(results_df, output_path)
        plot_success_rates_grouped(results_df, output_path)
        plot_plddt_comparison(results_df, output_path)
        plot_summary_table(results_df, output_path)
        
        print(f"  Plots saved to {output_dir}")
        
    except ImportError as e:
        print(f"  Warning: Could not import plotting module: {e}")
        print("  Generating basic plots instead...")
        generate_basic_plots(results_df, output_dir)


def generate_basic_plots(results_df: pd.DataFrame, output_dir: str):
    """Generate basic plots if plotting module not available."""
    import matplotlib.pyplot as plt
    
    if len(results_df) == 0:
        return
    
    # Plot: Hairpin detection rate by block
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # hairpin_found is boolean, so groupby(...).mean() gives the fraction of
    # True values per block -- i.e. each block's hairpin-formation success rate.
    block_success = results_df.groupby('block_idx')['hairpin_found'].mean()
    ax.bar(block_success.index, block_success.values, color='steelblue')
    ax.set_xlabel('Block Index')
    ax.set_ylabel('Hairpin Detection Rate')
    ax.set_title('Hairpin Detection Rate by Block')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)  # 50% reference line
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hairpin_by_block.png"), dpi=150)
    plt.close()
    
    # Plot: Hairpin detection by block × patch_mode
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # pivot_table reshapes the long-format results into a block_idx x
    # patch_mode matrix of mean success rates -- one row per block, one
    # column per patch_mode -- which pandas' .plot(kind='bar') below turns
    # directly into a grouped bar chart (one group of bars per block).
    pivot = results_df.pivot_table(
        values='hairpin_found', 
        index='block_idx', 
        columns='patch_mode', 
        aggfunc='mean'
    )
    pivot.plot(kind='bar', ax=ax, width=0.8)
    ax.set_xlabel('Block Index')
    ax.set_ylabel('Hairpin Detection Rate')
    ax.set_title('Hairpin Detection by Block × Patch Mode')
    ax.set_ylim(0, 1)
    ax.legend(title='Patch Mode')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hairpin_by_block_mode.png"), dpi=150)
    plt.close()
    
    # Plot: Heatmap of success by block × mask_mode
    # One subplot per unique patch_mode value.
    fig, axes = plt.subplots(1, len(results_df['patch_mode'].unique()), figsize=(16, 5))
    # plt.subplots(1, N) returns a bare Axes (not an array) when N == 1, which
    # isn't iterable; normalize that edge case into a list so the zip() loop
    # below works the same regardless of how many patch_modes are present.
    if not hasattr(axes, '__len__'):
        axes = [axes]
    
    for ax, patch_mode in zip(axes, results_df['patch_mode'].unique()):
        subset = results_df[results_df['patch_mode'] == patch_mode]
        # Per patch_mode, reshape into a patch_mask_mode x block_idx grid of
        # mean success rates for the heatmap below.
        pivot = subset.pivot_table(
            values='hairpin_found',
            index='patch_mask_mode',
            columns='block_idx',
            aggfunc='mean'
        )
        
        # Deferred import: seaborn is only needed for this fallback plotting
        # path, so it's imported here rather than at module load time.
        import seaborn as sns
        # vmin/vmax fixed to [0, 1] (rather than auto-scaled) so the color
        # scale is directly comparable across all patch_mode subplots.
        sns.heatmap(pivot, ax=ax, cmap='RdYlGn', vmin=0, vmax=1, 
                    annot=True, fmt='.2f', cbar_kws={'label': 'Success Rate'})
        ax.set_title(f'Patch Mode: {patch_mode}')
        ax.set_xlabel('Block Index')
        ax.set_ylabel('Mask Mode')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_block_mask.png"), dpi=150)
    plt.close()
    
    print(f"  Basic plots saved to {output_dir}")


# ============================================================================
# PART 6: MAIN EXPERIMENT
# ============================================================================

def run_experiment_on_dataset(
    parquet_path: str,
    output_dir: str,
    patch_modes: List[str] = ["sequence", "pairwise", "both"],
    patch_mask_modes: List[str] = ["intra", "touch"],
    save_pdbs: bool = False,
    n_cases: Optional[int] = None,
    device: Optional[str] = None,
    flush_every: int = 20,
) -> pd.DataFrame:
    """
    Run single-block patching experiments on the patching dataset.
    
    Tests each block individually to identify which blocks matter most.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if save_pdbs:
        pdb_dir = os.path.join(output_dir, "pdbs")
        os.makedirs(pdb_dir, exist_ok=True)
    else:
        pdb_dir = None
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model, tokenizer = load_esmfold(device)
    
    # Load dataset
    print(f"Loading patching dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    # This "dataset" is actually the *results* parquet produced by
    # module_patching.py's all-block experiment (see this file's --parquet
    # default: "all_block_patching_results.parquet"). Filtering to
    # patch_module == "trunk", hairpin_found == True, and patch_mask_mode ==
    # "intra" keeps only the cases where whole-trunk patching already
    # succeeded at forming a hairpin -- i.e. this single-block sweep only
    # re-runs cases known to work at the whole-trunk level, to find *which*
    # individual block(s) are responsible for that success.
    df = df[(df["patch_module"] == "trunk") & (df["hairpin_found"] == True) & (df['patch_mask_mode'] == 'intra')]
    if n_cases is not None:
        df = df.head(n_cases)
    print(f"Running {len(df)} cases")
    
    # Results storage
    all_results = []
    results_path = os.path.join(output_dir, "single_block_patching_results.parquet")
    
    # Columns to preserve from input
    # (carried through into every result row so each experiment result stays
    # traceable back to its source case)
    preserve_cols = [
        "target_name", "target_sequence", "target_length",
        "loop_idx", "loop_start", "loop_end", "loop_length", "loop_sequence",
        "target_patch_start", "target_patch_end", "patch_length",
        "donor_pdb", "donor_sequence", "donor_length",
        "donor_hairpin_start", "donor_hairpin_end", "donor_hairpin_length",
        "donor_hairpin_sequence",
        "donor_strand1_length", "donor_strand2_length",
        "donor_loop_sequence", "donor_loop_length",
        "donor_handedness_magnitude", "loop_similarity",
    ]
    
    for case_idx, row in tqdm(df.iterrows(), total=len(df), desc="Cases"):
        target_seq = row["target_sequence"]
        donor_seq = row["donor_sequence"]
        
        target_start = int(row["target_patch_start"])
        target_end = int(row["target_patch_end"])
        donor_start = int(row["donor_hairpin_start"])
        donor_end = int(row["donor_hairpin_end"])
        
        print(f"\nCase {case_idx}: loop {row.get('loop_idx', '?')} <- {row.get('donor_pdb', '?')}")
        print(f"  Target patch: [{target_start}:{target_end})")
        print(f"  Donor hairpin: [{donor_start}:{donor_end})")
        
        # Preserve metadata
        # (guarded by `if col in row.index` so this still works against input
        # data that is missing an optional column)
        case_meta = {col: row[col] for col in preserve_cols if col in row.index}
        case_meta["case_idx"] = case_idx
        case_meta["target_start"] = target_start
        case_meta["target_end"] = target_end
        case_meta["donor_start"] = donor_start
        case_meta["donor_end"] = donor_end
        
        # =====================================================================
        # COLLECT DONOR REPRESENTATIONS
        # =====================================================================
        print("  Collecting donor representations...")
        donor_outputs, donor_collected = run_and_collect(
            model, tokenizer, device, donor_seq
        )
        
        # Extract hairpin region for sequence representations
        # (slices each [B, L, D_s] tensor down to just the donor's hairpin span,
        # which is spliced directly into the target's patch region elsewhere --
        # assumes the two regions are equal length)
        donor_s_region = {
            k: v[:, donor_start:donor_end, :] 
            for k, v in donor_collected.s_blocks.items()
        }
        # Keep full z_blocks for pairwise patching
        # (mask handles coordinate mapping back to the relevant donor span)
        donor_z_full = donor_collected.z_blocks
        
        num_blocks = len(donor_s_region)  # 48 for the full folding trunk
        print(f"    Trunk blocks: {num_blocks}")
        
        # donor_outputs (full structure prediction) no longer needed once its
        # intermediate representations are extracted; free GPU memory before
        # the many acceptor forward passes run below.
        del donor_outputs
        torch.cuda.empty_cache()
        
        # Create pairwise masks
        pairwise_masks = {}
        for mask_mode in patch_mask_modes:
            pairwise_masks[mask_mode] = create_pairwise_mask(
                donor_start=donor_start,
                donor_end=donor_end,
                donor_len=len(donor_seq),
                target_start=target_start,
                target_end=target_end,
                target_len=len(target_seq),
                mode=mask_mode,
            )
        
        case_results = []
        
        # =====================================================================
        # SINGLE-BLOCK PATCHING EXPERIMENTS
        # =====================================================================
        # Triple sweep: for every (patch_mode, mask_mode) combination, patch
        # each of the 48 blocks individually (one model forward pass per
        # block) to build a per-block causal profile for this case.
        for patch_mode in patch_modes:
            for mask_mode in patch_mask_modes:
                # Skip redundant mask modes for sequence-only patching
                # (pairwise_mask only affects z (pairwise) patching, so it has
                # no effect when patch_mode == "sequence" -- run once using the
                # first mask mode as a placeholder label)
                if patch_mode == "sequence" and mask_mode != patch_mask_modes[0]:
                    continue
                
                for block_idx in range(num_blocks):
                    with patch_trunk_single_block(
                        model, donor_s_region, donor_z_full,
                        target_start, target_end, donor_start,
                        pairwise_masks[mask_mode], patch_mode, block_idx
                    ):
                        with torch.no_grad():
                            inputs = tokenizer(target_seq, return_tensors='pt', add_special_tokens=False).to(device)
                            outputs = model(**inputs, num_recycles=0)
                    
                    eval_result = evaluate_hairpin(
                        outputs, model, target_start, target_end
                    )
                    
                    result = {
                        "block_idx": block_idx,
                        "patch_mode": patch_mode,
                        "patch_mask_mode": mask_mode,
                        **eval_result,
                    }
                    result.update(case_meta)
                    
                    if save_pdbs and pdb_dir:
                        pdb_str = model.output_to_pdb(outputs)[0]
                        pdb_filename = f"case{case_idx}_block{block_idx}_{patch_mode}_{mask_mode}.pdb"
                        with open(os.path.join(pdb_dir, pdb_filename), 'w') as f:
                            f.write(pdb_str)
                    
                    case_results.append(result)
                    
                    del outputs
                    torch.cuda.empty_cache()
                
                # Print progress for this mode
                # Quick in-memory tally (rather than re-querying a DataFrame):
                # filters case_results down to just this (patch_mode,
                # mask_mode) combination's block sweep and divides by
                # num_blocks -- relies on the inner loop above having appended
                # exactly num_blocks matching entries for this combination.
                success_rate = sum(r['hairpin_found'] for r in case_results if r['patch_mode'] == patch_mode and r['patch_mask_mode'] == mask_mode) / num_blocks
                print(f"    {patch_mode}/{mask_mode}: {success_rate:.1%} success across {num_blocks} blocks")
        
        all_results.extend(case_results)
        
        # Flush results and regenerate plots every flush_every cases
        # (long-running experiment -- many cases x 2-3 patch_modes x 48 blocks
        # -- so periodic persistence avoids losing everything to a crash/OOM
        # partway through, and lets progress be checked via the regenerated
        # plots without waiting for the full run to finish)
        cases_completed = case_idx + 1
        if cases_completed % flush_every == 0 or cases_completed == len(df):
            print(f"\n  Flushing results ({cases_completed} cases completed)...")
            interim_df = pd.DataFrame(all_results)
            interim_df.to_parquet(results_path, index=False)
            generate_basic_plots(interim_df, output_dir)
        
        # Clean up
        # (this case's donor tensors, before moving on to the next case)
        del donor_collected, donor_s_region, donor_z_full
        torch.cuda.empty_cache()
    
    # Final summary
    results_df = pd.DataFrame(all_results)
    results_df.to_parquet(results_path, index=False)
    generate_basic_plots(results_df, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {results_path}")
    print(f"Total experiments: {len(results_df)}")
    print(f"\nHairpin detection rate by block_idx:")
    # hairpin_found is boolean, so groupby(...).mean() computes the fraction
    # of True values per block -- each block's hairpin-formation success rate
    # (this is the data behind the "blocks 0-10 matter most" finding).
    print(results_df.groupby("block_idx")["hairpin_found"].mean())
    print(f"\nHairpin detection rate by patch_mode:")
    print(results_df.groupby("patch_mode")["hairpin_found"].mean())
    
    return results_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    """CLI entry point: parse args and run the single-block patching experiment."""
    parser = argparse.ArgumentParser(
        description="Run single-block ESMFold patching experiments (refactored)"
    )
    parser.add_argument(
        "--parquet", type=str, default=os.path.join(_PROJECT_ROOT, "data", "all_block_patching_results.parquet"),
        help="Path to patching_dataset.parquet"
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.path.join(_PROJECT_ROOT, "results", "single_block_v2"),
        help="Output directory"
    )
    parser.add_argument(
        "--patch_modes", nargs="+", default=["sequence", "pairwise", "both"],
        help="Patch modes for trunk"
    )
    parser.add_argument(
        "--mask_modes", nargs="+", default=["intra", "touch"],
        help="Pairwise mask modes"
    )
    parser.add_argument(
        "--n_cases", type=int, default=None,
        help="Number of cases to run"
    )
    parser.add_argument(
        "--save_pdbs", action="store_true",
        help="Save PDB structures"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device"
    )
    parser.add_argument(
        "--flush_every", type=int, default=20,
        help="Save results and regenerate plots every N cases"
    )
    
    args = parser.parse_args()
    
    results_df = run_experiment_on_dataset(
        parquet_path=args.parquet,
        output_dir=args.output_dir,
        patch_modes=args.patch_modes,
        patch_mask_modes=args.mask_modes,
        save_pdbs=args.save_pdbs,
        n_cases=args.n_cases,
        device=args.device,
        flush_every=args.flush_every,
    )
    
    print(f"\nDone! Results in {args.output_dir}")


if __name__ == "__main__":
    main()