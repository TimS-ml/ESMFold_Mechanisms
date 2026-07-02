#!/usr/bin/env python
"""
Pairwise Distance Steering via Learned Probes
==============================================

Induces hairpin formation by steering the pairwise representation (z) toward
directions that a trained distance probe associates with closer CA-CA distances.

Approach:
1. Train Ridge regression probes to predict CA-CA distance from z[i,j]
2. Use the probe weights as steering directions (negative gradient = closer)
3. Apply sliding window interventions on helical sequences
4. Measure H-bond formation rate vs intervention block position

This provides a complementary approach to charge steering: instead of using
charge-derived directions, we learn distance-predictive directions directly
from the representations.

Usage:
    python contact_steering.py \
        --probing_dataset data/probing_dataset.csv \
        --patch_dataset data/patch_cases.csv \
        --output results/ \
        --window_sizes 3 5 10 \
        --magnitudes 1 2 3 5
"""

import argparse
import gc
import os
import random
import sys
import types
import tempfile
import warnings

# Add project root and src/ to path so imports work without PYTHONPATH
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_PROJECT_ROOT, 'src')
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import Ridge
from transformers import EsmForProteinFolding, AutoTokenizer
from transformers.models.esm.modeling_esmfold import (
    categorical_lddt,
    EsmFoldingTrunk,
    EsmForProteinFoldingOutput,
)
from transformers.models.esm.openfold_utils import make_atom14_masks
from transformers.utils import ContextManagers

from src.utils.representation_utils import CollectedRepresentations, TrunkHooks
from src.utils.model_utils import load_esmfold
# Reuse the DistanceProbe class, hook-based z collection, CA-distance calc, and
# probe train/eval helpers from the standalone distance-probing script so both
# scripts share one canonical probe implementation.
from main_paper.z_probing_distance import (
    DistanceProbe, run_and_collect_z, compute_ca_distances,
    AA_TO_IDX, evaluate_probes_online, plot_probe_results,
)


# ============================================================================
# CONSTANTS
# ============================================================================

NUM_BLOCKS = 48                # Number of blocks in the ESMFold folding trunk
CONTACT_THRESHOLD = 8.0        # Cβ-Cβ distance (Å) below which residues count as "in contact"
CLOSE_CONTACT_THRESHOLD = 6.0  # Stricter Cβ-Cβ distance (Å) for a "close" contact
TARGET_DISTANCE = 5.5  # Target Cβ-Cβ distance for β-sheet H-bonding
HBOND_DISTANCE = 3.5   # N-O distance threshold for H-bond


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class GradientDirections:
    """Cached gradient directions for all blocks."""
    directions: Dict[int, np.ndarray]  # block_idx -> unit-norm direction that decreases predicted distance
    stds: Dict[int, float]             # block_idx -> std of z projected onto the direction (for magnitude scaling)
    probes: Dict[int, DistanceProbe]   # block_idx -> trained Ridge probe (weights/bias) for that block


@dataclass
class HairpinTopology:
    """Defines the topology of a hypothetical hairpin."""
    strand1_start: int
    strand1_end: int
    turn_start: int
    turn_end: int
    strand2_start: int
    strand2_end: int
    # (i, j) residue index pairs across the two strands expected to H-bond in an ideal antiparallel hairpin
    cross_strand_pairs: List[Tuple[int, int]]


# ============================================================================
# PART 1: COLLECTION USING HOOKS
# ============================================================================

def get_baseline_structure(
    model,
    tokenizer,
    device: str,
    sequence: str,
) -> Tuple[EsmForProteinFoldingOutput, CollectedRepresentations]:
    """
    Run baseline forward pass and return full outputs with z_list.
    """
    # run_and_collect_z registers forward hooks on the trunk blocks (collect_z=True)
    # around a normal model call, so `collector.z_blocks` ends up with one [1, L, L, 128]
    # pairwise tensor per block in addition to the standard model outputs.
    outputs, collector = run_and_collect_z(model, tokenizer, device, sequence)
    
    L = len(sequence)
    B = 1
    
    structure = {
        "aatype": outputs.aatype,
        "positions": outputs.positions,
        "states": outputs.states,
        "s_s": outputs.s_s,
        "s_z": outputs.s_z,
    }
    
    # Re-derive atom14/atom37 masks and pLDDT from raw tensors (same computation the
    # model's own forward already performs) so we can repackage a standalone
    # EsmForProteinFoldingOutput with the extra z_list field attached below.
    make_atom14_masks(structure)
    structure["residue_index"] = torch.arange(L, device=device).unsqueeze(0)
    
    lddt_head = model.lddt_head(structure["states"]).reshape(
        structure["states"].shape[0], B, L, -1, model.lddt_bins
    )
    structure["lddt_head"] = lddt_head
    plddt = categorical_lddt(lddt_head[-1], bins=model.lddt_bins)
    structure["plddt"] = plddt
    
    full_output = EsmForProteinFoldingOutput(
        positions=structure["positions"],
        states=structure["states"],
        s_s=structure["s_s"],
        s_z=structure["s_z"],
        aatype=structure["aatype"],
        atom14_atom_exists=structure["atom14_atom_exists"],
        residx_atom14_to_atom37=structure["residx_atom14_to_atom37"],
        residx_atom37_to_atom14=structure["residx_atom37_to_atom14"],
        atom37_atom_exists=structure["atom37_atom_exists"],
        residue_index=structure["residue_index"],
        lddt_head=structure["lddt_head"],
        plddt=structure["plddt"],
    )
    
    # Attach per-block pairwise reps (dict[int, [1, L, L, 128]]) for later use as
    # the "current z" baseline when computing targeted steering vectors.
    full_output.z_list = collector.z_blocks
    
    return full_output, collector


# ============================================================================
# PART 2: INTERVENTION CONTEXT MANAGER
# ============================================================================

def make_z_intervention_forward(
    intervention_blocks: set,
    z_interventions: Dict[int, torch.Tensor],
):
    """
    Create trunk forward that applies z interventions at specified blocks.
    """
    # This is a near-verbatim copy of EsmFoldingTrunk.forward (transformers'
    # modeling_esmfold.py) with a single addition: after each block, if its index
    # is both requested (intervention_blocks) and has a prepared delta
    # (z_interventions), that delta is added to the pairwise repr z. Copying the
    # whole method (rather than patching just the block loop) is necessary because
    # the recycling/structure-module logic all lives in this one function body.
    def forward(self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles):
        """Drop-in replacement for EsmFoldingTrunk.forward that injects z interventions after chosen blocks."""
        device = seq_feats.device
        s_s_0, s_z_0 = seq_feats, pair_feats

        if no_recycles is None:
            no_recycles = self.config.max_recycles
        else:
            no_recycles += 1

        def trunk_iter(s, z, residx, mask):
            """Run one pass through all trunk blocks, adding z interventions after the targeted blocks."""
            z = z + self.pairwise_positional_embedding(residx, mask=mask)
            
            for block_idx, block in enumerate(self.blocks):
                s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size)
                
                # Apply intervention after block
                # z: [B, L, L, 128]; z_interventions[block_idx]: [L, L, 128] built without a
                # batch dim, so unsqueeze(0) broadcasts it onto the (batch=1) pairwise repr.
                if block_idx in intervention_blocks and block_idx in z_interventions:
                    z = z + z_interventions[block_idx].unsqueeze(0)
            
            return s, z

        s_s, s_z = s_s_0, s_z_0
        recycle_s = torch.zeros_like(s_s)
        recycle_z = torch.zeros_like(s_z)
        recycle_bins = torch.zeros(*s_z.shape[:-1], device=device, dtype=torch.int64)

        # Recycling: run the trunk+structure module up to `no_recycles` times, each time
        # feeding back the previous iteration's s/z (normalized) and a distogram computed
        # from the predicted Cβ coordinates. Every recycle re-applies the interventions
        # inside trunk_iter, so the steering is injected fresh on every recycling pass.
        # Only the final pass is done with gradients enabled (ContextManagers([]) vs no_grad()).
        for recycle_idx in range(no_recycles):
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
                # Bin the predicted Cβ-Cβ distances (inferred from N/CA/C backbone atoms) into
                # discrete distogram bins to condition the next recycling iteration.
                recycle_bins = EsmFoldingTrunk.distogram(
                    structure["positions"][-1][:, :, :3],
                    3.375, 21.375, self.recycle_bins,
                )

        structure["s_s"] = s_s
        structure["s_z"] = s_z
        structure["aatype"] = true_aa
        return structure
    
    return forward


@contextmanager
def z_intervention_context(
    model,
    intervention_blocks: set,
    z_interventions: Dict[int, torch.Tensor],
):
    """
    Context manager for applying z interventions during forward pass.
    """
    # Monkey-patch model.trunk.forward for the duration of the `with` block only:
    # save the bound original method, swap in the intervention version (bound via
    # types.MethodType so `self` resolves correctly), then always restore the
    # original in `finally` so the patch never leaks into subsequent calls/errors.
    original = model.trunk.forward
    
    intervention_forward = make_z_intervention_forward(intervention_blocks, z_interventions)
    model.trunk.forward = types.MethodType(intervention_forward, model.trunk)
    
    try:
        yield
    finally:
        model.trunk.forward = original


def run_with_intervention(
    model,
    tokenizer,
    device: str,
    sequence: str,
    intervention_blocks: set,
    z_interventions: Dict[int, torch.Tensor],
) -> EsmForProteinFoldingOutput:
    """
    Run forward pass with z interventions using context manager.
    """
    # num_recycles=0 -> exactly one trunk pass (no additional recycling iterations),
    # keeping the intervention's effect on structure prediction easy to interpret.
    with z_intervention_context(model, intervention_blocks, z_interventions):
        with torch.no_grad():
            inputs = tokenizer(sequence, return_tensors='pt', add_special_tokens=False).to(device)
            outputs = model(**inputs, num_recycles=0)
    
    L = len(sequence)
    B = 1
    
    structure = {
        "aatype": outputs.aatype,
        "positions": outputs.positions,
        "states": outputs.states,
        "s_s": outputs.s_s,
        "s_z": outputs.s_z,
    }
    
    # Same pattern as get_baseline_structure: recompute atom14/atom37 masks and pLDDT
    # from raw tensors to build a fresh, standalone output object.
    make_atom14_masks(structure)
    structure["residue_index"] = torch.arange(L, device=device).unsqueeze(0)
    
    lddt_head = model.lddt_head(structure["states"]).reshape(
        structure["states"].shape[0], B, L, -1, model.lddt_bins
    )
    structure["lddt_head"] = lddt_head
    plddt = categorical_lddt(lddt_head[-1], bins=model.lddt_bins)
    structure["plddt"] = plddt
    
    return EsmForProteinFoldingOutput(
        positions=structure["positions"],
        states=structure["states"],
        s_s=structure["s_s"],
        s_z=structure["s_z"],
        aatype=structure["aatype"],
        atom14_atom_exists=structure["atom14_atom_exists"],
        residx_atom14_to_atom37=structure["residx_atom14_to_atom37"],
        residx_atom37_to_atom14=structure["residx_atom37_to_atom14"],
        atom37_atom_exists=structure["atom37_atom_exists"],
        residue_index=structure["residue_index"],
        lddt_head=structure["lddt_head"],
        plddt=structure["plddt"],
    )


# ============================================================================
# PART 3: GEOMETRY UTILITIES
# ============================================================================

def compute_cb_distances(positions: torch.Tensor) -> torch.Tensor:
    """
    Compute Cβ-Cβ distance matrix.
    Falls back to CA for glycine or missing Cβ.
    """
    # atom14 ordering is N=0, CA=1, C=2, O=3, CB=4, so CB is index 4. (Was 3, which is the
    # backbone carbonyl O -- a bug: it computed O-O distances, not CB-CB. Note the sibling
    # compute_backbone_no_distances() below correctly uses O at index 3. With CB_IDX=4 the
    # `< 0.1` fallback below now correctly covers Gly, whose CB slot is zero-filled.)
    CB_IDX, CA_IDX = 4, 1
    L = positions.shape[0]
    
    cb_coords = []
    for i in range(L):
        cb = positions[i, CB_IDX]
        # Check if Cβ exists (non-zero coordinates)
        if torch.norm(cb) < 0.1:
            cb = positions[i, CA_IDX]
        cb_coords.append(cb)
    
    cb_coords = torch.stack(cb_coords)  # [L, 3]
    diff = cb_coords.unsqueeze(0) - cb_coords.unsqueeze(1)  # [L, L, 3] pairwise coordinate diffs
    return torch.sqrt((diff ** 2).sum(-1) + 1e-8)  # 1e-8 avoids NaN gradients from sqrt(0) at i==j


def compute_backbone_no_distances(positions: torch.Tensor) -> torch.Tensor:
    """
    Compute N-O distances for potential H-bond detection.
    Returns matrix where [i,j] is distance from N of residue i to O of residue j.
    
    Atom indices in atom14: N=0, CA=1, C=2, O=3 (approximately)
    """
    N_IDX, O_IDX = 0, 3
    
    n_coords = positions[:, N_IDX, :]  # [L, 3]
    o_coords = positions[:, O_IDX, :]  # [L, 3]
    
    # N[i] to O[j] distances
    diff = n_coords.unsqueeze(1) - o_coords.unsqueeze(0)  # [L, L, 3]
    return torch.sqrt((diff ** 2).sum(-1) + 1e-8)  # [L, L]; +1e-8 avoids sqrt(0) NaN gradients


def define_hairpin_topology(
    region_start: int,
    region_end: int,
    turn_length: int = 4,
) -> HairpinTopology:
    """Define a hypothetical hairpin topology for a region."""
    # Step 1: split the region evenly into strand1 + turn + strand2, biasing any
    # leftover residues (from integer division) into strand2.
    region_length = region_end - region_start
    strand_length = (region_length - turn_length) // 2
    
    # Step 2: guard against too-short strands (e.g. small regions) by enforcing a
    # minimum of 3 residues per strand, shrinking the turn to compensate.
    if strand_length < 3:
        strand_length = 3
        turn_length = max(2, region_length - 2 * strand_length)
    
    strand1_start = region_start
    strand1_end = region_start + strand_length
    turn_start = strand1_end
    turn_end = turn_start + turn_length
    strand2_start = turn_end
    strand2_end = min(strand2_start + strand_length, region_end)
    
    # Step 3: pair residues antiparallel-style, i.e. strand1 residue i (counting from
    # its N-terminal start) pairs with strand2 residue i counting from strand2's
    # C-terminal end. This mirrors real β-hairpins, where the turn reverses strand
    # direction so the two strands run antiparallel to each other.
    cross_strand_pairs = []
    actual_strand2_len = strand2_end - strand2_start
    actual_strand1_len = strand1_end - strand1_start
    
    for i in range(min(actual_strand1_len, actual_strand2_len)):
        res_i = strand1_start + i
        res_j = strand2_end - 1 - i
        if res_j >= strand2_start:
            cross_strand_pairs.append((res_i, res_j))
    
    return HairpinTopology(
        strand1_start=strand1_start,
        strand1_end=strand1_end,
        turn_start=turn_start,
        turn_end=turn_end,
        strand2_start=strand2_start,
        strand2_end=strand2_end,
        cross_strand_pairs=cross_strand_pairs,
    )


# ============================================================================
# PART 4: SLIDING WINDOW UTILITIES
# ============================================================================

def get_sliding_window_block_sets(
    window_size: int,
    total_blocks: int = NUM_BLOCKS,
) -> List[Tuple[str, set]]:
    """
    Generate sliding window block sets.
    
    Each window covers `window_size` consecutive blocks, sliding from
    block 0 to block (total_blocks - window_size).
    """
    block_sets = []
    
    for start_block in range(total_blocks - window_size + 1):
        end_block = start_block + window_size - 1
        # Name encodes window_size/start/end positionally so it can be parsed back
        # out later (see run_gradient_steering_experiment's block_set_name.split('_')).
        name = f'w{window_size}_blocks_{start_block}_to_{end_block}'
        blocks = set(range(start_block, start_block + window_size))
        block_sets.append((name, blocks))
    
    return block_sets


# ============================================================================
# PART 5: ONLINE RIDGE REGRESSION (SUFFICIENT STATISTICS)
# ============================================================================

@dataclass
class OnlineRidgeAccumulator:
    """
    Accumulates sufficient statistics for Ridge regression.
    
    Only stores X'X [dim, dim] and X'y [dim], not the raw data.
    This is mathematically equivalent to batch Ridge regression.
    """
    # Why: with hundreds of proteins x hundreds of sampled pairs each, holding every
    # [L, L, dim] z sample in memory to fit a single batch Ridge regression would be
    # prohibitive. X'X and X'y accumulate additively across batches/proteins, so this
    # gives the exact closed-form Ridge solution (see solve()) using only O(dim^2) memory.
    dim: int
    block_idx: int
    
    # Sufficient statistics
    XtX: np.ndarray = None  # [dim, dim]
    Xty: np.ndarray = None  # [dim]
    yty: float = 0.0        # scalar (for R² computation)
    y_sum: float = 0.0      # sum of y (for mean)
    n_samples: int = 0
    
    def __post_init__(self):
        """Allocate zero-initialized accumulator arrays now that `dim` is known (runs right after dataclass __init__)."""
        self.XtX = np.zeros((self.dim, self.dim), dtype=np.float64)
        self.Xty = np.zeros(self.dim, dtype=np.float64)
    
    def update(self, X: np.ndarray, y: np.ndarray):
        """
        Update sufficient statistics with a batch of samples.
        
        Args:
            X: Feature matrix [batch_size, dim]
            y: Target values [batch_size]
        """
        # Cast to float64: accumulating in float32 across many proteins/batches would
        # lose precision in the running sums (XtX, Xty, yty).
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        self.XtX += X.T @ X
        self.Xty += X.T @ y
        self.yty += np.dot(y, y)
        self.y_sum += np.sum(y)
        self.n_samples += len(y)
    
    def solve(self, alpha: float = 1.0) -> DistanceProbe:
        """
        Solve Ridge regression: w = (X'X + αI)^(-1) X'y
        
        Args:
            alpha: Regularization strength
        
        Returns:
            Trained DistanceProbe
        """
        if self.n_samples == 0:
            raise ValueError(f"No samples for block {self.block_idx}")
        
        # Solve (X'X + αI) w = X'y
        regularized = self.XtX + alpha * np.eye(self.dim)  # add L2 penalty (Ridge) to the diagonal
        weights = np.linalg.solve(regularized, self.Xty)  # solves the linear system directly (avoids explicit matrix inverse)
        
        # Bias: for Ridge without centering, bias = 0
        bias = 0.0
        
        # Compute training R² from sufficient statistics
        # R² = 1 - SS_res / SS_tot
        # SS_res = y'y - 2*w'X'y + w'X'X*w
        # SS_tot = y'y - n * y_mean²
        y_mean = self.y_sum / self.n_samples
        ss_tot = self.yty - self.n_samples * y_mean ** 2
        ss_res = self.yty - 2 * np.dot(weights, self.Xty) + weights @ self.XtX @ weights
        r2_train = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        return DistanceProbe(
            weights=weights,
            bias=bias,
            block_idx=self.block_idx,
            r2_train=r2_train,
        )


# ============================================================================
# PART 5b: PROBE TRAINING (MEMORY-EFFICIENT)
# ============================================================================

def train_distance_probes_chunked(
    sequences: List[str],
    labels: List[str],
    model,
    tokenizer,
    device: str,
    n_pairs_per_protein: int = 500,
    alpha: float = 1.0,
) -> GradientDirections:
    """
    Train linear distance probes using online sufficient statistics accumulation.
    
    Memory efficient: only stores [dim, dim] matrix per block, not raw samples.
    Mathematically identical to batch Ridge regression.
    
    Samples all (i,j) pairs where i != j (matching z_probing_distance.py setup).
    Uses bias=0 (no centering).
    """
    print(f"\nTraining distance probes (online) on {len(sequences)} sequences...")
    print(f"  n_pairs_per_protein: {n_pairs_per_protein}")
    
    dim = None
    accumulators = None
    
    processed = 0
    total_samples = 0
    
    for seq in tqdm(sequences, desc="Training probes (online)"):
        # Skip invalid sequences
        if not all(aa in AA_TO_IDX for aa in seq):
            continue
        if len(seq) < 10:
            continue
        
        L = len(seq)
        
        try:
            outputs, collector = run_and_collect_z(model, tokenizer, device, seq)
        except Exception as e:
            print(f"Error processing sequence: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            continue
        
        # Initialize accumulators on first protein (need z's feature dim, e.g. 128,
        # which isn't known until we've run the model once).
        if dim is None:
            first_z = next(iter(collector.z_blocks.values()))  # [1, L, L, dim]
            dim = first_z.shape[-1]
            print(f"Z dimension: {dim}")
            accumulators = {b: OnlineRidgeAccumulator(dim=dim, block_idx=b) for b in range(NUM_BLOCKS)}
        
        # Compute CA distances
        final_positions = outputs.positions[-1, 0]  # [L, 14, 3]: last IPA refinement iter, batch 0
        distances = compute_ca_distances(final_positions)
        
        # Sample pairs (all pairs where i != j)
        all_pairs = [(i, j) for i in range(L) for j in range(L) if i != j]
        
        # Subsample pairs per protein (rather than using all L^2 pairs) to bound the
        # per-protein cost and give every protein roughly equal weight in training.
        if len(all_pairs) > n_pairs_per_protein:
            sampled_indices = np.random.choice(len(all_pairs), n_pairs_per_protein, replace=False)
            sampled_pairs = [all_pairs[k] for k in sampled_indices]
        else:
            sampled_pairs = all_pairs
        
        # Update accumulators for each block
        for block_idx, z in collector.z_blocks.items():
            z_np = z[0].cpu().numpy()  # [L, L, dim]
            
            X_batch = np.array([z_np[i, j] for i, j in sampled_pairs], dtype=np.float64)
            y_batch = np.array([distances[i, j].item() for i, j in sampled_pairs], dtype=np.float64)
            
            accumulators[block_idx].update(X_batch, y_batch)
        
        processed += 1
        total_samples += len(sampled_pairs)
        
        # Clean up
        del outputs, collector
        torch.cuda.empty_cache()
        
        if processed % 50 == 0:
            gc.collect()
            tqdm.write(f"  Processed {processed} sequences, {total_samples} total samples")
    
    print(f"\nProcessed {processed} proteins, {total_samples} total samples")
    
    # Solve all probes
    print("\nSolving Ridge regression for each block...")
    probes = {}
    directions = {}
    stds = {}
    
    for block in tqdm(range(NUM_BLOCKS), desc="Solving probes"):
        # Skip blocks with too few samples (< 100) to fit a trustworthy probe;
        # such blocks are simply absent from the returned directions/probes/stds.
        if accumulators[block].n_samples < 100:
            continue
        
        probe = accumulators[block].solve(alpha=alpha)
        probes[block] = probe
        directions[block] = probe.get_gradient_direction()
        
        # Compute std from overall variance of z projected onto weight direction
        # diag(X'X)/n approximates E[x_k^2] per feature k (uncentered 2nd moment, since
        # bias=0 means no mean-subtraction); averaging across dims and sqrt gives a
        # rough per-block scale of z's magnitude, used to make steering `magnitude`
        # (in std units) comparable across blocks with different activation scales.
        n = accumulators[block].n_samples
        overall_var = accumulators[block].XtX.diagonal() / n
        stds[block] = float(np.sqrt(np.mean(np.maximum(overall_var, 0)) + 1e-8))
        
        if block % 8 == 0:
            tqdm.write(f"  Block {block}: R² = {probe.r2_train:.4f}, "
                       f"n = {accumulators[block].n_samples}, std={stds[block]:.3f}")
    
    # Clean up
    del accumulators
    gc.collect()
    
    return GradientDirections(directions=directions, stds=stds, probes=probes)

def save_directions(directions: GradientDirections, path: str):
    """Save directions to disk."""
    # Flatten each DistanceProbe dataclass into a plain dict of its fields so the
    # checkpoint is just plain Python/numpy objects (no custom class needed to unpickle).
    probes_data = {
        block: {'weights': p.weights, 'bias': p.bias, 'block_idx': p.block_idx, 'r2_train': p.r2_train}
        for block, p in directions.probes.items()
    }
    torch.save({
        'directions': directions.directions,
        'stds': directions.stds,
        'probes': probes_data,
    }, path)
    print(f"Saved directions to {path}")


def load_directions(path: str) -> GradientDirections:
    """Load directions from disk."""
    # weights_only=False: needed because the checkpoint holds plain dicts/floats, not
    # just tensors, which torch's newer weights_only=True safe-loader would reject.
    data = torch.load(path, weights_only=False)
    # Reconstruct DistanceProbe objects from the plain per-field dicts saved above.
    probes = {
        block: DistanceProbe(**pdata)
        for block, pdata in data['probes'].items()
    }
    return GradientDirections(
        directions=data['directions'],
        stds=data['stds'],
        probes=probes,
    )


# ============================================================================
# PART 6: INTERVENTION CONSTRUCTION
# ============================================================================

def create_targeted_steering_intervention(
    topology: HairpinTopology,
    seq_len: int,
    directions: GradientDirections,
    blocks: List[int],
    baseline_z_list: Dict[int, torch.Tensor],
    target_distance: float,
    magnitude: float,
    device: str,
) -> Dict[int, torch.Tensor]:
    """
    Create z interventions that steer cross-strand pairs toward target distance.
    
    Uses std-normalized scaling: magnitude is in units of standard deviations
    of projections onto the steering direction. This makes magnitude comparable
    across blocks and interpretable (magnitude=1 means move 1 std along direction).
    
    Args:
        topology: Hairpin topology with cross-strand pairs
        seq_len: Sequence length
        directions: Trained probes and directions (includes stds)
        blocks: Which blocks to intervene at
        baseline_z_list: Baseline z representations from forward pass
        target_distance: Target distance in Angstroms (e.g., 5.5)
        magnitude: Scaling factor in units of std deviations
        device: Torch device
    
    Returns:
        Dict mapping block_idx to intervention tensor
    """
    z_interventions = {}
    
    for block in blocks:
        if block not in directions.probes:
            continue
        
        probe = directions.probes[block]
        std = directions.stds.get(block, 1.0)
        baseline_z = baseline_z_list[block][0].cpu().numpy()  # [L, L, dim]
        
        # Per-block delta tensor, zero everywhere except at cross-strand pairs;
        # this gets added onto z at `block` inside the intervention forward.
        z_int = torch.zeros(seq_len, seq_len, len(probe.weights), 
                           dtype=torch.float32, device=device)
        
        for i, j in topology.cross_strand_pairs:
            # Get current z for this pair
            current_z = baseline_z[i, j]
            
            # Exact delta (in z-space) the linear probe says would move its predicted
            # distance for this pair from the current baseline value to target_distance.
            steering = probe.get_steering_vector(current_z, target_distance)
            
            # Normalize steering direction
            steering_norm = np.linalg.norm(steering)
            if steering_norm > 1e-8:
                steering_direction = steering / steering_norm
            else:
                steering_direction = steering
            
            # Discard the exact delta's magnitude (which depends on how far the
            # baseline already is from target) and instead rescale to a fixed,
            # std-normalized magnitude. This decouples intervention strength from
            # each pair's baseline distance, so `magnitude` is directly comparable
            # across residue pairs, blocks, and proteins.
            scaled_steering = magnitude * std * steering_direction
            
            # Apply symmetrically: z[i,j] and z[j,i] both represent the same residue
            # pair, so both entries get nudged identically to keep the intervention consistent.
            z_int[i, j] = torch.tensor(scaled_steering, dtype=torch.float32, device=device)
            z_int[j, i] = torch.tensor(scaled_steering, dtype=torch.float32, device=device)
        
        z_interventions[block] = z_int
    
    return z_interventions


def create_gradient_intervention(
    topology: HairpinTopology,
    seq_len: int,
    directions: GradientDirections,
    blocks: List[int],
    magnitude: float,
    device: str,
) -> Dict[int, torch.Tensor]:
    """
    Create z interventions using gradient directions (original method).
    Moves in the direction that decreases predicted distance.
    """
    # Note: unlike create_targeted_steering_intervention, this applies the same fixed
    # direction/magnitude to every cross-strand pair regardless of each pair's current
    # baseline z (simpler, but not aimed at a specific target_distance). Not currently
    # called from main(); create_targeted_steering_intervention is used instead.
    z_interventions = {}
    
    for block in blocks:
        if block not in directions.directions:
            continue
        
        direction = directions.directions[block]
        std = directions.stds[block]
        
        z_int = torch.zeros(seq_len, seq_len, len(direction), dtype=torch.float32, device=device)
        delta = torch.tensor(magnitude * std * direction, dtype=torch.float32, device=device)
        
        for i, j in topology.cross_strand_pairs:
            z_int[i, j] = delta
            z_int[j, i] = delta
        
        z_interventions[block] = z_int
    
    return z_interventions


# ============================================================================
# PART 7: EVALUATION
# ============================================================================

def compute_comprehensive_metrics(
    outputs: EsmForProteinFoldingOutput,
    topology: HairpinTopology,
    target_distance: float = TARGET_DISTANCE,
) -> Dict[str, Any]:
    """
    Compute comprehensive structural metrics for hairpin evaluation.
    
    Returns metrics for:
    - Cross-strand distances (Cβ-Cβ)
    - Contact counts at various thresholds
    - Distance to target
    - Potential H-bonds (N-O distances)
    - Structure quality (pLDDT)
    """
    positions = outputs.positions[-1, 0].cpu()  # [L, 14, 3]
    plddt = outputs.plddt[0].cpu()  # [L]
    
    # Compute distance matrices
    cb_distances = compute_cb_distances(positions)
    ca_distances = compute_ca_distances(positions)
    no_distances = compute_backbone_no_distances(positions)
    
    # Cross-strand Cβ-Cβ distances
    cb_dists = [cb_distances[i, j].item() for i, j in topology.cross_strand_pairs]
    ca_dists = [ca_distances[i, j].item() for i, j in topology.cross_strand_pairs]
    
    # N-O distances for H-bond detection (check both directions)
    no_dists = []
    for i, j in topology.cross_strand_pairs:
        # N[i]-O[j] and N[j]-O[i] for antiparallel strand
        no_dists.append(min(no_distances[i, j].item(), no_distances[j, i].item()))
    
    # Contact counts
    n_contacts_8A = sum(1 for d in cb_dists if d < CONTACT_THRESHOLD)
    n_contacts_6A = sum(1 for d in cb_dists if d < CLOSE_CONTACT_THRESHOLD)
    n_contacts_target = sum(1 for d in cb_dists if abs(d - target_distance) < 1.0)  # within +/-1A of target counts as "at target"
    
    # Potential H-bonds
    n_potential_hbonds = sum(1 for d in no_dists if d < HBOND_DISTANCE)
    
    # Distance to target
    mean_cb_dist = np.mean(cb_dists) if cb_dists else None
    # NOTE: possible bug -- `if mean_cb_dist` is a truthiness check, so a (practically
    # impossible but not-disallowed) mean distance of exactly 0.0 would be treated the
    # same as None/empty and silently produce dist_to_target=None instead of the true
    # value. Should be `if mean_cb_dist is not None`.
    dist_to_target = abs(mean_cb_dist - target_distance) if mean_cb_dist else None
    
    # pLDDT in region
    region_residues = list(range(topology.strand1_start, topology.strand1_end)) + \
                      list(range(topology.strand2_start, topology.strand2_end))
    region_plddt = plddt[region_residues].mean().item() if region_residues else None
    overall_plddt = plddt.mean().item()
    
    return {
        # Cross-strand Cβ distances
        'mean_cb_dist': mean_cb_dist,
        'min_cb_dist': np.min(cb_dists) if cb_dists else None,
        'max_cb_dist': np.max(cb_dists) if cb_dists else None,
        'std_cb_dist': np.std(cb_dists) if len(cb_dists) > 1 else None,
        
        # Cross-strand CA distances (for comparison)
        'mean_ca_dist': np.mean(ca_dists) if ca_dists else None,
        
        # Contact counts
        'n_contacts_8A': n_contacts_8A,
        'n_contacts_6A': n_contacts_6A,
        'n_at_target': n_contacts_target,
        'n_pairs': len(cb_dists),
        'contact_fraction_8A': n_contacts_8A / len(cb_dists) if cb_dists else None,
        'contact_fraction_6A': n_contacts_6A / len(cb_dists) if cb_dists else None,
        
        # Distance to target
        'dist_to_target': dist_to_target,
        
        # H-bond proxies
        'mean_no_dist': np.mean(no_dists) if no_dists else None,
        'min_no_dist': np.min(no_dists) if no_dists else None,
        'n_potential_hbonds': n_potential_hbonds,
        'hbond_fraction': n_potential_hbonds / len(no_dists) if no_dists else None,
        
        # Structure quality
        'region_plddt': region_plddt,
        'overall_plddt': overall_plddt,
    }


def compute_cross_strand_distances(
    outputs: EsmForProteinFoldingOutput,
    topology: HairpinTopology,
) -> Dict[str, float]:
    """Compute distances for cross-strand pairs (simplified version)."""
    # Thin alias kept for naming parity with the same-named function in
    # charge_steering.py; not otherwise used within this file.
    return compute_comprehensive_metrics(outputs, topology)


# ============================================================================
# PART 7b: PDB SAVING
# ============================================================================

def save_structure_as_pdb(outputs: EsmForProteinFoldingOutput, model, path: str):
    """Save structure as PDB file."""
    # .detach() is required because model.output_to_pdb() calls .numpy() on every
    # field internally, which fails on tensors still tracking gradients.
    clean = {
        "positions": outputs.positions.detach(),
        "aatype": outputs.aatype.detach(),
        "atom14_atom_exists": outputs.atom14_atom_exists.detach(),
        "residue_index": outputs.residue_index.detach(),
        "plddt": outputs.plddt.detach(),
        "atom37_atom_exists": outputs.atom37_atom_exists.detach(),
        "residx_atom14_to_atom37": outputs.residx_atom14_to_atom37.detach(),
        "residx_atom37_to_atom14": outputs.residx_atom37_to_atom14.detach(),
    }
    pdb_string = model.output_to_pdb(clean)[0]
    with open(path, 'w') as f:
        f.write(pdb_string)


# ============================================================================
# PART 8: MAIN EXPERIMENT
# ============================================================================

def run_gradient_steering_experiment(
    cases_df: pd.DataFrame,
    model,
    tokenizer,
    device: str,
    directions: GradientDirections,
    magnitudes: List[float],
    block_sets: List[Tuple[str, set]],
    output_dir: str,
    target_distance: float = TARGET_DISTANCE,
    save_every: int = 10,
    save_all_pdbs_cases: int = 2,
) -> pd.DataFrame:
    """
    Run probe-based contact steering experiments with sliding windows.
    
    Uses targeted steering to move cross-strand z representations toward
    predicting target_distance (default 5.5Å for β-sheet contacts).
    
    Args:
        cases_df: DataFrame with cases to process
        model: ESMFold model
        tokenizer: Tokenizer
        device: Torch device
        directions: Trained gradient directions
        magnitudes: List of magnitude values (in std units)
        block_sets: List of (name, block_set) tuples
        output_dir: Output directory
        target_distance: Target distance for metrics
        save_every: Checkpoint frequency
        save_all_pdbs_cases: Save all PDBs for first N cases (0 to disable)
    """
    results = []
    results_path = os.path.join(output_dir, 'gradient_steering_results.parquet')
    
    available_blocks = set(directions.directions.keys())
    
    # Create PDB directories
    induced_pdb_dir = os.path.join(output_dir, 'induced_hairpins')
    os.makedirs(induced_pdb_dir, exist_ok=True)
    
    if save_all_pdbs_cases > 0:
        all_pdbs_dir = os.path.join(output_dir, 'all_pdbs')
        os.makedirs(all_pdbs_dir, exist_ok=True)
    
    for case_idx, row in enumerate(cases_df.itertuples()):
        target_seq = row.target_sequence
        
        # Handle column names
        target_start = int(getattr(row, 'target_start', getattr(row, 'target_patch_start', 0)))
        target_end = int(getattr(row, 'target_end', getattr(row, 'target_patch_end', len(target_seq))))
        donor_hairpin_seq = getattr(row, 'donor_hairpin_sequence', '')
        target_name = getattr(row, 'target_name', 'unknown')
        
        L = len(target_seq)
        case_results = []
        
        # Should we save all PDBs for this case?
        save_all_for_case = case_idx < save_all_pdbs_cases
        if save_all_for_case:
            case_pdb_dir = os.path.join(all_pdbs_dir, f'case{case_idx}_{target_name}')
            os.makedirs(case_pdb_dir, exist_ok=True)
        
        topology = define_hairpin_topology(target_start, target_end, turn_length=4)
        
        tqdm.write(f"\nCase {case_idx}/{len(cases_df)}: {target_name}")
        tqdm.write(f"  Region: {target_start}-{target_end}")
        tqdm.write(f"  Cross-strand pairs: {len(topology.cross_strand_pairs)}")
        if save_all_for_case:
            tqdm.write(f"  [Saving ALL PDBs for this case]")
        
        # Baseline - need to collect z for targeted steering
        try:
            baseline_outputs, baseline_collector = get_baseline_structure(model, tokenizer, device, target_seq)
        except Exception as e:
            tqdm.write(f"  Baseline error: {e}")
            continue
        
        baseline_metrics = compute_comprehensive_metrics(baseline_outputs, topology, target_distance)

        tqdm.write(f"  Baseline: "
                   f"mean_cb={baseline_metrics['mean_cb_dist']:.1f}Å, "
                   f"pLDDT={baseline_metrics['region_plddt']:.1f}, "
                   f"contacts_6A={baseline_metrics['n_contacts_6A']}/{baseline_metrics['n_pairs']}")
        
        # Save baseline PDB if saving all for this case
        baseline_pdb_path = None
        if save_all_for_case:
            baseline_pdb_path = os.path.join(case_pdb_dir, 'baseline.pdb')
            save_structure_as_pdb(baseline_outputs, model, baseline_pdb_path)
        
        case_results.append({
            'case_idx': case_idx,
            'target_name': target_name,
            'target_sequence': target_seq,
            'region_start': target_start,
            'region_end': target_end,
            'donor_hairpin_sequence': donor_hairpin_seq,
            'block_set': 'baseline',
            'window_size': 0,
            'window_start': -1,
            'window_end': -1,
            'n_blocks': 0,
            'magnitude': 0.0,
            'target_distance': target_distance,
            'pdb_path': baseline_pdb_path,
            # Comprehensive metrics
            **baseline_metrics,
        })
        
        # Store baseline z for targeted steering
        baseline_z_list = baseline_collector.z_blocks
        
        del baseline_outputs
        torch.cuda.empty_cache()
        
        # Precompute valid block sets: restrict each sliding window to only the
        # blocks that actually have a trained probe/direction (some blocks may have
        # been skipped during training due to too few samples).
        valid_block_sets = []
        for block_set_name, intervention_blocks in block_sets:
            ib = intervention_blocks & available_blocks
            if len(ib) > 0:
                valid_block_sets.append((block_set_name, ib))

        n_total = len(valid_block_sets) * len(magnitudes)

        pbar = tqdm(total=n_total, desc=f"Case {case_idx}", leave=False)
        
        for block_set_name, intervention_blocks in valid_block_sets:
            # Extract window_size and window start/end from name, e.g.
            # 'w5_blocks_10_to_14'.split('_') -> ['w5','blocks','10','to','14']
            parts = block_set_name.split('_')
            window_size = int(parts[0][1:])  # Remove 'w' prefix
            window_start = int(parts[2])
            window_end = int(parts[4])
            
            for magnitude in magnitudes:
                pbar.set_postfix_str(f"w={window_size} start={window_start} mag={magnitude}σ")

                # Use targeted steering with std-based scaling
                z_interventions = create_targeted_steering_intervention(
                    topology=topology,
                    seq_len=L,
                    directions=directions,
                    blocks=list(intervention_blocks),
                    baseline_z_list=baseline_z_list,
                    target_distance=target_distance,
                    magnitude=magnitude,
                    device=device,
                )

                try:
                    int_outputs = run_with_intervention(
                        model, tokenizer, device, target_seq,
                        intervention_blocks, z_interventions,
                    )
                except Exception as e:
                    tqdm.write(f"  Intervention error ({block_set_name}, mag={magnitude}): {e}")
                    pbar.update(1)
                    continue

                int_metrics = compute_comprehensive_metrics(int_outputs, topology, target_distance)

                # Compute changes from baseline
                # NOTE: possible bug -- these truthiness checks (`if x and y`) would
                # skip the computation if either value happened to be exactly 0.0,
                # treating a legitimate zero the same as None/missing. Contrast with
                # contacts_6A_change/hbond_change below, which correctly use
                # `is not None` for their (integer, more plausibly zero) values.
                cb_dist_change = None
                if baseline_metrics['mean_cb_dist'] and int_metrics['mean_cb_dist']:
                    cb_dist_change = int_metrics['mean_cb_dist'] - baseline_metrics['mean_cb_dist']
                
                plddt_change = None
                if baseline_metrics['region_plddt'] and int_metrics['region_plddt']:
                    plddt_change = int_metrics['region_plddt'] - baseline_metrics['region_plddt']
                
                contacts_6A_change = None
                if baseline_metrics['n_contacts_6A'] is not None and int_metrics['n_contacts_6A'] is not None:
                    contacts_6A_change = int_metrics['n_contacts_6A'] - baseline_metrics['n_contacts_6A']
                
                hbond_change = None
                if baseline_metrics['n_potential_hbonds'] is not None and int_metrics['n_potential_hbonds'] is not None:
                    hbond_change = int_metrics['n_potential_hbonds'] - baseline_metrics['n_potential_hbonds']

                # Check if H-bonds were gained
                hbonds_induced = hbond_change is not None and hbond_change > 0

                # Determine PDB path
                pdb_path = None

                # Save PDB if H-bonds gained
                if hbonds_induced:
                    pdb_path = os.path.join(
                        induced_pdb_dir,
                        f'case{case_idx}_{target_name}_{block_set_name}_mag{magnitude:.1f}.pdb'
                    )
                    save_structure_as_pdb(int_outputs, model, pdb_path)
                    tqdm.write(f"  *** H-BONDS INDUCED! blocks={block_set_name}, mag={magnitude}σ, "
                               f"+{hbond_change} hbonds ***")

                # Save all PDBs for first N cases (even if not induced)
                elif save_all_for_case:
                    pdb_path = os.path.join(
                        case_pdb_dir,
                        f'{block_set_name}_mag{magnitude:.1f}.pdb'
                    )
                    save_structure_as_pdb(int_outputs, model, pdb_path)
                
                # Log significant improvements
                if contacts_6A_change and contacts_6A_change > 0:
                    tqdm.write(f"    +{contacts_6A_change} close contacts at mag={magnitude}σ")

                case_results.append({
                    'case_idx': case_idx,
                    'target_name': target_name,
                    'target_sequence': target_seq,
                    'region_start': target_start,
                    'region_end': target_end,
                    'donor_hairpin_sequence': donor_hairpin_seq,
                    'block_set': block_set_name,
                    'window_size': window_size,
                    'window_start': window_start,
                    'window_end': window_end,
                    'n_blocks': len(intervention_blocks),
                    'magnitude': magnitude,
                    'target_distance': target_distance,
                    'pdb_path': pdb_path,
                    # Comprehensive metrics
                    **int_metrics,
                    # Changes from baseline
                    'cb_dist_change': cb_dist_change,
                    'plddt_change': plddt_change,
                    'contacts_6A_change': contacts_6A_change,
                    'hbond_change': hbond_change,
                })

                del z_interventions, int_outputs
                torch.cuda.empty_cache()

                pbar.update(1)
        
        pbar.close()
        
        # Clean up baseline z
        del baseline_z_list, baseline_collector
        torch.cuda.empty_cache()
        gc.collect()
        
        results.extend(case_results)
        
        # Periodic save: write a checkpoint parquet every `save_every` cases so
        # partial results survive a crash/timeout during a long sweep.
        if (case_idx + 1) % save_every == 0:
            pd.DataFrame(results).to_parquet(results_path, index=False)
            tqdm.write(f"[Checkpoint] Saved {len(results)} results after {case_idx + 1} cases")
    
    # Final save
    results_df = pd.DataFrame(results)
    results_df.to_parquet(results_path, index=False)
    
    return results_df


# ============================================================================
# PART 9: ANALYSIS AND VISUALIZATION
# ============================================================================

def analyze_results(results_df: pd.DataFrame, output_dir: str):
    """Analyze and visualize results with comprehensive metrics."""
    print("\n" + "="*60)
    print("GRADIENT STEERING RESULTS (STD-NORMALIZED)")
    print("="*60)
    
    baseline = results_df[results_df['block_set'] == 'baseline']
    print(f"\nBaseline: {len(baseline)} cases")
    print(f"  Mean Cβ distance: {baseline['mean_cb_dist'].mean():.2f}Å")
    print(f"  Mean contacts (6Å): {baseline['n_contacts_6A'].mean():.1f}")
    print(f"  Mean potential H-bonds: {baseline['n_potential_hbonds'].mean():.1f}")
    print(f"  Mean region pLDDT: {baseline['region_plddt'].mean():.1f}")

    interventions = results_df[results_df['block_set'] != 'baseline']

    print(f"\nInterventions: {len(interventions)} total")
    
    # Get unique window sizes and magnitudes (window_size>0 defensively excludes any
    # leftover baseline-style rows, since baseline rows use window_size=0)
    window_sizes = sorted([ws for ws in interventions['window_size'].unique() if ws > 0])
    magnitudes = sorted(interventions['magnitude'].unique())
    
    if len(window_sizes) == 0 or len(magnitudes) == 0:
        print("No intervention data to analyze.")
        return
    
    print("\nResults by magnitude (σ):")
    for mag in magnitudes:
        subset = interventions[interventions['magnitude'] == mag]

        mean_cb_change = subset['cb_dist_change'].mean()
        mean_plddt_change = subset['plddt_change'].mean()
        mean_contacts_change = subset['contacts_6A_change'].mean()
        mean_hbond_change = subset['hbond_change'].mean()

        print(f"  mag={mag:.1f}σ: "
              f"ΔCβ={mean_cb_change:+.1f}Å, ΔpLDDT={mean_plddt_change:+.1f}, "
              f"Δcontacts={mean_contacts_change:+.1f}, Δhbonds={mean_hbond_change:+.1f}")
    
    # Plotting - 3x2 grid: distance/contact/H-bond/pLDDT change vs magnitude (top two
    # rows + pLDDT), plus distance-to-target and H-bond change vs window position (bottom row)
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    
    # Color map for window sizes
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(window_sizes)))
    ws_colors = {ws: colors[i] for i, ws in enumerate(window_sizes)}
    
    # Top-left: Cβ distance change by magnitude
    ax = axes[0, 0]
    for ws in window_sizes:
        ws_data = interventions[interventions['window_size'] == ws]
        means = [ws_data[ws_data['magnitude'] == m]['cb_dist_change'].mean() for m in magnitudes]
        # Standard error of the mean; +1 in the denominator guards against a
        # division by zero if a (window_size, magnitude) group is empty.
        stds = [ws_data[ws_data['magnitude'] == m]['cb_dist_change'].std() / 
                np.sqrt(len(ws_data[ws_data['magnitude'] == m]) + 1) for m in magnitudes]
        ax.errorbar(magnitudes, means, yerr=stds, fmt='o-', label=f'w={ws}', 
                   color=ws_colors[ws], linewidth=2, markersize=5, capsize=3)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Magnitude (σ)', fontsize=12)
    ax.set_ylabel('Cβ Distance Change (Å)', fontsize=12)
    ax.set_title('Cross-Strand Cβ Distance Change', fontsize=13)
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)
    
    # Top-right: Close contacts (6Å) change by magnitude
    ax = axes[0, 1]
    for ws in window_sizes:
        ws_data = interventions[interventions['window_size'] == ws]
        means = [ws_data[ws_data['magnitude'] == m]['contacts_6A_change'].mean() for m in magnitudes]
        ax.plot(magnitudes, means, 'o-', label=f'w={ws}', color=ws_colors[ws], linewidth=2, markersize=5)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Magnitude (σ)', fontsize=12)
    ax.set_ylabel('Change in Close Contacts (<6Å)', fontsize=12)
    ax.set_title('Close Contact Formation', fontsize=13)
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)
    
    # Middle-left: Potential H-bonds change by magnitude
    ax = axes[1, 0]
    for ws in window_sizes:
        ws_data = interventions[interventions['window_size'] == ws]
        means = [ws_data[ws_data['magnitude'] == m]['hbond_change'].mean() for m in magnitudes]
        ax.plot(magnitudes, means, 'o-', label=f'w={ws}', color=ws_colors[ws], linewidth=2, markersize=5)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Magnitude (σ)', fontsize=12)
    ax.set_ylabel('Change in Potential H-bonds', fontsize=12)
    ax.set_title('H-bond Formation (N-O < 3.5Å)', fontsize=13)
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)
    
    # Middle-right: pLDDT change by magnitude
    ax = axes[1, 1]
    for ws in window_sizes:
        ws_data = interventions[interventions['window_size'] == ws]
        means = [ws_data[ws_data['magnitude'] == m]['plddt_change'].mean() for m in magnitudes]
        ax.plot(magnitudes, means, 'o-', label=f'w={ws}', color=ws_colors[ws], linewidth=2, markersize=5)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Magnitude (σ)', fontsize=12)
    ax.set_ylabel('Region pLDDT Change', fontsize=12)
    ax.set_title('Structure Quality Change', fontsize=13)
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)
    
    # Bottom-left: Distance to target by window start
    ax = axes[2, 0]
    for ws in window_sizes:
        ws_data = interventions[interventions['window_size'] == ws]
        window_starts = sorted(ws_data['window_start'].unique())
        means = [ws_data[ws_data['window_start'] == wstart]['dist_to_target'].mean() for wstart in window_starts]
        ax.plot(window_starts, means, 'o-', label=f'w={ws}', color=ws_colors[ws], linewidth=2, markersize=4)
    
    # Add baseline reference
    baseline_dist_to_target = baseline['dist_to_target'].mean()
    ax.axhline(y=baseline_dist_to_target, color='red', linestyle='--', alpha=0.7, label='Baseline')
    
    ax.set_xlabel('Window Start Block', fontsize=12)
    ax.set_ylabel('Distance to Target (Å)', fontsize=12)
    ax.set_title(f'Distance to Target ({TARGET_DISTANCE}Å)', fontsize=13)
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)
    
    # Bottom-right: H-bond change by window start
    ax = axes[2, 1]
    for ws in window_sizes:
        ws_data = interventions[interventions['window_size'] == ws]
        window_starts = sorted(ws_data['window_start'].unique())
        means = [ws_data[ws_data['window_start'] == wstart]['hbond_change'].mean() for wstart in window_starts]
        ax.plot(window_starts, means, 'o-', label=f'w={ws}', color=ws_colors[ws], linewidth=2, markersize=4)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Window Start Block', fontsize=12)
    ax.set_ylabel('H-bond Change', fontsize=12)
    ax.set_title('H-bond Change by Window Position', fontsize=13)
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_steering_summary.png'), dpi=150)
    plt.close()
    
    # Heatmap: contacts gained by window_start x window_size
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Contacts gained heatmap
    ax = axes[0]
    # Build a long-form (window_size, window_start, mean_value) table, then pivot it
    # into a 2D grid so it can be rendered as an image with imshow.
    pivot_data = []
    for ws in window_sizes:
        ws_data = interventions[interventions['window_size'] == ws]
        window_starts = sorted(ws_data['window_start'].unique())
        for wstart in window_starts:
            subset = ws_data[ws_data['window_start'] == wstart]
            mean_change = subset['contacts_6A_change'].mean()
            pivot_data.append({'window_size': ws, 'window_start': wstart, 'value': mean_change})
    
    if len(pivot_data) > 0:
        pivot_df = pd.DataFrame(pivot_data)
        # Rows = window_size, columns = window_start; cells missing for a given
        # (size, start) combo (e.g. larger windows have fewer valid starts) become NaN.
        pivot_table = pivot_df.pivot(index='window_size', columns='window_start', values='value')
        
        # extent maps pixel edges to axis coordinates so gridlines/ticks line up with
        # cell centers; vmin/vmax fix the diverging colormap's zero-point at "no change".
        im = ax.imshow(pivot_table.values, aspect='auto', cmap='RdYlGn', 
                       extent=[pivot_table.columns.min()-0.5, pivot_table.columns.max()+0.5,
                               len(window_sizes)-0.5, -0.5],
                       vmin=-2, vmax=2)
        
        ax.set_yticks(range(len(window_sizes)))
        ax.set_yticklabels(window_sizes)
        ax.set_xlabel('Window Start Block', fontsize=12)
        ax.set_ylabel('Window Size', fontsize=12)
        ax.set_title('Mean Close Contacts Change (<6Å)', fontsize=13)
        plt.colorbar(im, ax=ax, label='Δ Contacts')
    
    # pLDDT change heatmap
    ax = axes[1]
    pivot_data = []
    for ws in window_sizes:
        ws_data = interventions[interventions['window_size'] == ws]
        window_starts = sorted(ws_data['window_start'].unique())
        for wstart in window_starts:
            subset = ws_data[ws_data['window_start'] == wstart]
            mean_change = subset['plddt_change'].mean()
            pivot_data.append({'window_size': ws, 'window_start': wstart, 'value': mean_change})
    
    if len(pivot_data) > 0:
        pivot_df = pd.DataFrame(pivot_data)
        pivot_table = pivot_df.pivot(index='window_size', columns='window_start', values='value')
        
        # Asymmetric color range: pLDDT changes are expected to be mostly negative
        # (steering can substantially degrade structure confidence) with only small
        # positive gains possible, so the scale is skewed accordingly (-20 to +5).
        im = ax.imshow(pivot_table.values, aspect='auto', cmap='RdYlGn', 
                       extent=[pivot_table.columns.min()-0.5, pivot_table.columns.max()+0.5,
                               len(window_sizes)-0.5, -0.5],
                       vmin=-20, vmax=5)
        
        ax.set_yticks(range(len(window_sizes)))
        ax.set_yticklabels(window_sizes)
        ax.set_xlabel('Window Start Block', fontsize=12)
        ax.set_ylabel('Window Size', fontsize=12)
        ax.set_title('Mean Region pLDDT Change', fontsize=13)
        plt.colorbar(im, ax=ax, label='Δ pLDDT')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_steering_heatmaps.png'), dpi=150)
    plt.close()
    
    print(f"\nPlots saved to {output_dir}/")


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    """Parse command-line arguments for the contact steering experiment script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--probing_dataset', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'data', 'probing_train_test.csv'),
                        help='Path to probing dataset CSV with train/test splits')
    parser.add_argument('--patch_dataset', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'data', 'single_block_patching_successes.csv'),
                        help='Path to patching successes CSV')
    parser.add_argument('--output', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'results', 'contact_steering'))
    parser.add_argument('--directions_path', type=str, default=None,
                        help='Path to load/save gradient directions')
    parser.add_argument('--n_cases', type=int, default=None,
                        help='Number of patch cases to run')
    parser.add_argument('--magnitudes', type=float, nargs='+',
                        default=[20.0],
                        help='Magnitude values in units of std')
    parser.add_argument('--target_distance', type=float, default=TARGET_DISTANCE,
                        help=f'Target Cβ-Cβ distance in Angstroms (default: {TARGET_DISTANCE})')
    parser.add_argument('--window_sizes', type=int, nargs='+', default=[10],
                        help='List of window sizes to test')
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--save_all_pdbs_cases', type=int, default=2,
                        help='Save all PDBs for first N cases (0 to disable)')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_pairs_per_protein', type=int, default=500,
                        help='Number of residue pairs to sample per protein (default: 500)')
    return parser.parse_args()


def main():
    """Train/load distance probes, then run the sliding-window contact steering sweep and analysis."""
    args = parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Target distance: {args.target_distance}Å")
    print(f"Magnitudes (σ): {args.magnitudes}")
    print(f"Save all PDBs for first {args.save_all_pdbs_cases} cases")
    print(f"Pairs per protein: {args.n_pairs_per_protein}")
    
    # Load model
    model, tokenizer = load_esmfold(device)

    # =========================================================================
    # STEP 1: Train or Load Probes
    # =========================================================================
    if args.directions_path and os.path.exists(args.directions_path):
        print(f"\nLoading gradient directions from {args.directions_path}...")
        directions = load_directions(args.directions_path)
        print(f"Loaded directions for {len(directions.directions)} blocks")
    else:
        if args.probing_dataset is None:
            raise ValueError("Must provide either --directions_path (existing file) or --probing_dataset to train new probes")

        directions_save_path = args.directions_path or os.path.join(args.output, 'gradient_directions.pt')

        print("\n" + "="*60)
        print("STEP 1: TRAIN DISTANCE PROBES")
        print("="*60)

        # Load probing dataset
        print(f"\nLoading probing dataset from {args.probing_dataset}...")
        probing_df = pd.read_csv(args.probing_dataset)

        # Show dataset composition
        print(f"\nDataset composition:")
        if 'split' in probing_df.columns and 'structure_type' in probing_df.columns:
            print(probing_df.groupby(['split', 'structure_type']).size().unstack(fill_value=0))

        # Get train sequences (use all train sequences for distance probe training)
        train_df = probing_df[probing_df['split'] == 'train'] if 'split' in probing_df.columns else probing_df

        # Handle column names
        seq_col = 'FullChainSequence' if 'FullChainSequence' in train_df.columns else 'sequence'
        train_seqs = train_df[seq_col].tolist()
        train_labels = train_df['structure_type'].tolist() if 'structure_type' in train_df.columns else ['unknown'] * len(train_seqs)

        # Filter for valid AA sequences
        valid_train = [(s, l) for s, l in zip(train_seqs, train_labels)
                       if all(aa in AA_TO_IDX for aa in s)]
        train_seqs = [s for s, l in valid_train]
        train_labels = [l for s, l in valid_train]

        print(f"\nTrain sequences after filtering: {len(train_seqs)}")

        # Shuffle training data
        combined = list(zip(train_seqs, train_labels))
        random.shuffle(combined)
        train_seqs, train_labels = zip(*combined)
        train_seqs, train_labels = list(train_seqs), list(train_labels)

        # Train probes
        directions = train_distance_probes_chunked(
            sequences=train_seqs,
            labels=train_labels,
            model=model,
            tokenizer=tokenizer,
            device=device,
            n_pairs_per_protein=args.n_pairs_per_protein,
        )

        # Save directions
        save_directions(directions, directions_save_path)

        print("\n" + "="*60)
        print("STEP 2: EVALUATE PROBES")
        print("="*60)

        # Get test sequences
        test_df = probing_df[probing_df['split'] == 'test'] if 'split' in probing_df.columns else probing_df

        seq_col = 'FullChainSequence' if 'FullChainSequence' in test_df.columns else 'sequence'
        test_seqs = test_df[seq_col].tolist()

        # Filter for valid AA sequences
        test_seqs = [s for s in test_seqs if all(aa in AA_TO_IDX for aa in s)]

        print(f"\nTest sequences after filtering: {len(test_seqs)}")

        # Evaluate using imported function (cap at 50 sequences to keep evaluation fast)
        eval_df = evaluate_probes_online(
            probes=directions.probes,
            sequences=test_seqs[:50],
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

        # Save and plot evaluation
        eval_df.to_csv(os.path.join(args.output, 'probe_evaluation.csv'), index=False)
        plot_probe_results(eval_df, args.output)

        print("\nProbe Evaluation Results:")
        print(eval_df.to_string(index=False))
    
    # =========================================================================
    # STEP 3: Run Steering Experiments
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 3: STD-NORMALIZED STEERING EXPERIMENTS")
    print("="*60)
    
    # Load patch dataset
    print(f"\nLoading patch dataset from {args.patch_dataset}...")
    patch_df = pd.read_csv(args.patch_dataset)
    print(f"Total rows: {len(patch_df)}")
    
    # The single_block_patching_successes.csv is already filtered for successful patches
    # Just limit if requested
    if args.n_cases is not None:
        patch_df = patch_df.head(args.n_cases)
        print(f"Limited to {len(patch_df)} cases")
    
    patch_df = patch_df.reset_index(drop=True)
    
    if len(patch_df) == 0:
        print("No cases to run after filtering!")
        return
    
    # Set up sliding window block sets for all window sizes
    all_block_sets = []
    for window_size in args.window_sizes:
        block_sets = get_sliding_window_block_sets(
            window_size=window_size,
            total_blocks=NUM_BLOCKS,
        )
        all_block_sets.extend(block_sets)
    
    print(f"\nWindow sizes: {args.window_sizes}")
    print(f"Total number of windows across all sizes: {len(all_block_sets)}")
    print(f"Magnitudes (σ): {args.magnitudes}")
    print(f"Target distance: {args.target_distance}Å")
    
    # Run experiments
    results_df = run_gradient_steering_experiment(
        cases_df=patch_df,
        model=model,
        tokenizer=tokenizer,
        device=device,
        directions=directions,
        magnitudes=args.magnitudes,
        block_sets=all_block_sets,
        output_dir=args.output,
        target_distance=args.target_distance,
        save_every=args.save_every,
        save_all_pdbs_cases=args.save_all_pdbs_cases,
    )
    
    if len(results_df) > 0:
        analyze_results(results_df, args.output)
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    print(f"\nOutput saved to: {args.output}/")
    print(f"  - Results: gradient_steering_results.parquet")
    print(f"  - Induced hairpin PDBs: induced_hairpins/")
    if args.save_all_pdbs_cases > 0:
        print(f"  - All PDBs for first {args.save_all_pdbs_cases} cases: all_pdbs/")


if __name__ == '__main__':
    main()