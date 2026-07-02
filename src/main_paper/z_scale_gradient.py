"""
Z Scaling Gradient Analysis
===========================

This script computes the gradient of mean pairwise CA distance with respect to 
scaling the pairwise representation (z) before the structure module.

Key insight: We don't want gradients flowing through the entire model (memory explosion).
We only care about the structure module's sensitivity to z scaling.

Approach:
1. Run model up to structure module with torch.no_grad()
2. Detach s and z tensors
3. Create learnable z_scale parameter
4. Run structure module with gradients enabled
5. Compute mean pairwise distance and backprop to get gradient

This tells us: "How much does the output geometry change per unit change in z scale?"
"""

import os
import sys
import types
import argparse
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from transformers import EsmForProteinFolding, AutoTokenizer
# Rigid/Rotation: OpenFold-style rigid-body (rotation + translation) transform utilities used
# below to hand-replicate the structure module's per-block backbone frame update math.
from transformers.models.esm.openfold_utils import Rigid, Rotation

from src.utils.model_utils import load_esmfold  # shared loader: handles precision/device/eval-mode setup


# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_PARQUET_PATH = 'data/block_patching_successes.csv'
DEFAULT_OUTPUT_DIR = './z_gradient_analysis'
DEFAULT_N_CASES = 400


# ============================================================================
# GEOMETRY UTILITIES (must be differentiable!)
# ============================================================================
# These recompute structural metrics as tensor ops (no .item()/numpy conversions) so that
# autograd can trace a path from the final metric value back to the z_scale/s_scale parameter.
def compute_mean_ca_distance_differentiable(positions: torch.Tensor) -> torch.Tensor:
    """
    Compute mean CA-CA distance (differentiable).
    
    Args:
        positions: [batch, seq_len, 14, 3] or [seq_len, 14, 3]
    
    Returns:
        Scalar tensor (mean pairwise CA distance)
    """
    if positions.dim() == 3:
        # Add a batch dim if a single (unbatched) structure was passed, so the rest of the
        # function can always assume a leading batch axis.
        positions = positions.unsqueeze(0)
    
    # CA is atom index 1
    # (atom14 layout: N=0, CA=1, C=2, O=3, then side-chain atoms - CA is always index 1)
    ca_pos = positions[:, :, 1, :]  # [batch, seq_len, 3]
    
    # Compute pairwise distances
    # (broadcast outer difference over the seq_len axis -> all pairwise displacement vectors)
    diff = ca_pos.unsqueeze(2) - ca_pos.unsqueeze(1)  # [batch, seq_len, seq_len, 3]
    # +1e-8 avoids a NaN gradient at sqrt(0) (the diagonal has zero self-distance)
    distances = torch.sqrt((diff ** 2).sum(-1) + 1e-8)  # [batch, seq_len, seq_len]
    
    # Mean of upper triangle (excluding diagonal)
    seq_len = distances.shape[1]
    # triu(diagonal=1) selects each unique (i<j) pair exactly once, skipping the zero-valued
    # diagonal (self-distance) and avoiding double-counting the symmetric (j, i) entry.
    mask = torch.triu(torch.ones(seq_len, seq_len, device=distances.device), diagonal=1)
    n_pairs = mask.sum()
    mean_dist = (distances * mask).sum() / n_pairs
    
    return mean_dist


def compute_radius_of_gyration_differentiable(positions: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Compute radius of gyration for a region (differentiable)."""
    if positions.dim() == 3:
        positions = positions.unsqueeze(0)
    
    # Indexes batch element 0 only (this script always runs one sequence at a time, so B == 1).
    ca_pos = positions[0, start:end, 1, :]  # [region_len, 3]
    # Center of mass of the region
    com = ca_pos.mean(dim=0)  # [3]
    diff = ca_pos - com
    rg = torch.sqrt((diff ** 2).sum(-1).mean() + 1e-8)  # +1e-8 avoids NaN gradient at sqrt(0)
    
    return rg


def compute_local_ca_distance_differentiable(positions: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Compute mean CA distance within a region (differentiable)."""
    if positions.dim() == 3:
        positions = positions.unsqueeze(0)
    
    ca_pos = positions[0, start:end, 1, :]  # [region_len, 3]
    
    diff = ca_pos.unsqueeze(1) - ca_pos.unsqueeze(0)  # [region_len, region_len, 3]
    distances = torch.sqrt((diff ** 2).sum(-1) + 1e-8)
    
    region_len = distances.shape[0]
    # Same upper-triangle trick as above: count each residue pair within the region once.
    mask = torch.triu(torch.ones(region_len, region_len, device=distances.device), diagonal=1)
    n_pairs = mask.sum()
    mean_dist = (distances * mask).sum() / (n_pairs + 1e-8)  # +1e-8 also guards region_len<2 (n_pairs=0)
    
    return mean_dist


# ============================================================================
# MODIFIED TRUNK FORWARD TO INTERCEPT BEFORE STRUCTURE MODULE
# ============================================================================
class TrunkOutputs:
    """Container for trunk outputs before structure module."""
    def __init__(self, s_s, s_z, s_s_proj, s_z_proj, aa, position_ids, mask):
        self.s_s = s_s              # Trunk sequence state
        self.s_z = s_z              # Trunk pairwise state
        self.s_s_proj = s_s_proj    # Projected for structure module (single)
        self.s_z_proj = s_z_proj    # Projected for structure module (pair)
        self.aa = aa                # Amino acid types
        self.position_ids = position_ids
        self.mask = mask


def get_trunk_outputs(model, tokenizer, device, sequence: str, num_recycles: int = 0) -> TrunkOutputs:
    """
    Run ESMFold up to (but not including) the structure module.
    Returns the intermediate representations that would be fed to the structure module.
    """
    # STEP 1 (see module docstring): the entire trunk forward pass (ESM-2 encoder, all
    # evoformer-like blocks, and every recycle iteration) runs under torch.no_grad(), so no
    # autograd graph is built for any of it. This is intentionally a fixed feature-extraction
    # pass - gradients are only needed later, for the structure module, once a fresh scale
    # parameter is introduced (see compute_z_scale_gradient/compute_s_scale_gradient below).
    with torch.no_grad():
        # Tokenize
        inputs = tokenizer(sequence, return_tensors='pt', add_special_tokens=False).to(device)
        input_ids = inputs['input_ids']
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        
        cfg = model.config.esmfold_config
        aa = input_ids
        B, L = aa.shape
        
        # ESM language model
        esmaa = model.af2_idx_to_esm_idx(aa, attention_mask)  # remap AF2 vocab ids -> ESM-2 vocab ids
        esm_s = model.compute_language_model_representations(esmaa)  # runs full 36-layer ESM-2 encoder, all layers' hidden states
        esm_s = esm_s.to(model.esm_s_combine.dtype)  # match dtype of the learned per-layer mixing weights below
        
        if cfg.esm_ablate_sequence:
            esm_s = esm_s * 0  # config-gated ablation switch (unused here, inherited from base model forward)
        
        esm_s = esm_s.detach()  # no-op under no_grad(); kept to mirror the original model code exactly
        
        # Preprocessing
        # esm_s_combine: one learned scalar weight per ESM-2 layer; softmax -> convex combination
        # across layers, then matmul collapses the per-layer axis into a single per-residue vector.
        esm_s = (model.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
        s_s_0 = model.esm_s_mlp(esm_s)  # project combined ESM-2 features down to the trunk's working dim (c_s)
        s_z_0 = s_s_0.new_zeros(B, L, L, cfg.trunk.pairwise_state_dim)  # pairwise state starts at zero; built up by the blocks
        
        if model.config.esmfold_config.embed_aa:
            s_s_0 = s_s_0 + model.embedding(aa)  # optionally add a raw learned per-amino-acid identity embedding
        
        # Run trunk (evoformer-like blocks)
        trunk = model.trunk
        
        # "+1": the first trunk pass is the ordinary forward pass, not itself a "recycle"
        # (recycling = extra passes that feed back the previous pass's output as input).
        no_recycles = num_recycles if num_recycles is not None else trunk.config.max_recycles
        no_recycles += 1
        
        s_s = s_s_0
        s_z = s_z_0
        # Zero-initialized so the first iteration's "recycled" contribution below is a no-op.
        recycle_s = torch.zeros_like(s_s)
        recycle_z = torch.zeros_like(s_z)
        recycle_bins = torch.zeros(*s_z.shape[:-1], device=device, dtype=torch.int64)
        
        def trunk_iter(s, z, residx, mask):
            """Add relative-position pairwise embedding, then run one pass through all trunk blocks."""
            z = z + trunk.pairwise_positional_embedding(residx, mask=mask)
            for block in trunk.blocks:
                s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=trunk.chunk_size)
            return s, z
        
        for recycle_idx in range(no_recycles):
            # .detach() here mirrors the real EsmFoldingTrunk.forward, which only wraps
            # non-final recycle iterations in no_grad and needs an explicit detach to cut the
            # graph between iterations. Here everything is already inside the outer no_grad(),
            # so these detach() calls are inert - kept for parity with the original code. Note
            # this is a deliberate divergence from the real model: the real forward keeps its
            # LAST recycle iteration differentiable, but this helper forces every iteration
            # (including the last) through no_grad, since no trunk-level gradients are needed.
            recycle_s = trunk.recycle_s_norm(recycle_s.detach())
            recycle_z = trunk.recycle_z_norm(recycle_z.detach())
            # Embed the previous iteration's predicted CA-CA distance histogram as an additive
            # pairwise feature - this is how coordinates from the last recycle feed back into z.
            recycle_z = recycle_z + trunk.recycle_disto(recycle_bins.detach())
            
            s_s, s_z = trunk_iter(s_s_0 + recycle_s, s_z_0 + recycle_z, position_ids, attention_mask)
            
            if recycle_idx < no_recycles - 1:
                # Run structure module to get recycle bins (but don't save)
                # Calls the ORIGINAL (unscaled) structure module purely to get positions for
                # the next iteration's recycling distogram; unrelated to the scaling experiment.
                # Skipped on the final iteration since there is no next iteration to feed.
                structure = trunk.structure_module(
                    {"single": trunk.trunk2sm_s(s_s), "pair": trunk.trunk2sm_z(s_z)},
                    aa,
                    attention_mask.float(),
                )
                recycle_s = s_s
                recycle_z = s_z
                # positions[-1]: final IPA block's coords (dict_multimap-stacked per-block outputs).
                # [:, :, :3] keeps only N, CA, C (atom14 indices 0-2), needed to infer CB and bin
                # distances. 3.375/21.375 A are AlphaFold's standard distogram bin-range bounds;
                # trunk.recycle_bins (=15) is the number of bins.
                recycle_bins = trunk.distogram(
                    structure["positions"][-1][:, :, :3],
                    3.375,
                    21.375,
                    trunk.recycle_bins,
                )
        
        # Project to structure module dimensions
        # These are the actual tensors the real model feeds into the structure module, and the
        # ones the scaling/gradient experiments below manipulate.
        s_s_proj = trunk.trunk2sm_s(s_s)
        s_z_proj = trunk.trunk2sm_z(s_z)
        
        # STEP 2 (module docstring): detach everything before returning. This is redundant given
        # the outer no_grad() (nothing here has a grad_fn anyway), but makes explicit that the
        # returned tensors are plain constants with no ties back into this trunk computation -
        # the next step builds a brand new, much smaller autograd graph rooted only at a fresh
        # scale parameter, which is what keeps the eventual backward pass cheap.
        return TrunkOutputs(
            s_s=s_s.detach(),
            s_z=s_z.detach(),
            s_s_proj=s_s_proj.detach(),
            s_z_proj=s_z_proj.detach(),
            aa=aa.detach(),
            position_ids=position_ids.detach(),
            mask=attention_mask.float().detach(),
        )


# ============================================================================
# GRADIENT COMPUTATION
# ============================================================================
def compute_z_scale_gradient(
    model,
    trunk_outputs: TrunkOutputs,
    z_scale_value: float = 1.0,
    metric: str = 'mean_ca_dist',
    hp_start: Optional[int] = None,
    hp_end: Optional[int] = None,
    debug: bool = False,
    verify_numerical: bool = False,
) -> Dict[str, float]:
    """
    Compute the gradient of a structural metric with respect to z_scale.
    
    NOTE: We need to scale z AFTER the layer norm inside the structure module,
    otherwise the layer norm will normalize away the scaling effect!
    
    Args:
        model: ESMFold model
        trunk_outputs: Pre-computed trunk outputs
        z_scale_value: The point at which to compute the gradient
        metric: Which metric to compute gradient for
        hp_start, hp_end: Hairpin region (for local metrics)
        debug: If True, print debug info
        verify_numerical: If True, also compute numerical gradient for comparison
    
    Returns:
        Dictionary with gradient and metric value
    """
    from transformers.models.esm.modeling_esmfold import dict_multimap
    
    device = trunk_outputs.s_s_proj.device
    dtype = trunk_outputs.s_z_proj.dtype
    
    # Get inputs
    # .clone() a fresh copy of the (already-detached, no-grad) cached trunk tensors: trunk_outputs
    # may be reused across many scale values/metrics, and cloning avoids autograd in-place/version
    # issues from starting multiple independent graphs off the exact same underlying tensor storage.
    s = trunk_outputs.s_s_proj.clone()
    z = trunk_outputs.s_z_proj.clone()
    aa = trunk_outputs.aa
    mask = trunk_outputs.mask
    
    # Create z_scale as a learnable parameter
    # STEP 3 (module docstring). This scalar is the ONLY leaf tensor with requires_grad=True in
    # this function. Since s/z/aa are plain constants, autograd will only build a graph along
    # the path that flows through z_scale - i.e. just the structure module ops below, not the
    # trunk that produced s/z.
    z_scale = torch.tensor(z_scale_value, dtype=dtype, device=device, requires_grad=True)
    
    structure_module = model.trunk.structure_module
    
    # STEP 4 (module docstring): manually run the structure module with gradients enabled so a
    # scale factor can be injected. We can't just call structure_module(...) directly because its
    # forward() has no hook for a scale parameter, so its logic is hand-unrolled here instead.
    # We need to manually run the structure module with scaling AFTER layer norm
    # This mimics what the working z_vs_s_scaling_experiment.py does
    
    if mask is None:
        mask = s.new_ones(s.shape[:-1])
    
    # Apply layer norms first
    s_normed = structure_module.layer_norm_s(s)
    z_normed = structure_module.layer_norm_z(z)
    
    # SCALE Z HERE - after layer norm!
    # (scaling BEFORE the norm would be pointless: LayerNorm renormalizes to fixed mean/variance
    # regardless of input scale, so any pre-norm scaling would just get normalized back away)
    z_scaled = z_normed * z_scale
    
    if debug:
        print(f"  z dtype: {z.dtype}, z_scale dtype: {z_scale.dtype}")
        print(f"  z_normed mean: {z_normed.mean().item():.4f}, std: {z_normed.std().item():.4f}")
        print(f"  z_scaled mean: {z_scaled.mean().item():.4f}, std: {z_scaled.std().item():.4f}")
        print(f"  z_scaled requires_grad: {z_scaled.requires_grad}")
    
    # Continue with structure module forward pass
    s_initial = s_normed  # unscaled here since this function only perturbs z, not s
    s_current = structure_module.linear_in(s_normed)
    
    # Identity rotation+translation frame per residue - the starting backbone frame before
    # any IPA block has run.
    rigids = Rigid.identity(s_current.shape[:-1], s_current.dtype, s_current.device, 
                           structure_module.training, fmt="quat")
    
    outputs = []
    # Re-implementation of EsmFoldStructureModule.forward's block loop, with z_scaled substituted
    # for the ordinary z. Each of the `num_blocks` iterations refines s_current/rigids using IPA
    # (invariant point attention, conditioned on both s and the pairwise z_scaled) and records a
    # full structure prediction; only the LAST block's positions are used below (positions[-1]).
    for i in range(structure_module.config.num_blocks):
        s_current = s_current + structure_module.ipa(s_current, z_scaled, rigids, mask)
        s_current = structure_module.ipa_dropout(s_current)
        s_current = structure_module.layer_norm_ipa(s_current)
        s_current = structure_module.transition(s_current)
        
        rigids = rigids.compose_q_update_vec(structure_module.bb_update(s_current))
        
        # Convert quaternion-based rigids to rotation-matrix form to match AlphaFold's convention.
        backb_to_global = Rigid(
            Rotation(rot_mats=rigids.get_rots().get_rot_mats(), quats=None),
            rigids.get_trans(),
        )
        # trans_scale_factor rescales from the model's internal (roughly unit-scale) training
        # coordinates back up to physical Angstrom units.
        backb_to_global = backb_to_global.scale_translation(structure_module.config.trans_scale_factor)
        
        unnormalized_angles, angles = structure_module.angle_resnet(s_current, s_initial)
        all_frames_to_global = structure_module.torsion_angles_to_frames(backb_to_global, angles, aa)
        pred_xyz = structure_module.frames_and_literature_positions_to_atom14_pos(all_frames_to_global, aa)
        
        scaled_rigids = rigids.scale_translation(structure_module.config.trans_scale_factor)
        
        preds = {
            "frames": scaled_rigids.to_tensor_7(),
            "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
            "unnormalized_angles": unnormalized_angles,
            "angles": angles,
            "positions": pred_xyz,
            "states": s_current,
        }
        outputs.append(preds)
        # Detach ONLY the rotation component before the next block (standard AlphaFold/OpenFold
        # stabilization trick): prevents rotation gradients from compounding across num_blocks
        # iterations, while translation gradients still flow normally. This is a second, separate
        # gradient-truncation point inside the structure module itself (distinct from the
        # trunk-level no_grad() truncation done earlier in get_trunk_outputs).
        rigids = rigids.stop_rot_gradient()
    
    # Stack the per-block dicts into a single dict of tensors, each with a new leading
    # num_blocks dimension (see dict_multimap in modeling_esmfold.py).
    outputs = dict_multimap(torch.stack, outputs)
    
    # Get final positions
    positions = outputs["positions"][-1]  # [B, N, 14, 3]
    # (index -1 takes the last block's prediction, after all num_blocks IPA refinements)
    
    if debug:
        print(f"  positions requires_grad: {positions.requires_grad}")
        print(f"  positions dtype: {positions.dtype}")
    
    # Compute metric
    if metric == 'mean_ca_dist':
        metric_value = compute_mean_ca_distance_differentiable(positions)
    elif metric == 'full_rg':
        metric_value = compute_radius_of_gyration_differentiable(positions, 0, positions.shape[1])
    elif metric == 'hairpin_ca_dist' and hp_start is not None and hp_end is not None:
        metric_value = compute_local_ca_distance_differentiable(positions, hp_start, hp_end)
    elif metric == 'hairpin_rg' and hp_start is not None and hp_end is not None:
        metric_value = compute_radius_of_gyration_differentiable(positions, hp_start, hp_end)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    if debug:
        print(f"  metric_value: {metric_value.item():.4f}, requires_grad: {metric_value.requires_grad}")
    
    # Backprop
    # STEP 5 (module docstring). Since z_scale is the only requires_grad leaf feeding into
    # metric_value, this is a cheap backward pass through just the structure module (num_blocks
    # IPA iterations), not the trunk or ESM-2 encoder.
    metric_value.backward()
    
    gradient = z_scale.grad  # d(metric_value) / d(z_scale), accumulated into the leaf's .grad
    
    if debug:
        print(f"  z_scale.grad (autodiff): {gradient}")
    
    if gradient is None:
        print(f"  WARNING: gradient is None! The computation graph may be broken.")
        gradient_value = 0.0
    else:
        gradient_value = gradient.item()
    
    # Optionally verify with numerical gradient
    numerical_grad = None
    if verify_numerical:
        eps = 0.1  # finite-difference step size for central-difference approximation
        
        def run_with_scale(scale_val):
            """Recompute final atom14 positions at a given z scale, without autograd (for finite-difference gradient checks)."""
            with torch.no_grad():
                z_s = z_normed * scale_val
                s_i = s_normed
                s_c = structure_module.linear_in(s_normed)
                rigs = Rigid.identity(s_c.shape[:-1], s_c.dtype, s_c.device, False, fmt="quat")
                
                # Same block loop as above, but under no_grad and simplified: since only the
                # FINAL block's positions are needed (not every intermediate block), frames/
                # angles/positions are computed once after the loop instead of on every
                # iteration - mathematically equivalent to outputs["positions"][-1] above,
                # cheaper because no per-block outputs need to be retained.
                for _ in range(structure_module.config.num_blocks):
                    s_c = s_c + structure_module.ipa(s_c, z_s, rigs, mask)
                    s_c = structure_module.ipa_dropout(s_c)
                    s_c = structure_module.layer_norm_ipa(s_c)
                    s_c = structure_module.transition(s_c)
                    rigs = rigs.compose_q_update_vec(structure_module.bb_update(s_c))
                    rigs = rigs.stop_rot_gradient()  # no-op under no_grad(); kept for parity
                
                backb = Rigid(Rotation(rot_mats=rigs.get_rots().get_rot_mats(), quats=None), rigs.get_trans())
                backb = backb.scale_translation(structure_module.config.trans_scale_factor)
                _, angles = structure_module.angle_resnet(s_c, s_i)
                frames = structure_module.torsion_angles_to_frames(backb, angles, aa)
                pos = structure_module.frames_and_literature_positions_to_atom14_pos(frames, aa)
                return pos
        
        # Central difference: (f(x+eps) - f(x-eps)) / (2*eps) approximates f'(x); used as an
        # independent sanity check against the autograd-computed gradient above.
        pos_plus = run_with_scale(z_scale_value + eps)
        pos_minus = run_with_scale(z_scale_value - eps)
        
        if metric == 'mean_ca_dist':
            metric_plus = compute_mean_ca_distance_differentiable(pos_plus).item()
            metric_minus = compute_mean_ca_distance_differentiable(pos_minus).item()
        elif metric == 'full_rg':
            metric_plus = compute_radius_of_gyration_differentiable(pos_plus, 0, pos_plus.shape[1]).item()
            metric_minus = compute_radius_of_gyration_differentiable(pos_minus, 0, pos_minus.shape[1]).item()
        elif metric == 'hairpin_ca_dist':
            metric_plus = compute_local_ca_distance_differentiable(pos_plus, hp_start, hp_end).item()
            metric_minus = compute_local_ca_distance_differentiable(pos_minus, hp_start, hp_end).item()
        elif metric == 'hairpin_rg':
            metric_plus = compute_radius_of_gyration_differentiable(pos_plus, hp_start, hp_end).item()
            metric_minus = compute_radius_of_gyration_differentiable(pos_minus, hp_start, hp_end).item()
        
        numerical_grad = (metric_plus - metric_minus) / (2 * eps)
        
        if debug:
            print(f"  metric at scale {z_scale_value + eps:.2f}: {metric_plus:.4f}")
            print(f"  metric at scale {z_scale_value - eps:.2f}: {metric_minus:.4f}")
            print(f"  numerical gradient: {numerical_grad:.6f}")
            print(f"  autodiff gradient:  {gradient_value:.6e}")
            if abs(gradient_value) > 1e-10:
                print(f"  ratio (numerical/autodiff): {numerical_grad / gradient_value:.2f}")
    
    result = {
        'metric': metric,
        'z_scale': z_scale_value,
        'metric_value': metric_value.item(),
        'gradient': gradient_value,
    }
    
    if numerical_grad is not None:
        result['numerical_gradient'] = numerical_grad
    
    return result


def compute_gradient_across_scales(
    model,
    trunk_outputs: TrunkOutputs,
    scales: List[float],
    metrics: List[str],
    hp_start: Optional[int] = None,
    hp_end: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute gradients at multiple z_scale values for multiple metrics.
    """
    results = []
    
    for scale in tqdm(scales, desc="Computing gradients", leave=False):
        for metric in metrics:
            try:
                result = compute_z_scale_gradient(
                    model, trunk_outputs, z_scale_value=scale,
                    metric=metric, hp_start=hp_start, hp_end=hp_end
                )
                results.append(result)
            except Exception as e:
                # Substitute a NaN row instead of aborting the whole sweep, so one bad
                # (scale, metric) combination doesn't lose results for all the others;
                # pandas aggregations (mean/std used downstream) skip NaNs by default.
                print(f"Error computing gradient for scale={scale}, metric={metric}: {e}")
                results.append({
                    'metric': metric,
                    'z_scale': scale,
                    'metric_value': float('nan'),
                    'gradient': float('nan'),
                })
    
    return pd.DataFrame(results)


# ============================================================================
# ALSO COMPUTE S GRADIENTS FOR COMPARISON
# ============================================================================
def compute_s_scale_gradient(
    model,
    trunk_outputs: TrunkOutputs,
    s_scale_value: float = 1.0,
    metric: str = 'mean_ca_dist',
    hp_start: Optional[int] = None,
    hp_end: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute the gradient of a structural metric with respect to s_scale.
    Scale is applied AFTER layer norm to avoid normalization undoing the effect.
    """
    from transformers.models.esm.modeling_esmfold import dict_multimap
    
    device = trunk_outputs.s_s_proj.device
    dtype = trunk_outputs.s_s_proj.dtype
    
    # Mirrors compute_z_scale_gradient above (same truncation strategy: cloned/detached trunk
    # outputs as constants, gradients only introduced via a fresh scale leaf tensor), but scales
    # s (single/per-residue repr) instead of z (pairwise repr) for the comparison analysis.
    s = trunk_outputs.s_s_proj.clone()
    z = trunk_outputs.s_z_proj.clone()
    aa = trunk_outputs.aa
    mask = trunk_outputs.mask
    
    s_scale = torch.tensor(s_scale_value, dtype=dtype, device=device, requires_grad=True)
    
    structure_module = model.trunk.structure_module
    
    if mask is None:
        mask = s.new_ones(s.shape[:-1])
    
    # Apply layer norms first
    s_normed = structure_module.layer_norm_s(s)
    z_normed = structure_module.layer_norm_z(z)  # left unscaled - only s is perturbed here
    
    # SCALE S HERE - after layer norm!
    s_scaled = s_normed * s_scale
    
    # Continue with structure module forward pass
    s_initial = s_scaled  # Note: s_initial should be the scaled version for angle_resnet
    s_current = structure_module.linear_in(s_scaled)
    
    rigids = Rigid.identity(s_current.shape[:-1], s_current.dtype, s_current.device, 
                           structure_module.training, fmt="quat")
    
    outputs = []
    # Same per-block IPA loop as compute_z_scale_gradient, with z_normed (unscaled) passed in
    # instead of a scaled z, and rigids.stop_rot_gradient() again truncating rotation gradients
    # between blocks (see detailed comments in compute_z_scale_gradient above).
    for i in range(structure_module.config.num_blocks):
        s_current = s_current + structure_module.ipa(s_current, z_normed, rigids, mask)
        s_current = structure_module.ipa_dropout(s_current)
        s_current = structure_module.layer_norm_ipa(s_current)
        s_current = structure_module.transition(s_current)
        
        rigids = rigids.compose_q_update_vec(structure_module.bb_update(s_current))
        
        backb_to_global = Rigid(
            Rotation(rot_mats=rigids.get_rots().get_rot_mats(), quats=None),
            rigids.get_trans(),
        )
        backb_to_global = backb_to_global.scale_translation(structure_module.config.trans_scale_factor)
        
        unnormalized_angles, angles = structure_module.angle_resnet(s_current, s_initial)
        all_frames_to_global = structure_module.torsion_angles_to_frames(backb_to_global, angles, aa)
        pred_xyz = structure_module.frames_and_literature_positions_to_atom14_pos(all_frames_to_global, aa)
        
        scaled_rigids = rigids.scale_translation(structure_module.config.trans_scale_factor)
        
        preds = {
            "frames": scaled_rigids.to_tensor_7(),
            "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
            "unnormalized_angles": unnormalized_angles,
            "angles": angles,
            "positions": pred_xyz,
            "states": s_current,
        }
        outputs.append(preds)
        rigids = rigids.stop_rot_gradient()
    
    outputs = dict_multimap(torch.stack, outputs)
    positions = outputs["positions"][-1]
    
    if metric == 'mean_ca_dist':
        metric_value = compute_mean_ca_distance_differentiable(positions)
    elif metric == 'full_rg':
        metric_value = compute_radius_of_gyration_differentiable(positions, 0, positions.shape[1])
    elif metric == 'hairpin_ca_dist' and hp_start is not None and hp_end is not None:
        metric_value = compute_local_ca_distance_differentiable(positions, hp_start, hp_end)
    elif metric == 'hairpin_rg' and hp_start is not None and hp_end is not None:
        metric_value = compute_radius_of_gyration_differentiable(positions, hp_start, hp_end)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    metric_value.backward()
    gradient = s_scale.grad
    
    if gradient is None:
        gradient_value = 0.0
    else:
        gradient_value = gradient.item()
    
    return {
        'metric': metric,
        's_scale': s_scale_value,
        'metric_value': metric_value.item(),
        'gradient': gradient_value,
    }


def compute_both_gradients(
    model,
    trunk_outputs: TrunkOutputs,
    scales: List[float],
    metrics: List[str],
    hp_start: Optional[int] = None,
    hp_end: Optional[int] = None,
    debug_first: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute gradients for both z and s scaling."""
    
    z_results = []
    s_results = []
    
    first_call = True
    
    for scale in tqdm(scales, desc="Computing gradients"):
        for metric in metrics:
            # Z gradient
            try:
                # debug/verify_numerical only fire once per call to compute_both_gradients (the
                # very first (scale, metric) combination): this runs the autograd-vs-finite-
                # difference sanity check (see verify_numerical in compute_z_scale_gradient)
                # exactly once to build confidence in the gradient setup, without printing/
                # recomputing it for every scale x metric combination.
                z_result = compute_z_scale_gradient(
                    model, trunk_outputs, z_scale_value=scale,
                    metric=metric, hp_start=hp_start, hp_end=hp_end,
                    debug=(debug_first and first_call),
                    verify_numerical=(debug_first and first_call)
                )
                z_results.append(z_result)
                first_call = False
            except Exception as e:
                print(f"Z gradient error: scale={scale}, metric={metric}: {e}")
                import traceback
                traceback.print_exc()
                z_results.append({
                    'metric': metric, 'z_scale': scale,
                    'metric_value': float('nan'), 'gradient': float('nan'),
                })
            
            # S gradient
            try:
                s_result = compute_s_scale_gradient(
                    model, trunk_outputs, s_scale_value=scale,
                    metric=metric, hp_start=hp_start, hp_end=hp_end
                )
                s_results.append(s_result)
            except Exception as e:
                print(f"S gradient error: scale={scale}, metric={metric}: {e}")
                s_results.append({
                    'metric': metric, 's_scale': scale,
                    'metric_value': float('nan'), 'gradient': float('nan'),
                })
        
        torch.cuda.empty_cache()
    
    return pd.DataFrame(z_results), pd.DataFrame(s_results)


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_gradients(
    z_results: pd.DataFrame,
    s_results: pd.DataFrame,
    output_dir: str,
    case_name: str,
):
    """Plot gradient comparison for z vs s scaling."""
    
    metrics = z_results['metric'].unique()
    n_metrics = len(metrics)
    n_scales = len(z_results['z_scale'].unique())
    
    if n_scales == 1:
        # Single scale - just show bar chart of gradients
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        z_grads = [z_results[z_results['metric'] == m]['gradient'].values[0] for m in metrics]
        s_grads = [s_results[s_results['metric'] == m]['gradient'].values[0] for m in metrics]
        
        # Standard grouped-bar-chart layout: one tick per metric at integer position x, with the
        # z/s bars offset by +/- half the bar width so they sit side by side without overlapping.
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, z_grads, width, label='∂/∂(z_scale)', color='steelblue')
        ax.bar(x + width/2, s_grads, width, label='∂/∂(s_scale)', color='indianred')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_ylabel('Gradient (Å per unit scale)')
        ax.set_title(f'Gradients at scale=1.0: {case_name}')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Add ratio annotations
        for i, (zg, sg) in enumerate(zip(z_grads, s_grads)):
            ratio = abs(zg) / (abs(sg) + 1e-10)
            ax.annotate(f'{ratio:.1f}x', xy=(i, max(abs(zg), abs(sg)) * 1.1),
                       ha='center', fontsize=8, color='green' if ratio > 1 else 'purple')
        
        plt.tight_layout()
    else:
        # Multiple scales - show metric value and gradient vs scale
        fig, axes = plt.subplots(2, n_metrics, figsize=(5*n_metrics, 10))
        if n_metrics == 1:
            axes = axes.reshape(2, 1)
        
        for idx, metric in enumerate(metrics):
            z_metric = z_results[z_results['metric'] == metric]
            s_metric = s_results[s_results['metric'] == metric]
            
            # Top row: Metric value vs scale
            ax = axes[0, idx]
            ax.plot(z_metric['z_scale'], z_metric['metric_value'], 'b-o', linewidth=2, markersize=6, label='z scaling')
            ax.plot(s_metric['s_scale'], s_metric['metric_value'], 'r-s', linewidth=2, markersize=6, label='s scaling')
            ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Scale Factor')
            ax.set_ylabel(f'{metric} (Å)')
            ax.set_title(f'{metric}: Value vs Scale')
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Bottom row: Gradient vs scale
            ax = axes[1, idx]
            ax.plot(z_metric['z_scale'], z_metric['gradient'], 'b-o', linewidth=2, markersize=6, label='∂/∂(z_scale)')
            ax.plot(s_metric['s_scale'], s_metric['gradient'], 'r-s', linewidth=2, markersize=6, label='∂/∂(s_scale)')
            ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax.set_xlabel('Scale Factor')
            ax.set_ylabel('Gradient (Å per unit scale)')
            ax.set_title(f'{metric}: Gradient vs Scale')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.suptitle(f'Z vs S Scaling Gradients: {case_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'{case_name}_gradients.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_gradient_summary(
    all_z_results: pd.DataFrame,
    all_s_results: pd.DataFrame,
    output_dir: str,
):
    """Plot summary of gradients across all cases."""
    
    metrics = all_z_results['metric'].unique()
    n_scales = len(all_z_results['z_scale'].unique())
    
    if n_scales == 1:
        # Single scale - simplified summary
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Gradient at scale=1.0 for each metric (bar chart)
        ax = axes[0]
        
        # groupby('metric') indexes the result by metric name, sorted alphabetically by default -
        # NOT necessarily in the same order as `metrics` (from .unique(), i.e. first-seen order).
        # `.loc[metrics, ...]` re-indexes back into `metrics`'s original order so the bars below
        # line up with the correct metric labels (same trick used again further down this file).
        z_at_1 = all_z_results[all_z_results['z_scale'] == 1.0].groupby('metric')['gradient'].agg(['mean', 'std'])
        s_at_1 = all_s_results[all_s_results['s_scale'] == 1.0].groupby('metric')['gradient'].agg(['mean', 'std'])
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, z_at_1.loc[metrics, 'mean'].values, width, 
               yerr=z_at_1.loc[metrics, 'std'].values, 
               label='z gradient', color='steelblue', capsize=3)
        ax.bar(x + width/2, s_at_1.loc[metrics, 'mean'].values, width, 
               yerr=s_at_1.loc[metrics, 'std'].values,
               label='s gradient', color='indianred', capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_ylabel('Gradient at scale=1.0')
        ax.set_title('Gradient Magnitude Comparison')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Plot 2: Gradient ratio (|z|/|s|) at scale=1.0
        ax = axes[1]
        
        ratios = np.abs(z_at_1.loc[metrics, 'mean'].values) / (np.abs(s_at_1.loc[metrics, 'mean'].values) + 1e-10)
        colors = ['steelblue' if r > 1 else 'indianred' for r in ratios]
        bars = ax.bar(range(len(metrics)), ratios, color=colors)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_ylabel('|z gradient| / |s gradient|')
        ax.set_title('Gradient Ratio (Z vs S)')
        ax.grid(alpha=0.3, axis='y')
        
        # Add value labels
        for bar, ratio in zip(bars, ratios):
            ax.annotate(f'{ratio:.2f}x', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Z vs S Scaling: Gradient Analysis Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
    else:
        # Multiple scales - full summary
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Gradient at scale=1.0 for each metric (bar chart)
        ax = axes[0, 0]
        
        z_at_1 = all_z_results[all_z_results['z_scale'] == 1.0].groupby('metric')['gradient'].agg(['mean', 'std'])
        s_at_1 = all_s_results[all_s_results['s_scale'] == 1.0].groupby('metric')['gradient'].agg(['mean', 'std'])
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, z_at_1.loc[metrics, 'mean'].values, width, 
               yerr=z_at_1.loc[metrics, 'std'].values, 
               label='z gradient', color='steelblue', capsize=3)
        ax.bar(x + width/2, s_at_1.loc[metrics, 'mean'].values, width, 
               yerr=s_at_1.loc[metrics, 'std'].values,
               label='s gradient', color='indianred', capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_ylabel('Gradient at scale=1.0')
        ax.set_title('Gradient Magnitude at Normal Scale')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Plot 2: Gradient ratio (z/s) at scale=1.0
        ax = axes[0, 1]
        
        ratios = np.abs(z_at_1.loc[metrics, 'mean'].values) / (np.abs(s_at_1.loc[metrics, 'mean'].values) + 1e-10)
        colors = ['steelblue' if r > 1 else 'indianred' for r in ratios]
        bars = ax.bar(range(len(metrics)), ratios, color=colors)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_ylabel('|z gradient| / |s gradient|')
        ax.set_title('Gradient Ratio at scale=1.0')
        ax.grid(alpha=0.3, axis='y')
        
        # Add value labels
        for bar, ratio in zip(bars, ratios):
            ax.annotate(f'{ratio:.2f}x', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Gradient vs scale for mean_ca_dist (averaged across cases)
        ax = axes[1, 0]
        
        if 'mean_ca_dist' in metrics:
            z_mcd = all_z_results[all_z_results['metric'] == 'mean_ca_dist'].groupby('z_scale')['gradient'].agg(['mean', 'std'])
            s_mcd = all_s_results[all_s_results['metric'] == 'mean_ca_dist'].groupby('s_scale')['gradient'].agg(['mean', 'std'])
            
            ax.errorbar(z_mcd.index, z_mcd['mean'], yerr=z_mcd['std'], fmt='b-o', capsize=3, linewidth=2, label='z gradient')
            ax.errorbar(s_mcd.index, s_mcd['mean'], yerr=s_mcd['std'], fmt='r-s', capsize=3, linewidth=2, label='s gradient')
            ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax.set_xlabel('Scale Factor')
            ax.set_ylabel('Gradient')
            ax.set_title('Mean CA Distance Gradient vs Scale')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Plot 4: Metric value vs scale for mean_ca_dist
        ax = axes[1, 1]
        
        if 'mean_ca_dist' in metrics:
            z_mcd_val = all_z_results[all_z_results['metric'] == 'mean_ca_dist'].groupby('z_scale')['metric_value'].agg(['mean', 'std'])
            s_mcd_val = all_s_results[all_s_results['metric'] == 'mean_ca_dist'].groupby('s_scale')['metric_value'].agg(['mean', 'std'])
            
            ax.errorbar(z_mcd_val.index, z_mcd_val['mean'], yerr=z_mcd_val['std'], fmt='b-o', capsize=3, linewidth=2, label='z scaling')
            ax.errorbar(s_mcd_val.index, s_mcd_val['mean'], yerr=s_mcd_val['std'], fmt='r-s', capsize=3, linewidth=2, label='s scaling')
            ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Scale Factor')
            ax.set_ylabel('Mean CA Distance (Å)')
            ax.set_title('Mean CA Distance vs Scale')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.suptitle('Z vs S Scaling: Gradient Analysis Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'gradient_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def run_gradient_analysis(
    parquet_path: str,
    n_cases: int,
    output_dir: str,
    device: Optional[str] = None,
    scales: Optional[List[float]] = None,
):
    """Run the gradient analysis experiment."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if scales is None:
        scales = [1.0]  # Only need gradient at the normal operating point
    
    # Metrics to analyze
    metrics = ['mean_ca_dist', 'full_rg', 'hairpin_ca_dist', 'hairpin_rg']
    
    # Load model
    # Don't use model.requires_grad_(False) - we need gradients through structure module
    model, tokenizer = load_esmfold(device)
    
    # Load data
    print(f"\nLoading data from {parquet_path}...")
    if parquet_path.endswith('.parquet'):
        df = pd.read_parquet(parquet_path)
    else:
        df = pd.read_csv(parquet_path)
    print(f"Loaded {len(df)} rows")
    
    cases = df.head(n_cases)
    
    all_z_results = []
    all_s_results = []
    
    for idx, row in tqdm(cases.iterrows(), total=len(cases), desc="Analyzing cases"):
        case_name = f"case_{idx}"
        
        target_seq = row['target_sequence']
        hp_start = int(row['target_patch_start'])
        hp_end = int(row['target_patch_end'])
        
        print(f"\n{'='*60}")
        print(f"Case {idx}: {row.get('target_name', 'Unknown')}")
        print(f"Sequence length: {len(target_seq)}")
        print(f"Hairpin region: {hp_start}-{hp_end}")
        print(f"{'='*60}")
        
        # Get trunk outputs (run model up to structure module)
        # Computed once per case: this is the (relatively) expensive no_grad trunk forward pass.
        # It's reused below for every (scale, metric) combination in compute_both_gradients, so
        # only the cheap structure-module-only re-run happens repeatedly, not the whole trunk.
        print("  Computing trunk outputs...")
        trunk_outputs = get_trunk_outputs(model, tokenizer, device, target_seq, num_recycles=0)
        
        # Compute gradients
        print("  Computing gradients...")
        z_results, s_results = compute_both_gradients(
            model, trunk_outputs, scales, metrics,
            hp_start=hp_start, hp_end=hp_end
        )
        
        # Add case info
        z_results['case_idx'] = idx
        z_results['case_name'] = row.get('target_name', f'case_{idx}')
        s_results['case_idx'] = idx
        s_results['case_name'] = row.get('target_name', f'case_{idx}')
        
        all_z_results.append(z_results)
        all_s_results.append(s_results)
        
        # Plot individual case
        plot_gradients(z_results, s_results, output_dir, case_name)
        
        torch.cuda.empty_cache()
    
    # Combine results
    combined_z = pd.concat(all_z_results, ignore_index=True)
    combined_s = pd.concat(all_s_results, ignore_index=True)
    
    # Save results
    combined_z.to_csv(os.path.join(output_dir, 'z_gradients.csv'), index=False)
    combined_s.to_csv(os.path.join(output_dir, 's_gradients.csv'), index=False)
    print(f"\nSaved results to {output_dir}")
    
    # Summary plot
    plot_gradient_summary(combined_z, combined_s, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Gradient Analysis at scale=1.0")
    print("="*60)
    
    z_at_1 = combined_z[combined_z['z_scale'] == 1.0]
    s_at_1 = combined_s[combined_s['s_scale'] == 1.0]
    
    for metric in metrics:
        z_grad = z_at_1[z_at_1['metric'] == metric]['gradient'].mean()
        s_grad = s_at_1[s_at_1['metric'] == metric]['gradient'].mean()
        z_grad_std = z_at_1[z_at_1['metric'] == metric]['gradient'].std()
        s_grad_std = s_at_1[s_at_1['metric'] == metric]['gradient'].std()
        ratio = abs(z_grad) / (abs(s_grad) + 1e-10)
        
        print(f"\n{metric}:")
        print(f"  Z gradient: {z_grad:.6e} ± {z_grad_std:.6e}")
        print(f"  S gradient: {s_grad:.6e} ± {s_grad_std:.6e}")
        print(f"  |Z|/|S| ratio: {ratio:.2f}x")
    
    print(f"\nAll outputs saved to: {output_dir}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    """CLI entry point: parse arguments and run the z/s scale gradient analysis experiment."""
    parser = argparse.ArgumentParser(
        description="Compute gradients of structural metrics w.r.t. z and s scaling"
    )
    parser.add_argument("--parquet", type=str, default=DEFAULT_PARQUET_PATH,
                        help=f"Path to data file (default: {DEFAULT_PARQUET_PATH})")
    parser.add_argument("--n_cases", type=int, default=DEFAULT_N_CASES,
                        help=f"Number of cases to analyze (default: {DEFAULT_N_CASES})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: auto-detect)")
    parser.add_argument("--scales", type=float, nargs='+', default=None,
                        help="Scale factors to compute gradients at (default: 1.0 only)")
    
    args = parser.parse_args()
    
    run_gradient_analysis(
        parquet_path=args.parquet,
        n_cases=args.n_cases,
        output_dir=args.output_dir,
        device=args.device,
        scales=args.scales,
    )


if __name__ == "__main__":
    main()