"""
Z vs S Scaling & Gradient Experiment
=====================================

This script analyzes the relative importance of pair representations (z) vs
single representations (s) by:

1. Scaling them independently before they enter ESMFold's structure module
   and measuring structural metric changes (discrete scaling experiment).
2. Computing autograd gradients of structural metrics w.r.t. z_scale and
   s_scale at the structure module input (gradient analysis).

Hypothesis: If z encodes crucial pairwise distance/geometry information,
scaling z should have a larger effect on output geometry than scaling s.

The structure module receives:
    - s: [batch, N_res, C_s] - per-residue features
    - z: [batch, N_res, N_res, C_z] - pairwise features
"""

import os
import sys
import types
import argparse
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from transformers import EsmForProteinFolding, AutoTokenizer
from src.utils.model_utils import load_esmfold  # shared loader: handles precision/device/eval-mode setup
from transformers.models.esm.modeling_esmfold import (
    EsmFoldingTrunk, 
    EsmForProteinFoldingOutput,
    EsmFoldStructureModule,
)
# Rigid/Rotation: OpenFold-style rigid-body (rotation + translation) transform utilities used
# below to hand-replicate the structure module's per-block backbone frame update math.
from transformers.models.esm.openfold_utils import Rigid, Rotation


# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_PARQUET_PATH = 'data/block_patching_successes.csv'
DEFAULT_OUTPUT_DIR = './z_vs_s_scaling'
DEFAULT_N_CASES = 400


# ============================================================================
# GEOMETRY UTILITIES
# ============================================================================
def compute_ca_distances(positions: torch.Tensor) -> torch.Tensor:
    """
    Compute CA-CA distance matrix from positions.
    
    Args:
        positions: [batch, seq_len, 14, 3] or [seq_len, 14, 3]
    
    Returns:
        Distance matrix
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
    diff = ca_pos.unsqueeze(2) - ca_pos.unsqueeze(1)
    distances = torch.sqrt((diff ** 2).sum(-1) + 1e-8)  # +1e-8 avoids NaN at sqrt(0) on the diagonal
    
    # Undo the batch dim we may have added above, so single-structure callers get back a plain
    # [seq_len, seq_len] matrix instead of a [1, seq_len, seq_len] one.
    return distances.squeeze(0) if distances.shape[0] == 1 else distances


def compute_radius_of_gyration(positions: torch.Tensor, start: int, end: int) -> float:
    """Compute radius of gyration for a region."""
    if positions.dim() == 3:
        positions = positions.unsqueeze(0)
    
    # Indexes batch element 0 only (this script always runs one sequence at a time, so B == 1).
    ca_pos = positions[0, start:end, 1, :]
    com = ca_pos.mean(dim=0)  # center of mass of the region
    diff = ca_pos - com
    rg = torch.sqrt((diff ** 2).sum(-1).mean())
    
    return rg.item()  # not used for backprop, so .item() (plain float) is fine here


def compute_strand_separation(positions: torch.Tensor, hp_start: int, hp_end: int) -> float:
    """Compute average distance between paired residues in a hairpin."""
    if positions.dim() == 3:
        positions = positions.unsqueeze(0)
    
    hp_len = hp_end - hp_start
    half_len = hp_len // 2
    
    ca_pos = positions[0, hp_start:hp_end, 1, :]
    
    # Assumes an antiparallel beta-hairpin: pairs residue i (from the N-terminal strand) with
    # residue (hp_len-1-i) (the mirrored position on the C-terminal strand). `i < j` skips the
    # middle residue when hp_len is odd (i == j there) and avoids counting each pair twice.
    separations = []
    for i in range(half_len):
        j = hp_len - 1 - i
        if i < j:
            dist = torch.sqrt(((ca_pos[i] - ca_pos[j]) ** 2).sum())
            separations.append(dist.item())
    
    return np.mean(separations) if separations else 0.0


def compute_contact_map(positions: torch.Tensor, threshold: float = 8.0) -> torch.Tensor:
    """Compute binary contact map (CA-CA < threshold)."""
    # 8.0 A is a standard CA-CA contact-distance cutoff used in protein structure analysis.
    distances = compute_ca_distances(positions)
    contacts = (distances < threshold).float()
    return contacts


# ============================================================================
# DIFFERENTIABLE GEOMETRY UTILITIES (for gradient computation)
# ============================================================================
# Unlike the plain versions above (which call .item() and return Python floats), these stay as
# tensors end to end, so autograd can trace a path from the metric back to z_scale/s_scale.
def compute_mean_ca_distance_differentiable(positions: torch.Tensor) -> torch.Tensor:
    """Compute mean CA-CA distance (differentiable, returns scalar tensor)."""
    if positions.dim() == 3:
        positions = positions.unsqueeze(0)

    ca_pos = positions[:, :, 1, :]
    diff = ca_pos.unsqueeze(2) - ca_pos.unsqueeze(1)
    distances = torch.sqrt((diff ** 2).sum(-1) + 1e-8)  # +1e-8 avoids NaN gradient at sqrt(0)

    # triu(diagonal=1): each unique (i<j) pair counted once, skipping the zero-valued diagonal.
    seq_len = distances.shape[1]
    mask = torch.triu(torch.ones(seq_len, seq_len, device=distances.device), diagonal=1)
    n_pairs = mask.sum()
    return (distances * mask).sum() / n_pairs


def compute_radius_of_gyration_differentiable(positions: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Compute radius of gyration for a region (differentiable, returns tensor)."""
    if positions.dim() == 3:
        positions = positions.unsqueeze(0)

    # Batch element 0 only (this script always runs one sequence at a time, so B == 1).
    ca_pos = positions[0, start:end, 1, :]
    com = ca_pos.mean(dim=0)
    diff = ca_pos - com
    return torch.sqrt((diff ** 2).sum(-1).mean() + 1e-8)  # +1e-8 avoids NaN gradient at sqrt(0)


def compute_local_ca_distance_differentiable(positions: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Compute mean CA distance within a region (differentiable, returns tensor)."""
    if positions.dim() == 3:
        positions = positions.unsqueeze(0)

    ca_pos = positions[0, start:end, 1, :]
    diff = ca_pos.unsqueeze(1) - ca_pos.unsqueeze(0)
    distances = torch.sqrt((diff ** 2).sum(-1) + 1e-8)

    # Same upper-triangle trick: count each residue pair within the region exactly once.
    region_len = distances.shape[0]
    mask = torch.triu(torch.ones(region_len, region_len, device=distances.device), diagonal=1)
    n_pairs = mask.sum()
    return (distances * mask).sum() / (n_pairs + 1e-8)  # +1e-8 also guards region_len<2 (n_pairs=0)


# ============================================================================
# TRUNK OUTPUT INTERCEPTION (for gradient computation)
# ============================================================================
class TrunkOutputs:
    """Container for trunk outputs before structure module."""
    def __init__(self, s_s, s_z, s_s_proj, s_z_proj, aa, position_ids, mask):
        self.s_s = s_s
        self.s_z = s_z
        self.s_s_proj = s_s_proj
        self.s_z_proj = s_z_proj
        self.aa = aa
        self.position_ids = position_ids
        self.mask = mask


def get_trunk_outputs(model, tokenizer, device, sequence: str, num_recycles: int = 0) -> TrunkOutputs:
    """
    Run ESMFold up to (but not including) the structure module.
    Returns the intermediate representations that would be fed to the structure module.
    """
    # The entire trunk forward pass (ESM-2 encoder, all evoformer-like blocks, and every recycle
    # iteration) runs under torch.no_grad(), so no autograd graph is built for any of it. This is
    # intentionally a fixed feature-extraction pass; gradients (in compute_scale_gradient below)
    # are only introduced later via a fresh scale parameter applied inside the structure module,
    # which keeps the eventual backward pass cheap (it never has to touch the trunk or ESM-2).
    with torch.no_grad():
        inputs = tokenizer(sequence, return_tensors='pt', add_special_tokens=False).to(device)
        input_ids = inputs['input_ids']
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

        cfg = model.config.esmfold_config
        aa = input_ids
        B, L = aa.shape

        esmaa = model.af2_idx_to_esm_idx(aa, attention_mask)  # remap AF2 vocab ids -> ESM-2 vocab ids
        esm_s = model.compute_language_model_representations(esmaa)  # runs full 36-layer ESM-2 encoder
        esm_s = esm_s.to(model.esm_s_combine.dtype)  # match dtype of the learned per-layer mixing weights below

        if cfg.esm_ablate_sequence:
            esm_s = esm_s * 0  # config-gated ablation switch (unused here, inherited from base model forward)

        esm_s = esm_s.detach()  # no-op under no_grad(); kept to mirror the original model code exactly

        # esm_s_combine: one learned scalar weight per ESM-2 layer; softmax -> convex combination
        # across layers, then matmul collapses the per-layer axis into a single per-residue vector.
        esm_s = (model.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
        s_s_0 = model.esm_s_mlp(esm_s)  # project combined ESM-2 features down to the trunk's working dim (c_s)
        s_z_0 = s_s_0.new_zeros(B, L, L, cfg.trunk.pairwise_state_dim)  # pairwise state starts at zero

        if model.config.esmfold_config.embed_aa:
            s_s_0 = s_s_0 + model.embedding(aa)  # optionally add a raw learned per-amino-acid identity embedding

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
            # .detach() calls here are inert given the outer no_grad(), but mirror the real
            # EsmFoldingTrunk.forward (which only wraps non-final recycle iterations in no_grad
            # and needs an explicit detach between iterations). Divergence from the real model:
            # this helper forces EVERY iteration, including the last, through no_grad, since no
            # trunk-level gradients are needed for this experiment.
            recycle_s = trunk.recycle_s_norm(recycle_s.detach())
            recycle_z = trunk.recycle_z_norm(recycle_z.detach())
            # Embed the previous iteration's predicted CA-CA distance histogram as an additive
            # pairwise feature - this is how coordinates from the last recycle feed back into z.
            recycle_z = recycle_z + trunk.recycle_disto(recycle_bins.detach())

            s_s, s_z = trunk_iter(s_s_0 + recycle_s, s_z_0 + recycle_z, position_ids, attention_mask)

            if recycle_idx < no_recycles - 1:
                # Calls the ORIGINAL (unscaled) structure module purely to get positions for the
                # next iteration's recycling distogram; unrelated to the scaling experiment below.
                # Skipped on the final iteration since there is no next iteration to feed.
                structure = trunk.structure_module(
                    {"single": trunk.trunk2sm_s(s_s), "pair": trunk.trunk2sm_z(s_z)},
                    aa, attention_mask.float(),
                )
                recycle_s = s_s
                recycle_z = s_z
                # positions[-1]: final IPA block's coords. [:, :, :3] keeps only N, CA, C (atom14
                # indices 0-2), needed to infer CB and bin distances. 3.375/21.375 A are
                # AlphaFold's standard distogram bin-range bounds; trunk.recycle_bins (=15) is
                # the number of bins.
                recycle_bins = trunk.distogram(
                    structure["positions"][-1][:, :, :3],
                    3.375, 21.375, trunk.recycle_bins,
                )

        # These are the actual tensors the real model feeds into the structure module, and the
        # ones the scaling/gradient experiments below manipulate.
        s_s_proj = trunk.trunk2sm_s(s_s)
        s_z_proj = trunk.trunk2sm_z(s_z)

        # Detach everything before returning - redundant given the outer no_grad() (nothing here
        # has a grad_fn anyway), but makes explicit that the returned tensors are plain constants
        # with no ties back into this trunk computation.
        return TrunkOutputs(
            s_s=s_s.detach(), s_z=s_z.detach(),
            s_s_proj=s_s_proj.detach(), s_z_proj=s_z_proj.detach(),
            aa=aa.detach(), position_ids=position_ids.detach(),
            mask=attention_mask.float().detach(),
        )


# ============================================================================
# GRADIENT COMPUTATION
# ============================================================================
def _run_structure_module_with_scale(
    structure_module, s_normed, z_normed, scale_param, aa, mask,
    scale_target='z',
):
    """
    Run structure module forward with a differentiable scale on z or s.
    Scale is applied AFTER layer norm to avoid normalization undoing the effect.
    """
    from transformers.models.esm.modeling_esmfold import dict_multimap

    # Scale exactly one of z/s (selected by scale_target) and leave the other at its normed,
    # unscaled value - keeps the two perturbations orthogonal so their effects are separable.
    if scale_target == 'z':
        z_input = z_normed * scale_param
        s_input = s_normed
    else:
        z_input = z_normed
        s_input = s_normed * scale_param

    s_initial = s_input  # angle_resnet needs whichever s was actually used going forward (scaled or not)
    s_current = structure_module.linear_in(s_input)

    # Identity rotation+translation frame per residue - the starting backbone frame before any
    # IPA block has run.
    rigids = Rigid.identity(s_current.shape[:-1], s_current.dtype, s_current.device,
                           structure_module.training, fmt="quat")
    outputs = []
    # Re-implementation of EsmFoldStructureModule.forward's block loop (can't call it directly -
    # it has no hook for injecting a scale factor). z_input/s_input (one of which carries the
    # scale_param perturbation) feed into IPA each of the `num_blocks` iterations; only the LAST
    # block's positions are used below (positions[-1]).
    for i in range(structure_module.config.num_blocks):
        s_current = s_current + structure_module.ipa(s_current, z_input, rigids, mask)
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
        # iterations, while translation gradients still flow normally. A second, separate
        # gradient-truncation point inside the structure module itself (distinct from the
        # trunk-level no_grad() truncation done earlier in get_trunk_outputs).
        rigids = rigids.stop_rot_gradient()

    # Stack the per-block dicts into a single dict of tensors, each with a new leading
    # num_blocks dimension (see dict_multimap in modeling_esmfold.py).
    outputs = dict_multimap(torch.stack, outputs)
    return outputs["positions"][-1]  # final block's positions: [B, N, 14, 3]


def _compute_metric(positions, metric, hp_start=None, hp_end=None):
    """Compute a differentiable structural metric from positions."""
    if metric == 'mean_ca_dist':
        return compute_mean_ca_distance_differentiable(positions)
    elif metric == 'full_rg':
        return compute_radius_of_gyration_differentiable(positions, 0, positions.shape[1])
    elif metric == 'hairpin_ca_dist' and hp_start is not None and hp_end is not None:
        return compute_local_ca_distance_differentiable(positions, hp_start, hp_end)
    elif metric == 'hairpin_rg' and hp_start is not None and hp_end is not None:
        return compute_radius_of_gyration_differentiable(positions, hp_start, hp_end)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_scale_gradient(
    model,
    trunk_outputs: TrunkOutputs,
    scale_value: float,
    metric: str,
    scale_target: str = 'z',
    hp_start: Optional[int] = None,
    hp_end: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute the gradient of a structural metric w.r.t. a scale parameter.

    Args:
        scale_target: 'z' or 's'
    """
    device = trunk_outputs.s_s_proj.device
    dtype = trunk_outputs.s_z_proj.dtype

    # .clone() a fresh copy of the (already-detached, no-grad) cached trunk tensors:
    # trunk_outputs may be reused across many scale values/metrics, and cloning avoids autograd
    # in-place/version issues from starting multiple independent graphs off the same storage.
    s = trunk_outputs.s_s_proj.clone()
    z = trunk_outputs.s_z_proj.clone()
    aa = trunk_outputs.aa
    mask = trunk_outputs.mask

    # The only leaf tensor with requires_grad=True here. Since s/z/aa are plain constants,
    # autograd only builds a graph along the path flowing through scale_param - i.e. just the
    # structure module ops in _run_structure_module_with_scale, not the trunk that produced s/z.
    scale_param = torch.tensor(scale_value, dtype=dtype, device=device, requires_grad=True)

    sm = model.trunk.structure_module

    if mask is None:
        mask = s.new_ones(s.shape[:-1])

    s_normed = sm.layer_norm_s(s)
    z_normed = sm.layer_norm_z(z)

    positions = _run_structure_module_with_scale(
        sm, s_normed, z_normed, scale_param, aa, mask, scale_target=scale_target,
    )

    metric_value = _compute_metric(positions, metric, hp_start, hp_end)
    # Since scale_param is the only requires_grad leaf feeding into metric_value, this is a cheap
    # backward pass through just the structure module, not the trunk or ESM-2 encoder.
    metric_value.backward()

    gradient = scale_param.grad  # d(metric_value) / d(scale_param), accumulated into the leaf's .grad
    gradient_value = gradient.item() if gradient is not None else 0.0

    return {
        'metric': metric,
        f'{scale_target}_scale': scale_value,
        'metric_value': metric_value.item(),
        'gradient': gradient_value,
    }


def compute_both_gradients(
    model,
    trunk_outputs: TrunkOutputs,
    metrics: List[str],
    hp_start: Optional[int] = None,
    hp_end: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute gradients for both z and s at the normal operating point (scale=1.0)."""
    z_results = []
    s_results = []

    # scale=1.0 is the "operating point": the model's actual, unperturbed behavior. Local
    # sensitivity (gradient) there is the most directly meaningful reading of "how much does
    # geometry change per unit change in z/s scale" for the model as it's normally used.
    for metric in metrics:
        try:
            z_results.append(compute_scale_gradient(
                model, trunk_outputs, 1.0, metric, 'z', hp_start, hp_end))
        except Exception as e:
            # Substitute a NaN row instead of aborting the whole loop, so one failing metric
            # doesn't lose results for the others; downstream pandas aggregations skip NaNs.
            print(f"Z gradient error: metric={metric}: {e}")
            z_results.append({'metric': metric, 'z_scale': 1.0,
                              'metric_value': float('nan'), 'gradient': float('nan')})

        try:
            s_results.append(compute_scale_gradient(
                model, trunk_outputs, 1.0, metric, 's', hp_start, hp_end))
        except Exception as e:
            print(f"S gradient error: metric={metric}: {e}")
            s_results.append({'metric': metric, 's_scale': 1.0,
                              'metric_value': float('nan'), 'gradient': float('nan')})

    torch.cuda.empty_cache()

    return pd.DataFrame(z_results), pd.DataFrame(s_results)


# ============================================================================
# MODIFIED STRUCTURE MODULE WITH Z/S SCALING
# ============================================================================
def create_scaled_sm_forward(
    s_scale: float = 1.0,
    z_scale: float = 1.0,
):
    """
    Create a structure module forward that scales s and/or z before processing.
    
    Args:
        s_scale: Scale factor for single representations
        z_scale: Scale factor for pair representations
    """
    from transformers.models.esm.modeling_esmfold import dict_multimap
    
    # This closure captures s_scale/z_scale from the enclosing function's arguments - a factory
    # pattern that lets us parametrize a monkey-patched replacement method without a class.
    # Unlike compute_scale_gradient above (which hand-runs the structure module under a fresh,
    # tiny autograd graph for gradients), this is used for the discrete-sweep experiment: the
    # whole model is called normally (see run_with_scaling), and this forward is patched in only
    # to inject a fixed scale multiplier - no gradients needed here.
    def modified_forward(self_sm, evoformer_output_dict, aatype, mask=None, _offload_inference=False):
        """Structure module forward pass with s/z scaled by s_scale/z_scale after layer norm."""
        # _offload_inference accepted for signature parity with the real forward, but the
        # _offload_inference=True CPU-offload branch (see real EsmFoldStructureModule.forward)
        # is not reimplemented here - always behaves as if False. Confirmed harmless for how this
        # is actually invoked: run_with_scaling below only calls the top-level model(...), whose
        # own trunk.forward always calls structure_module with the (default) False.
        # Get s and z from evoformer output
        s = evoformer_output_dict["single"]
        z = evoformer_output_dict["pair"]
        
        if mask is None:
            mask = s.new_ones(s.shape[:-1])
        
        # Apply layer norms first (as in original)
        s = self_sm.layer_norm_s(s)
        z = self_sm.layer_norm_z(z)
        
        # SCALE S AND Z HERE - this is the key intervention
        # (applied after layer norm - scaling before it would just get normalized back away)
        s = s * s_scale
        z = z * z_scale
        
        s_initial = s
        s = self_sm.linear_in(s)
        
        rigids = Rigid.identity(s.shape[:-1], s.dtype, s.device, self_sm.training, fmt="quat")
        outputs = []
        
        # Re-implementation of EsmFoldStructureModule.forward's block loop (this whole function
        # replaces structure_module.forward via types.MethodType in run_with_scaling below).
        for i in range(self_sm.config.num_blocks):
            s = s + self_sm.ipa(s, z, rigids, mask)
            s = self_sm.ipa_dropout(s)
            s = self_sm.layer_norm_ipa(s)
            s = self_sm.transition(s)
            
            rigids = rigids.compose_q_update_vec(self_sm.bb_update(s))
            
            backb_to_global = Rigid(
                Rotation(rot_mats=rigids.get_rots().get_rot_mats(), quats=None),
                rigids.get_trans(),
            )
            backb_to_global = backb_to_global.scale_translation(self_sm.config.trans_scale_factor)
            
            unnormalized_angles, angles = self_sm.angle_resnet(s, s_initial)
            all_frames_to_global = self_sm.torsion_angles_to_frames(backb_to_global, angles, aatype)
            pred_xyz = self_sm.frames_and_literature_positions_to_atom14_pos(all_frames_to_global, aatype)
            
            scaled_rigids = rigids.scale_translation(self_sm.config.trans_scale_factor)
            
            preds = {
                "frames": scaled_rigids.to_tensor_7(),
                "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "positions": pred_xyz,
                "states": s,
            }
            outputs.append(preds)
            # stop_rot_gradient(): detaches only the rotation component before the next block
            # (standard AlphaFold/OpenFold stabilization trick). Doesn't matter for gradients
            # here (this path runs under no_grad(), see run_with_scaling), but kept for parity
            # with the real EsmFoldStructureModule.forward.
            rigids = rigids.stop_rot_gradient()
        
        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s  # matches the real forward, which also returns the final "single" state
        return outputs
    
    return modified_forward


# ============================================================================
# EXPERIMENT RUNNERS
# ============================================================================
def run_with_scaling(
    model, tokenizer, device, sequence: str,
    s_scale: float = 1.0,
    z_scale: float = 1.0,
) -> EsmForProteinFoldingOutput:
    """
    Run ESMFold with scaled s and/or z inputs to structure module.
    """
    # Store original forward
    original_sm_forward = model.trunk.structure_module.forward
    
    # Patch with scaled version
    # types.MethodType binds the closure as an instance method on the SAME structure module
    # instance, so `self_sm` inside modified_forward is the real module (with working access to
    # self_sm.ipa, self_sm.layer_norm_s, etc.), exactly as if it were the original bound method.
    model.trunk.structure_module.forward = types.MethodType(
        create_scaled_sm_forward(s_scale=s_scale, z_scale=z_scale),
        model.trunk.structure_module
    )
    
    try:
        # no_grad(): only the resulting geometry is needed here, not gradients - this is the
        # discrete-sweep experiment, distinct from the autograd-based gradient functions above.
        with torch.no_grad():
            inputs = tokenizer(sequence, return_tensors='pt', add_special_tokens=False).to(device)
            outputs = model(**inputs, num_recycles=0)  # num_recycles=0: single trunk pass, keeps runs comparable
    finally:
        # Restore original
        # try/finally guarantees this runs even if the call above raises - since this function is
        # invoked in a loop over many scale values (run_scaling_comparison below), leaving the
        # patch in place after a failure would silently corrupt later calls with a stale scale.
        model.trunk.structure_module.forward = original_sm_forward
    
    return outputs


def run_scaling_comparison(
    model, tokenizer, device, sequence: str,
    hp_start: int, hp_end: int,
    scales: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run scaling experiments for both z and s independently.
    
    Returns:
        z_results: DataFrame with z-scaling results
        s_results: DataFrame with s-scaling results
    """
    z_results = []
    s_results = []
    
    # First, get baseline (scale=1.0 for both)
    print("  Getting baseline...")
    baseline_outputs = run_with_scaling(model, tokenizer, device, sequence, s_scale=1.0, z_scale=1.0)
    # positions[-1]: final structure-module block; [..., 0]: first (only) batch element ->
    # [N_res, 14, 3]. Reused as the reference for every scaled run below.
    baseline_pos = baseline_outputs.positions[-1, 0]
    baseline_contacts = compute_contact_map(baseline_pos)
    
    # Scale z only
    # (s pinned at 1.0; s is scaled on its own further below with z pinned at 1.0 - keeps the two
    # representations' effects orthogonal/separable for a valid ceteris-paribus comparison)
    print("  Scaling z (pair representations)...")
    for scale in tqdm(scales, desc="Z scaling", leave=False):
        outputs = run_with_scaling(model, tokenizer, device, sequence, s_scale=1.0, z_scale=scale)
        final_pos = outputs.positions[-1, 0]
        
        metrics = compute_structure_metrics(
            final_pos, baseline_pos, baseline_contacts, hp_start, hp_end, outputs
        )
        metrics['scale'] = scale
        metrics['scaled_repr'] = 'z (pair)'
        z_results.append(metrics)
        
        # Frees cached (unused) GPU memory back to the driver between runs - not strictly needed
        # per-call, but avoids fragmentation/OOM over the many sequences x scales x cases this
        # experiment loops over (see run_z_vs_s_experiment).
        torch.cuda.empty_cache()
    
    # Scale s only
    print("  Scaling s (single representations)...")
    for scale in tqdm(scales, desc="S scaling", leave=False):
        outputs = run_with_scaling(model, tokenizer, device, sequence, s_scale=scale, z_scale=1.0)
        final_pos = outputs.positions[-1, 0]
        
        metrics = compute_structure_metrics(
            final_pos, baseline_pos, baseline_contacts, hp_start, hp_end, outputs
        )
        metrics['scale'] = scale
        metrics['scaled_repr'] = 's (single)'
        s_results.append(metrics)
        
        torch.cuda.empty_cache()
    
    return pd.DataFrame(z_results), pd.DataFrame(s_results)


def compute_structure_metrics(
    positions: torch.Tensor,
    baseline_positions: torch.Tensor,
    baseline_contacts: torch.Tensor,
    hp_start: int,
    hp_end: int,
    outputs: EsmForProteinFoldingOutput,
) -> Dict[str, float]:
    """Compute various structural metrics."""
    
    # Basic geometry
    hairpin_rg = compute_radius_of_gyration(positions, hp_start, hp_end)
    full_rg = compute_radius_of_gyration(positions, 0, positions.shape[0])
    strand_sep = compute_strand_separation(positions, hp_start, hp_end)
    
    # CA distances
    ca_dists = compute_ca_distances(positions)
    hp_mean_dist = ca_dists[hp_start:hp_end, hp_start:hp_end].mean().item()
    full_mean_dist = ca_dists.mean().item()
    
    # RMSD from baseline (CA only)
    ca_pos = positions[:, 1, :]  # [N, 3]
    baseline_ca = baseline_positions[:, 1, :]
    
    # Simple RMSD (no alignment)
    # (coordinates are compared directly, with no rotation/translation superposition first, so a
    # rigid-body shift between the scaled and baseline structures would itself show up as RMSD -
    # acceptable here since s/z scaling is expected to act directly on absolute coordinates)
    rmsd_all = torch.sqrt(((ca_pos - baseline_ca) ** 2).sum(-1).mean()).item()
    rmsd_hairpin = torch.sqrt(((ca_pos[hp_start:hp_end] - baseline_ca[hp_start:hp_end]) ** 2).sum(-1).mean()).item()
    
    # Contact map comparison
    # precision: of the contacts predicted in this (scaled) structure, fraction also present in
    # baseline. recall: of the contacts present in baseline, fraction recovered here. +1e-8 avoids
    # divide-by-zero if a structure ends up with zero contacts (e.g. fully unfolded by scaling).
    contacts = compute_contact_map(positions)
    contact_precision = ((contacts == 1) & (baseline_contacts == 1)).sum() / (contacts.sum() + 1e-8)
    contact_recall = ((contacts == 1) & (baseline_contacts == 1)).sum() / (baseline_contacts.sum() + 1e-8)
    
    # pLDDT
    mean_plddt = outputs.plddt[0].mean().item()
    hairpin_plddt = outputs.plddt[0, hp_start:hp_end].mean().item()
    
    return {
        'hairpin_rg': hairpin_rg,
        'full_rg': full_rg,
        'strand_sep': strand_sep,
        'hairpin_mean_ca_dist': hp_mean_dist,
        'full_mean_ca_dist': full_mean_dist,
        'rmsd_all': rmsd_all,
        'rmsd_hairpin': rmsd_hairpin,
        'contact_precision': contact_precision.item(),
        'contact_recall': contact_recall.item(),
        'mean_plddt': mean_plddt,
        'hairpin_plddt': hairpin_plddt,
    }


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_z_vs_s_comparison(
    z_results: pd.DataFrame,
    s_results: pd.DataFrame,
    output_dir: str,
    case_name: str,
):
    """Create comparison plots for z vs s scaling effects."""
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    
    z_scales = z_results['scale'].values
    s_scales = s_results['scale'].values
    
    # Row 1: Geometry metrics
    # Plot 1: Hairpin RG
    ax = axes[0, 0]
    ax.plot(z_scales, z_results['hairpin_rg'], 'b-o', linewidth=2, markersize=6, label='z (pair)')
    ax.plot(s_scales, s_results['hairpin_rg'], 'r-s', linewidth=2, markersize=6, label='s (single)')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Scale Factor')
    ax.set_ylabel('Radius of Gyration (Å)')
    ax.set_title('Hairpin RG')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Full protein RG
    ax = axes[0, 1]
    ax.plot(z_scales, z_results['full_rg'], 'b-o', linewidth=2, markersize=6, label='z (pair)')
    ax.plot(s_scales, s_results['full_rg'], 'r-s', linewidth=2, markersize=6, label='s (single)')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Scale Factor')
    ax.set_ylabel('Radius of Gyration (Å)')
    ax.set_title('Full Protein RG')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Strand separation
    ax = axes[0, 2]
    ax.plot(z_scales, z_results['strand_sep'], 'b-o', linewidth=2, markersize=6, label='z (pair)')
    ax.plot(s_scales, s_results['strand_sep'], 'r-s', linewidth=2, markersize=6, label='s (single)')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Scale Factor')
    ax.set_ylabel('Strand Separation (Å)')
    ax.set_title('Hairpin Strand Separation')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Row 2: Distance/RMSD metrics
    # Plot 4: Mean CA distance (hairpin and full)
    ax = axes[1, 0]
    ax.plot(z_scales, z_results['hairpin_mean_ca_dist'], 'b-o', linewidth=2, markersize=6, label='z hairpin')
    ax.plot(z_scales, z_results['full_mean_ca_dist'], 'b--^', linewidth=2, markersize=6, label='z full')
    ax.plot(s_scales, s_results['hairpin_mean_ca_dist'], 'r-s', linewidth=2, markersize=6, label='s hairpin')
    ax.plot(s_scales, s_results['full_mean_ca_dist'], 'r--v', linewidth=2, markersize=6, label='s full')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Scale Factor')
    ax.set_ylabel('Mean CA Distance (Å)')
    ax.set_title('Mean CA Distance')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Plot 5: RMSD from baseline (full)
    ax = axes[1, 1]
    ax.plot(z_scales, z_results['rmsd_all'], 'b-o', linewidth=2, markersize=6, label='z (pair)')
    ax.plot(s_scales, s_results['rmsd_all'], 'r-s', linewidth=2, markersize=6, label='s (single)')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Scale Factor')
    ax.set_ylabel('RMSD from Baseline (Å)')
    ax.set_title('Full Protein RMSD')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 6: RMSD from baseline (hairpin only)
    ax = axes[1, 2]
    ax.plot(z_scales, z_results['rmsd_hairpin'], 'b-o', linewidth=2, markersize=6, label='z (pair)')
    ax.plot(s_scales, s_results['rmsd_hairpin'], 'r-s', linewidth=2, markersize=6, label='s (single)')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Scale Factor')
    ax.set_ylabel('RMSD from Baseline (Å)')
    ax.set_title('Hairpin RMSD')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Row 3: Quality metrics
    # Plot 7: pLDDT
    ax = axes[2, 0]
    ax.plot(z_scales, z_results['mean_plddt'], 'b-o', linewidth=2, markersize=6, label='z full')
    ax.plot(z_scales, z_results['hairpin_plddt'], 'b--^', linewidth=2, markersize=6, label='z hairpin')
    ax.plot(s_scales, s_results['mean_plddt'], 'r-s', linewidth=2, markersize=6, label='s full')
    ax.plot(s_scales, s_results['hairpin_plddt'], 'r--v', linewidth=2, markersize=6, label='s hairpin')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Scale Factor')
    ax.set_ylabel('pLDDT')
    ax.set_title('Confidence Scores')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Plot 8: Contact map metrics
    ax = axes[2, 1]
    ax.plot(z_scales, z_results['contact_precision'], 'b-o', linewidth=2, markersize=6, label='z precision')
    ax.plot(z_scales, z_results['contact_recall'], 'b--^', linewidth=2, markersize=6, label='z recall')
    ax.plot(s_scales, s_results['contact_precision'], 'r-s', linewidth=2, markersize=6, label='s precision')
    ax.plot(s_scales, s_results['contact_recall'], 'r--v', linewidth=2, markersize=6, label='s recall')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Scale Factor')
    ax.set_ylabel('Metric')
    ax.set_title('Contact Map Quality')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Plot 9: Normalized sensitivity comparison
    ax = axes[2, 2]
    
    # Compute sensitivity as (max - min) / baseline for key metrics
    def compute_sensitivity(results, metric):
        """Relative sensitivity: (max-min across scales) / value at the scale=1.0 baseline."""
        baseline_val = results.loc[results['scale'] == 1.0, metric].values[0]
        if baseline_val == 0:
            return 0
        range_val = results[metric].max() - results[metric].min()
        return range_val / baseline_val
    
    metrics_to_compare = ['hairpin_rg', 'full_rg', 'strand_sep', 'hairpin_mean_ca_dist']
    metric_labels = ['HP RG', 'Full RG', 'Strand Sep', 'HP CA Dist']
    
    z_sensitivities = [compute_sensitivity(z_results, m) for m in metrics_to_compare]
    s_sensitivities = [compute_sensitivity(s_results, m) for m in metrics_to_compare]
    
    # Standard grouped-bar-chart layout: one tick per metric at integer position x, with the
    # z/s bars offset by +/- half the bar width so they sit side by side without overlapping.
    x = np.arange(len(metric_labels))
    width = 0.35
    
    ax.bar(x - width/2, z_sensitivities, width, label='z (pair)', color='steelblue')
    ax.bar(x + width/2, s_sensitivities, width, label='s (single)', color='indianred')
    ax.set_ylabel('Sensitivity (range/baseline)')
    ax.set_title('Metric Sensitivity to Scaling')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.suptitle(f'Z vs S Scaling Comparison: {case_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'{case_name}_z_vs_s_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_summary_across_cases(
    all_z_results: pd.DataFrame,
    all_s_results: pd.DataFrame,
    output_dir: str,
):
    """Plot summary statistics across all cases."""
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    
    # Select only numeric columns for aggregation
    # all_z_results/all_s_results also carry non-numeric columns (case_name, scaled_repr, ...)
    # which would break/warn under a numeric .agg(['mean', 'std']) - restrict to numeric_cols first.
    numeric_cols = ['scale', 'hairpin_rg', 'full_rg', 'strand_sep', 'hairpin_mean_ca_dist',
                    'full_mean_ca_dist', 'rmsd_all', 'rmsd_hairpin', 'contact_precision', 
                    'contact_recall', 'mean_plddt', 'hairpin_plddt']
    
    z_numeric = all_z_results[numeric_cols]
    s_numeric = all_s_results[numeric_cols]
    
    # Group by scale
    # .agg(['mean', 'std']) on a DataFrame produces MultiIndex columns: top level = original
    # column name (e.g. 'hairpin_rg'), second level = stat name ('mean'/'std'). Indexed below via
    # z_grouped[(metric, 'mean')] etc.
    z_grouped = z_numeric.groupby('scale').agg(['mean', 'std'])
    s_grouped = s_numeric.groupby('scale').agg(['mean', 'std'])
    scales = z_grouped.index.values
    
    metrics = [
        ('hairpin_rg', 'Hairpin RG (Å)', axes[0, 0]),
        ('full_rg', 'Full Protein RG (Å)', axes[0, 1]),
        ('strand_sep', 'Strand Separation (Å)', axes[0, 2]),
        ('hairpin_mean_ca_dist', 'Hairpin Mean CA Dist (Å)', axes[0, 3]),
        ('full_mean_ca_dist', 'Full Mean CA Dist (Å)', axes[1, 0]),
        ('rmsd_all', 'RMSD from Baseline (Å)', axes[1, 1]),
        ('mean_plddt', 'pLDDT', axes[1, 2]),
        ('rmsd_hairpin', 'Hairpin RMSD (Å)', axes[1, 3]),
    ]
    
    for metric, ylabel, ax in metrics:
        z_mean = z_grouped[(metric, 'mean')].values
        z_std = z_grouped[(metric, 'std')].values
        s_mean = s_grouped[(metric, 'mean')].values
        s_std = s_grouped[(metric, 'std')].values
        
        ax.errorbar(scales, z_mean, yerr=z_std, fmt='b-o', capsize=3, linewidth=2, 
                   markersize=6, label='z (pair)')
        ax.errorbar(scales, s_mean, yerr=s_std, fmt='r-s', capsize=3, linewidth=2,
                   markersize=6, label='s (single)')
        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Scale Factor')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.suptitle('Z vs S Scaling: Summary Across All Cases', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'z_vs_s_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_effect_size_comparison(
    all_z_results: pd.DataFrame,
    all_s_results: pd.DataFrame,
    output_dir: str,
):
    """
    Create a summary figure showing the relative effect sizes of z vs s scaling.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Compute effect sizes (deviation from baseline at each scale)
    metrics = ['hairpin_rg', 'full_rg', 'strand_sep', 'rmsd_all', 'hairpin_mean_ca_dist', 'full_mean_ca_dist']
    
    # Get baseline values (scale = 1.0)
    z_baseline = all_z_results[all_z_results['scale'] == 1.0][metrics].mean()
    s_baseline = all_s_results[all_s_results['scale'] == 1.0][metrics].mean()
    
    # Left plot: Effect size at extreme scales (0.0 and 2.0)
    # 0.0 = representation fully zeroed out, 2.0 = doubled; the two most extreme perturbation
    # conditions in `scales`, used here as a quick "how much does this matter" summary.
    ax = axes[0]
    
    extreme_scales = [0.0, 2.0]
    bar_data = {'z': [], 's': []}
    
    for scale in extreme_scales:
        z_at_scale = all_z_results[all_z_results['scale'] == scale]
        s_at_scale = all_s_results[all_s_results['scale'] == scale]
        
        for metric in metrics:
            z_effect = abs(z_at_scale[metric].mean() - z_baseline[metric]) / (z_baseline[metric] + 1e-8)
            s_effect = abs(s_at_scale[metric].mean() - s_baseline[metric]) / (s_baseline[metric] + 1e-8)
            bar_data['z'].append(z_effect)
            bar_data['s'].append(s_effect)
    
    x = np.arange(len(metrics) * len(extreme_scales))
    width = 0.35
    
    ax.bar(x - width/2, bar_data['z'], width, label='z (pair)', color='steelblue')
    ax.bar(x + width/2, bar_data['s'], width, label='s (single)', color='indianred')
    
    # Label order here (scale outer loop, metric inner loop) must match the order bar_data was
    # filled above (also scale-outer/metric-inner) - `metrics` and this label list are also kept
    # in the same 1:1 order (hairpin_rg->HP_RG, full_rg->Full_RG, ...).
    labels = [f'{m}\n(s={s})' for s in extreme_scales for m in ['HP_RG', 'Full_RG', 'Strand', 'RMSD', 'HP_CA', 'Full_CA']]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Relative Effect Size (|change|/baseline)')
    ax.set_title('Effect Magnitude at Extreme Scales')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Right plot: Summary bar chart of total sensitivity
    ax = axes[1]
    
    metric_labels = ['Hairpin RG', 'Full RG', 'Strand Sep', 'RMSD', 'HP CA Dist', 'Full CA Dist']
    
    z_total_sensitivity = []
    s_total_sensitivity = []
    
    for metric in metrics:
        # Total range normalized by baseline
        z_range = (all_z_results.groupby('scale')[metric].mean().max() - 
                   all_z_results.groupby('scale')[metric].mean().min())
        z_base = z_baseline[metric]
        z_total_sensitivity.append(z_range / (z_base + 1e-8))
        
        s_range = (all_s_results.groupby('scale')[metric].mean().max() - 
                   all_s_results.groupby('scale')[metric].mean().min())
        s_base = s_baseline[metric]
        s_total_sensitivity.append(s_range / (s_base + 1e-8))
    
    x = np.arange(len(metrics))
    
    ax.bar(x - width/2, z_total_sensitivity, width, label='z (pair)', color='steelblue')
    ax.bar(x + width/2, s_total_sensitivity, width, label='s (single)', color='indianred')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax.set_ylabel('Total Sensitivity (range/baseline)')
    ax.set_title('Overall Metric Sensitivity')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Add ratio annotations
    for i, (z_sens, s_sens) in enumerate(zip(z_total_sensitivity, s_total_sensitivity)):
        ratio = z_sens / (s_sens + 1e-8)
        ax.annotate(f'{ratio:.1f}x', xy=(i, max(z_sens, s_sens) + 0.05), 
                   ha='center', fontsize=8, color='green' if ratio > 1 else 'purple')
    
    plt.suptitle('Z vs S: Which Representation Matters More?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'z_vs_s_effect_sizes.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_running_averages(
    all_z_results: pd.DataFrame,
    all_s_results: pd.DataFrame,
    output_dir: str,
    plot_every_n: int = 10,
):
    """
    Plot running averages of metrics, saving a plot every N cases.
    Shows how the average results stabilize as more cases are added.
    """
    # Get unique case indices in order
    case_indices = sorted(all_z_results['case_idx'].unique())
    n_total_cases = len(case_indices)
    
    if n_total_cases < plot_every_n:
        print(f"Not enough cases ({n_total_cases}) to plot running averages every {plot_every_n} cases")
        return
    
    metrics = ['hairpin_rg', 'full_rg', 'strand_sep', 'hairpin_mean_ca_dist', 'full_mean_ca_dist', 'rmsd_all']
    metric_labels = ['Hairpin RG (Å)', 'Full RG (Å)', 'Strand Sep (Å)', 'HP CA Dist (Å)', 'Full CA Dist (Å)', 'RMSD (Å)']
    
    # Checkpoints to plot
    # e.g. plot_every_n=10 -> [10, 20, 30, ...]; the final append ensures the last, "all cases"
    # checkpoint is always included even when n_total_cases isn't an exact multiple of
    # plot_every_n, so there's always a plot reflecting the complete/converged result.
    checkpoints = list(range(plot_every_n, n_total_cases + 1, plot_every_n))
    if n_total_cases not in checkpoints:
        checkpoints.append(n_total_cases)
    
    # For each checkpoint, compute and plot average results
    # (re-aggregates over a growing prefix of cases - an expanding window - so the saved plots
    # show how the mean/std stabilize ("converge") as more cases are included)
    for n_cases in checkpoints:
        cases_to_include = case_indices[:n_cases]
        
        z_subset = all_z_results[all_z_results['case_idx'].isin(cases_to_include)]
        s_subset = all_s_results[all_s_results['case_idx'].isin(cases_to_include)]
        
        # Select only numeric columns for aggregation
        numeric_cols = ['scale', 'hairpin_rg', 'full_rg', 'strand_sep', 'hairpin_mean_ca_dist',
                        'full_mean_ca_dist', 'rmsd_all', 'rmsd_hairpin', 'contact_precision', 
                        'contact_recall', 'mean_plddt', 'hairpin_plddt']
        
        z_numeric = z_subset[numeric_cols]
        s_numeric = s_subset[numeric_cols]
        
        z_grouped = z_numeric.groupby('scale').agg(['mean', 'std'])
        s_grouped = s_numeric.groupby('scale').agg(['mean', 'std'])
        scales = z_grouped.index.values
        
        # Create plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx]
            
            z_mean = z_grouped[(metric, 'mean')].values
            z_std = z_grouped[(metric, 'std')].values
            s_mean = s_grouped[(metric, 'mean')].values
            s_std = s_grouped[(metric, 'std')].values
            
            ax.errorbar(scales, z_mean, yerr=z_std, fmt='b-o', capsize=3, linewidth=2, 
                       markersize=6, label='z (pair)')
            ax.errorbar(scales, s_mean, yerr=s_std, fmt='r-s', capsize=3, linewidth=2,
                       markersize=6, label='s (single)')
            ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Scale Factor')
            ax.set_ylabel(label)
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.suptitle(f'Z vs S Scaling: Average over {n_cases} Cases', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'z_vs_s_avg_{n_cases:03d}_cases.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    # Also create a summary plot showing how metrics evolve with number of cases
    plot_metric_convergence(all_z_results, all_s_results, output_dir, checkpoints)


def plot_metric_convergence(
    all_z_results: pd.DataFrame,
    all_s_results: pd.DataFrame,
    output_dir: str,
    checkpoints: List[int],
):
    """
    Plot how the Z/S effect ratio converges as more cases are added.
    """
    # Tests whether the "which representation matters more" conclusion is robust across sample
    # size, or just an artifact of small n (as opposed to plot_running_averages, which tracks raw
    # metric values rather than the summary z/s ratio).
    case_indices = sorted(all_z_results['case_idx'].unique())
    
    metrics = ['hairpin_rg', 'full_rg', 'strand_sep', 'hairpin_mean_ca_dist', 'full_mean_ca_dist']
    metric_labels = ['Hairpin RG', 'Full RG', 'Strand Sep', 'HP CA Dist', 'Full CA Dist']
    
    # Compute Z/S ratio at each checkpoint
    ratios_per_checkpoint = {m: [] for m in metrics}
    
    for n_cases in checkpoints:
        cases_to_include = case_indices[:n_cases]
        
        z_subset = all_z_results[all_z_results['case_idx'].isin(cases_to_include)]
        s_subset = all_s_results[all_s_results['case_idx'].isin(cases_to_include)]
        
        for metric in metrics:
            z_range = z_subset.groupby('scale')[metric].mean().max() - z_subset.groupby('scale')[metric].mean().min()
            s_range = s_subset.groupby('scale')[metric].mean().max() - s_subset.groupby('scale')[metric].mean().min()
            ratio = z_range / (s_range + 1e-8)
            ratios_per_checkpoint[metric].append(ratio)
    
    # Plot convergence
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Z/S ratio convergence
    ax = axes[0]
    for metric, label in zip(metrics, metric_labels):
        ax.plot(checkpoints, ratios_per_checkpoint[metric], '-o', linewidth=2, markersize=6, label=label)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Equal effect')
    ax.set_xlabel('Number of Cases')
    ax.set_ylabel('Z/S Effect Ratio')
    ax.set_title('Effect Ratio Convergence')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Right plot: Final bar chart comparison
    ax = axes[1]
    final_ratios = [ratios_per_checkpoint[m][-1] for m in metrics]
    colors = ['steelblue' if r > 1 else 'indianred' for r in final_ratios]
    bars = ax.bar(metric_labels, final_ratios, color=colors)  # categorical x-axis (string labels)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Z/S Effect Ratio')
    ax.set_title(f'Final Z/S Ratios ({checkpoints[-1]} Cases)')
    # NOTE: possible bug -- set_xticklabels() is called here without a preceding set_xticks()
    # call, unlike every other bar chart in this file (which always pairs set_xticks + 
    # set_xticklabels). It happens to work because ax.bar() with string categories pre-creates
    # one tick per bar, but current matplotlib emits: "UserWarning: set_ticklabels() should only
    # be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator."
    # (verified empirically). Harmless today, but fragile if bar() ever stops auto-fixing ticks.
    ax.set_xticklabels(metric_labels, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, ratio in zip(bars, final_ratios):
        ax.annotate(f'{ratio:.2f}x', 
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=9)
    
    ax.grid(alpha=0.3, axis='y')
    
    plt.suptitle('Z vs S: Effect Ratio Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'z_vs_s_convergence.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def run_z_vs_s_experiment(
    parquet_path: str,
    n_cases: int,
    output_dir: str,
    device: Optional[str] = None,
    scales: Optional[List[float]] = None,
    plot_every_n: int = 10,
):
    """Run the full z vs s scaling experiment with gradient analysis."""

    os.makedirs(output_dir, exist_ok=True)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if scales is None:
        scales = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    gradient_metrics = ['mean_ca_dist', 'full_rg', 'hairpin_ca_dist', 'hairpin_rg']

    # Load model
    model, tokenizer = load_esmfold(device)

    # Load data
    print(f"\nLoading data from {parquet_path}...")
    if parquet_path.endswith('.parquet'):
        df = pd.read_parquet(parquet_path)
    else:
        df = pd.read_csv(parquet_path)
    print(f"Loaded {len(df)} rows")

    # Select cases
    cases = df.head(n_cases)

    all_z_results = []
    all_s_results = []
    all_z_grad_results = []
    all_s_grad_results = []

    for idx, row in tqdm(cases.iterrows(), total=len(cases), desc="Analyzing cases"):
        case_name = f"case_{idx}"

        target_seq = row['target_sequence']
        hp_start = int(row['target_patch_start'])
        hp_end = int(row['target_patch_end'])

        print(f"\n{'='*60}")
        print(f"Case {idx}: {row.get('target_name', 'Unknown')}")
        print(f"Sequence length: {len(target_seq)}")
        print(f"Hairpin region: {hp_start}-{hp_end} ({hp_end - hp_start} residues)")
        print(f"{'='*60}")

        # Run scaling comparison
        # Discrete sweep (monkey-patched forward, no gradients) - see run_scaling_comparison.
        z_results, s_results = run_scaling_comparison(
            model, tokenizer, device, target_seq,
            hp_start, hp_end,
            scales=scales,
        )

        # Add case info
        z_results['case_idx'] = idx
        z_results['case_name'] = row.get('target_name', f'case_{idx}')
        s_results['case_idx'] = idx
        s_results['case_name'] = row.get('target_name', f'case_{idx}')

        all_z_results.append(z_results)
        all_s_results.append(s_results)

        # Plot individual case
        plot_z_vs_s_comparison(z_results, s_results, output_dir, case_name)

        # Compute gradients
        # Autograd-based analysis at the operating point (scale=1.0) - see compute_scale_gradient.
        # get_trunk_outputs runs the (expensive) trunk pass once per case; compute_both_gradients
        # then reuses it for every metric via cheap structure-module-only re-runs.
        print("  Computing gradients...")
        trunk_outputs = get_trunk_outputs(model, tokenizer, device, target_seq, num_recycles=0)

        z_grad_results, s_grad_results = compute_both_gradients(
            model, trunk_outputs, gradient_metrics,
            hp_start=hp_start, hp_end=hp_end,
        )

        z_grad_results['case_idx'] = idx
        z_grad_results['case_name'] = row.get('target_name', f'case_{idx}')
        s_grad_results['case_idx'] = idx
        s_grad_results['case_name'] = row.get('target_name', f'case_{idx}')

        all_z_grad_results.append(z_grad_results)
        all_s_grad_results.append(s_grad_results)

        # Explicit cleanup: trunk_outputs holds several large per-residue/pairwise tensors, and
        # this loop may run over hundreds of cases (see DEFAULT_N_CASES) - del + empty_cache
        # prevents cumulative GPU memory growth across iterations.
        del trunk_outputs
        torch.cuda.empty_cache()

    # Combine results
    combined_z = pd.concat(all_z_results, ignore_index=True)
    combined_s = pd.concat(all_s_results, ignore_index=True)

    # Save scaling results
    combined_z.to_csv(os.path.join(output_dir, 'z_scaling_results.csv'), index=False)
    combined_s.to_csv(os.path.join(output_dir, 's_scaling_results.csv'), index=False)

    # Save gradient results
    combined_z_grad = pd.concat(all_z_grad_results, ignore_index=True)
    combined_s_grad = pd.concat(all_s_grad_results, ignore_index=True)
    combined_z_grad.to_csv(os.path.join(output_dir, 'z_gradients.csv'), index=False)
    combined_s_grad.to_csv(os.path.join(output_dir, 's_gradients.csv'), index=False)

    print(f"\nSaved results to {output_dir}")

    # Summary plots (scaling only)
    if len(all_z_results) > 1:
        plot_summary_across_cases(combined_z, combined_s, output_dir)

    plot_effect_size_comparison(combined_z, combined_s, output_dir)

    # Plot running averages every N cases
    plot_running_averages(combined_z, combined_s, output_dir, plot_every_n=plot_every_n)

    # Print scaling summary
    print("\n" + "="*60)
    print("SUMMARY: Z vs S Scaling Effects")
    print("="*60)

    for metric in ['hairpin_rg', 'full_rg', 'strand_sep', 'rmsd_all', 'hairpin_mean_ca_dist', 'full_mean_ca_dist']:
        z_range = combined_z.groupby('scale')[metric].mean().max() - combined_z.groupby('scale')[metric].mean().min()
        s_range = combined_s.groupby('scale')[metric].mean().max() - combined_s.groupby('scale')[metric].mean().min()

        print(f"\n{metric}:")
        print(f"  Z scaling range: {z_range:.2f}")
        print(f"  S scaling range: {s_range:.2f}")
        print(f"  Z/S ratio: {z_range/(s_range+1e-8):.2f}x")

    # Print gradient summary
    print("\n" + "="*60)
    print("SUMMARY: Gradient Analysis at scale=1.0")
    print("="*60)

    for metric in gradient_metrics:
        z_grad = combined_z_grad[combined_z_grad['metric'] == metric]['gradient'].mean()
        s_grad = combined_s_grad[combined_s_grad['metric'] == metric]['gradient'].mean()
        z_grad_std = combined_z_grad[combined_z_grad['metric'] == metric]['gradient'].std()
        s_grad_std = combined_s_grad[combined_s_grad['metric'] == metric]['gradient'].std()
        ratio = abs(z_grad) / (abs(s_grad) + 1e-10)

        print(f"\n{metric}:")
        print(f"  Z gradient: {z_grad:.6e} +/- {z_grad_std:.6e}")
        print(f"  S gradient: {s_grad:.6e} +/- {s_grad_std:.6e}")
        print(f"  |Z|/|S| ratio: {ratio:.2f}x")

    print(f"\nAll outputs saved to: {output_dir}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    """CLI entry point: parse arguments and run the full z vs s scaling + gradient experiment."""
    parser = argparse.ArgumentParser(
        description="Compare effects of scaling z (pair) vs s (single) representations"
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
                        help="Scale factors to test (default: 0.0 0.25 0.5 0.75 1.0 1.25 1.5 2.0)")
    parser.add_argument("--plot_every_n", type=int, default=10,
                        help="Plot running averages every N cases (default: 10)")

    args = parser.parse_args()

    run_z_vs_s_experiment(
        parquet_path=args.parquet,
        n_cases=args.n_cases,
        output_dir=args.output_dir,
        device=args.device,
        scales=args.scales,
        plot_every_n=args.plot_every_n,
    )


if __name__ == "__main__":
    main()