"""
Pair-to-Sequence Bias Analysis Plotting Script
===============================================
Standalone script for plotting pair-to-sequence attention bias analysis results.

Usage:
    python plot_bias_analysis.py
    python plot_bias_analysis.py --metrics path/to/metrics.parquet --output_dir plots/
"""

import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

# Shared helper: loads facebook/esmfold_v1 + tokenizer with auto-selected precision (see model_utils.py).
from src.utils.model_utils import load_esmfold

# =============================================================================
# CONFIGURATION - Edit these to iterate quickly
# =============================================================================

# Font sizes
TITLE_FONTSIZE = 24
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 16
LEGEND_FONTSIZE = 24

# Figure size
FIGSIZE = (8, 7)

# Colors
COLOR_GLOBAL = '#d95f02'    # Orange
COLOR_HAIRPIN = '#1b9e77'   # Teal

# Fill alpha
FILL_ALPHA = 0.3

# Smoothing window
SMOOTHING_WINDOW = 1

# Contact map settings
CONTACT_THRESHOLD = 8.0  # Angstroms

# =============================================================================
# Plotting Functions
# =============================================================================

def plot_global_auc(metrics_df: pd.DataFrame, output_dir: str):
    """Plot Global Contact AUC across blocks."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    # Average AUC across all analyzed sequences, per block.
    agg = metrics_df.groupby('block_idx')['global_auc'].mean().reset_index()
    
    # Apply smoothing
    # center=True: window centered on each block (uses neighbors on both sides);
    # min_periods=1 avoids NaN at the edge blocks where a full window isn't available.
    # SMOOTHING_WINDOW=1 above makes this a no-op (rolling mean over a single point).
    agg['global_auc_smooth'] = agg['global_auc'].rolling(
        window=SMOOTHING_WINDOW, center=True, min_periods=1
    ).mean()
    
    # Fill under curve
    # Shades the area between chance-level (0.5) and the AUC curve, visually
    # emphasizing how far above/below chance the contact-prediction AUC is.
    ax.fill_between(agg['block_idx'], 0.5, agg['global_auc_smooth'],
                   alpha=FILL_ALPHA, color=COLOR_GLOBAL)
    # Line
    ax.plot(agg['block_idx'], agg['global_auc_smooth'], 
           'o-', color=COLOR_GLOBAL, label='Global AUC', markersize=5, linewidth=2.5)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)  # chance-level AUC reference line
    
    ax.set_xlabel('Block', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Contact Prediction AUC', fontsize=LABEL_FONTSIZE)
    ax.set_title('Global: Pair2Seq Bias Contact AUC', fontsize=TITLE_FONTSIZE, pad=12)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlim(0, 47)  # 47 = index of the last (48th) trunk block
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bias_global_auc.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: bias_global_auc.png")


def plot_hairpin_auc(metrics_df: pd.DataFrame, output_dir: str):
    """Plot Hairpin Contact AUC across blocks."""
    # Same aggregation/smoothing/fill pattern as plot_global_auc() above, restricted
    # to the hairpin_auc metric instead of global_auc.
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    agg = metrics_df.groupby('block_idx')['hairpin_auc'].mean().reset_index()
    
    # Apply smoothing
    agg['hairpin_auc_smooth'] = agg['hairpin_auc'].rolling(
        window=SMOOTHING_WINDOW, center=True, min_periods=1
    ).mean()
    
    # Fill under curve
    ax.fill_between(agg['block_idx'], 0.5, agg['hairpin_auc_smooth'],
                   alpha=FILL_ALPHA, color=COLOR_HAIRPIN)
    # Line
    ax.plot(agg['block_idx'], agg['hairpin_auc_smooth'], 
           'o-', color=COLOR_HAIRPIN, label='Hairpin AUC', markersize=5, linewidth=2.5)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Block', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Contact Prediction AUC', fontsize=LABEL_FONTSIZE)
    ax.set_title('Pair2Seq Bias Aligns With Contact Maps (Hairpin)', fontsize=TITLE_FONTSIZE, pad=12)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlim(0, 47)
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bias_hairpin_auc.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: bias_hairpin_auc.png")


def plot_contact_map(sequence: str, name: str, hairpin_start: int, hairpin_end: int,
                     model, tokenizer, device, output_dir: str):
    """Generate and plot bias + contacts overlay for a sequence."""
    import types  # local import: only needed inside this function
    
    print(f"  Generating bias + contacts for {name}...")
    
    # We need to collect the bias at a specific block - use block 30 as default
    VIZ_BLOCK = 30
    
    # Collect bias using modified forward
    # Same monkey-patch-the-trunk-forward technique used throughout this codebase:
    # temporarily replace model.trunk.forward with a version that intercepts
    # block.pair_to_sequence's output (the bias) at VIZ_BLOCK, then restore it.
    collected_data = {'biases': {}}
    original_trunk_forward = model.trunk.forward
    
    def collect_bias_trunk_forward(self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles):
        from transformers.models.esm.modeling_esmfold import EsmFoldingTrunk
        from transformers.utils import ContextManagers
        
        device = seq_feats.device
        s_s_0, s_z_0 = seq_feats, pair_feats
        
        if no_recycles is None:
            no_recycles = self.config.max_recycles
        else:
            no_recycles += 1

        def trunk_iter(s, z, residx, mask):
            z = z + self.pairwise_positional_embedding(residx, mask=mask)
            for block_idx, block in enumerate(self.blocks):
                if block_idx == VIZ_BLOCK:
                    bias = block.pair_to_sequence(z)  # [1, L, L, num_heads]
                    collected_data['biases'][block_idx] = bias.detach().cpu()
                s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size)
            return s, z

        s_s, s_z = s_s_0, s_z_0
        recycle_s = torch.zeros_like(s_s)
        recycle_z = torch.zeros_like(s_z)
        recycle_bins = torch.zeros(*s_z.shape[:-1], device=device, dtype=torch.int64)

        # Standard ESMFold/AlphaFold recycling loop (num_recycles=0 below -> a single,
        # non-recycled pass through trunk_iter).
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
                # Discretize predicted CA-CA distances into bins spanning 3.375-21.375 Å
                # for the next recycle's distance-embedding input.
                recycle_bins = EsmFoldingTrunk.distogram(
                    structure["positions"][-1][:, :, :3], 3.375, 21.375, self.recycle_bins,
                )

        structure["s_s"], structure["s_z"] = s_s, s_z
        return structure
    
    model.trunk.forward = types.MethodType(collect_bias_trunk_forward, model.trunk)
    
    with torch.no_grad():
        inputs = tokenizer(sequence, return_tensors='pt', add_special_tokens=False).to(device)
        outputs = model(**inputs, num_recycles=0)
    
    # NOTE: possible bug -- no try/finally around the patched forward call above; if
    # model(**inputs, ...) raises, model.trunk.forward is never restored and stays
    # monkey-patched for the lifetime of this model object.
    model.trunk.forward = original_trunk_forward
    
    # Get CA positions and compute distances
    ca_positions = outputs.positions[-1, 0, :, 1, :]  # CA is atom index 1
    diff = ca_positions.unsqueeze(0) - ca_positions.unsqueeze(1)
    distances = torch.sqrt((diff ** 2).sum(-1)).cpu().numpy()
    contacts = (distances < CONTACT_THRESHOLD).astype(float)  # binary contact map at the configured Å cutoff
    
    # Get bias
    bias = collected_data['biases'][VIZ_BLOCK]
    bias_avg = bias[0].mean(dim=-1).cpu().numpy()  # average over heads -> [L, L]
    
    # Create single plot: Global Bias + Contacts overlay
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    vmax = np.percentile(np.abs(bias_avg), 98)
    im = ax.imshow(bias_avg, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.contour(contacts, levels=[0.5], colors='lime', linewidths=2)
    
    ax.set_xlabel('Residue', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Residue', fontsize=LABEL_FONTSIZE)
    ax.set_title('Global Pair2Seq Bias (Block 30)', fontsize=TITLE_FONTSIZE, pad=12)
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    
    # Colorbar for bias
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pair2Seq Bias Value', fontsize=LABEL_FONTSIZE - 2)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE - 2)
    
    # Legend for contacts
    # ax.contour() doesn't automatically produce a legend entry, so a Line2D proxy
    # artist (never actually drawn) is created purely to supply a legend label/color.
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='lime', linewidth=2, label=f'Contacts (<{CONTACT_THRESHOLD}Å)')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=LEGEND_FONTSIZE - 4)
    
    plt.tight_layout()
    filename = f'bias_contacts_{name}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")
    
    # Clean up
    del outputs
    torch.cuda.empty_cache()


def generate_contact_maps(seq_info_df: pd.DataFrame, output_dir: str, n_maps: int):
    """Generate contact maps for first N sequences."""
    print(f"\n--- Generating contact maps for {n_maps} sequences ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = load_esmfold(device)
    
    for idx, row in seq_info_df.head(n_maps).iterrows():
        plot_contact_map(
            sequence=row['sequence'],
            name=row['donor_pdb'],
            hairpin_start=int(row['hairpin_start']),
            hairpin_end=int(row['hairpin_end']),
            model=model,
            tokenizer=tokenizer,
            device=device,
            output_dir=output_dir,
        )
    
    # This function loads its own model instance (separate from any model used
    # elsewhere), so explicitly free it and release cached GPU memory once done.
    del model
    torch.cuda.empty_cache()


def generate_all_plots(metrics_df: pd.DataFrame, output_dir: str, 
                       seq_info_df: pd.DataFrame = None, n_contact_maps: int = 2):
    """Generate all plots."""
    print("\n--- Generating metric plots ---")
    plot_global_auc(metrics_df, output_dir)
    plot_hairpin_auc(metrics_df, output_dir)
    
    # Contact map generation requires a loaded model (re-running forward passes),
    # so it's skipped entirely if no sequence info was provided/loaded.
    if seq_info_df is not None and len(seq_info_df) > 0:
        generate_contact_maps(seq_info_df, output_dir, n_contact_maps)


def main():
    """Entry point: load saved bias-analysis metrics (and optionally sequence info) and generate all plots."""
    parser = argparse.ArgumentParser(description="Plot pair-to-sequence bias analysis results")
    parser.add_argument("--metrics", type=str, default="better_bias_analysis/metrics.parquet", help="Path to metrics parquet")
    parser.add_argument("--seq_info", type=str, default="better_bias_analysis/sequence_info.parquet", help="Path to sequence info parquet")
    parser.add_argument("--output_dir", type=str, default="bias_plotting", help="Output directory")
    parser.add_argument("--n_contact_maps", type=int, default=2, help="Number of contact maps to generate")
    parser.add_argument("--skip_contact_maps", action="store_true", help="Skip contact map generation")
    args = parser.parse_args()
    
    print(f"Loading {args.metrics}...")
    metrics_df = pd.read_parquet(args.metrics)
    print(f"Loaded {len(metrics_df)} rows")
    
    print(f"\nUnique sequences: {metrics_df['seq_idx'].nunique()}")
    print(f"Blocks: {sorted(metrics_df['block_idx'].unique())}")
    
    # Load sequence info for contact maps
    seq_info_df = None
    if not args.skip_contact_maps and os.path.exists(args.seq_info):
        print(f"\nLoading {args.seq_info}...")
        seq_info_df = pd.read_parquet(args.seq_info)
        print(f"Loaded {len(seq_info_df)} sequences")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    generate_all_plots(
        metrics_df, 
        args.output_dir, 
        seq_info_df if not args.skip_contact_maps else None,
        args.n_contact_maps
    )
    
    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()