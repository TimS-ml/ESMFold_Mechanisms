"""
ESMFold Block Patching Analysis - Standalone Plotting Script
=============================================================
Generates summary plots from experiment results.

Usage:
    python plot_block_patching_results.py --results results/block_patching_results.csv --output_dir results/
"""

import argparse
import os
from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION - Edit these to iterate quickly
# =============================================================================

# Font sizes for formation rate lines plot
FORMATION_TITLE_FONTSIZE = 28  # currently unused: plot_formation_rate_lines() never calls set_title()
FORMATION_LABEL_FONTSIZE = 18
FORMATION_TICK_FONTSIZE = 18
FORMATION_LEGEND_FONTSIZE = 18

# Y-axis limits for formation rate plot
# Fixed (not auto-scaled) 0-60% window tuned to this experiment's observed
# formation rates; a block/mode combination with a rate above 60% would be
# silently clipped rather than shown in full.
FORMATION_YLIM = (0.0, 60.0)
# Global, import-time matplotlib rcParam change -- takes effect as soon as
# this module is imported (not just when a plotting function is called), and
# affects any other plot drawn in the same process. If "Trebuchet MS" isn't
# installed, matplotlib falls back to a default font with a warning rather
# than raising an error.
plt.rcParams['font.family'] = 'Trebuchet MS'
# Figure size
FORMATION_FIGSIZE = (14, 4)

# =============================================================================
# Plotting Functions
# =============================================================================

def plot_formation_rate_lines(results_df: pd.DataFrame, output_dir: str):
    """
    Line plot showing hairpin formation rate by block for sequence and pairwise (touch) modes,
    with filled area under the curves.
    """
    fig, ax = plt.subplots(figsize=FORMATION_FIGSIZE)
    
    # Slightly muted, professional colors
    # colors = {"sequence": "#E07B53", "pairwise": "#5BA08C"}  # Muted orange, Muted teal
    colors = {"sequence": "#d95f02", "pairwise": "#1b9e77"}  # Burnt orange, Teal

    # Track block range for xlim
    all_blocks = []
    
    # Plot sequence mode
    seq_df = results_df[results_df["patch_mode"] == "sequence"]
    if len(seq_df) > 0:
        # hairpin_found is boolean, so groupby(...).mean() = fraction True =
        # per-block success rate; x100 converts it to a percentage.
        fraction_per_block = seq_df.groupby("block_idx")["hairpin_found"].mean()
        pct_per_block = 100.0 * fraction_per_block

        # Record which block indices actually have data (not assumed to be a
        # fixed 0-47 range), so ax.set_xlim below can match the real tested
        # range exactly.
        all_blocks.extend(pct_per_block.index.tolist())
        ax.plot(pct_per_block.index, pct_per_block.values,
                label="Sequence Patch",
                color=colors["sequence"], linewidth=2.5)
        ax.fill_between(pct_per_block.index, pct_per_block.values,
                        alpha=0.25, color=colors["sequence"])
    
    # Plot pairwise (touch) mode only
    # NOTE: possible bug -- this comment says "touch" mode, but the filter
    # below actually selects patch_mask_mode == "intra", not "touch". Either
    # the comment is stale or the filter is testing the wrong mask mode.
    pair_df = results_df[(results_df["patch_mode"] == "pairwise") & 
                         (results_df["patch_mask_mode"] == "intra")]
    if len(pair_df) > 0:
        fraction_per_block = pair_df.groupby("block_idx")["hairpin_found"].mean()
        pct_per_block = 100.0 * fraction_per_block

        all_blocks.extend(pct_per_block.index.tolist())
        ax.plot(pct_per_block.index, pct_per_block.values,
                label="Pairwise Patch",
                color=colors["pairwise"], linewidth=2.5)
        ax.fill_between(pct_per_block.index, pct_per_block.values,
                        alpha=0.25, color=colors["pairwise"])

    
    # Set x limits to remove whitespace
    if all_blocks:
        ax.set_xlim(min(all_blocks), max(all_blocks))
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # # Subtle horizontal grid only
    # ax.xaxis.grid(True, linestyle='-', alpha=0.4, color='gray')
    # ax.yaxis.grid(False)
    # ax.set_axisbelow(True)
    
    ax.set_xlabel("Block Index", fontsize=FORMATION_LABEL_FONTSIZE)
    ax.set_ylabel("% of Outputs with Hairpin", fontsize=FORMATION_LABEL_FONTSIZE)
    ax.set_ylim(FORMATION_YLIM)
    # Ticks every 20 percentage points; np.arange's stop bound is exclusive,
    # so without the "+ 0.01" the last tick (60.0) would be silently dropped
    # since 0 -> 20 -> 40 -> 60 lands exactly on the exclusive stop of 60.
    ax.set_yticks(np.arange(FORMATION_YLIM[0], FORMATION_YLIM[1] + 0.01, 20))

    ax.tick_params(axis='both', labelsize=FORMATION_TICK_FONTSIZE)
    ax.legend(loc='center right', fontsize=FORMATION_LEGEND_FONTSIZE, frameon=False)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "formation_rate_lines.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def generate_all_plots(results_df: pd.DataFrame, output_dir: str, include_helix: bool = False):
    """Generate all summary plots."""
    # NOTE: possible bug -- include_helix is accepted but never used in this
    # function body; unlike print_summary_statistics() below, no helix-related
    # plot is generated here regardless of this flag's value.
    print("\nGenerating summary plots...")
    
    # Hairpin formation plots
    plot_formation_rate_lines(results_df, output_dir)

    print("\nAll plots generated!")


def print_summary_statistics(results_df: pd.DataFrame, include_helix: bool = False):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    print(f"\nTotal experiments: {len(results_df)}")
    print(f"Unique cases: {results_df['case_idx'].nunique()}")
    print(f"Blocks tested: {results_df['block_idx'].nunique()}")
    
    mask_modes = results_df['patch_mask_mode'].unique()
    
    for mask_mode in mask_modes:
        print(f"\n{'='*60}")
        print(f"MASK MODE: {mask_mode.upper()}")
        print("="*60)
        
        mask_df = results_df[results_df['patch_mask_mode'] == mask_mode]
        
        print("\nFormation rates by patch mode:")
        # hairpin_found is boolean, so groupby(...).mean() = success rate.
        rates = mask_df.groupby('patch_mode')['hairpin_found'].mean()
        for mode, rate in rates.items():
            n = len(mask_df[mask_df['patch_mode'] == mode])
            # Recover the raw success count from rate * n (both derived from
            # the same subset, so this is mathematically exact aside from
            # possible floating-point rounding); int() truncates rather than
            # rounds, so a result landing just under the true integer (e.g.
            # 6.999999999 instead of 7.0) would under-report by one.
            print(f"  {mode:12s}: {rate:.1%} ({int(rate*n)}/{n})")
        
        print("\nMean magnitude when hairpin found:")
        # NOTE: possible bug -- there is no "magnitude" column in
        # block_patching.py's output schema (evaluate_hairpin() there returns
        # hairpin_found/mean_plddt/patch_region_plddt/ptm only; "magnitude"
        # appears to be from a different experiment's schema, e.g. a steering
        # sweep). mode_df['magnitude'] would raise a KeyError here whenever
        # mode_df is non-empty (i.e. whenever any hairpin was found for this
        # patch_mode), so this block only "works" by accident when there
        # happen to be zero hairpin-found rows for every mode.
        for mode in ["sequence", "pairwise", "both"]:
            mode_df = mask_df[(mask_df['patch_mode'] == mode) & 
                              (mask_df['hairpin_found'] == True)]
            if len(mode_df) > 0:
                mean_mag = mode_df['magnitude'].mean()
                print(f"  {mode:12s}: {mean_mag:.3f}")
            else:
                print(f"  {mode:12s}: N/A (no hairpins found)")
        
        print("\nFormation rate by block (averaged across patch modes):")
        block_rates = mask_df.groupby('block_idx')['hairpin_found'].mean()
        if len(block_rates) > 0:
            # idxmin()/idxmax() return the *index label* (here, the block_idx
            # value) of the min/max rate, not the rate itself -- that's why
            # both idxmin()/min() and idxmax()/max() are called.
            print(f"  Min: Block {block_rates.idxmin()} ({block_rates.min():.1%})")
            print(f"  Max: Block {block_rates.idxmax()} ({block_rates.max():.1%})")
        
        # Alpha helix statistics
        # Guarded on 'patched_helix_pct' being present since this is an
        # optional column (block_patching.py's run_experiment_on_dataset has
        # no compute_helix option and never produces these columns itself --
        # this branch is only reachable if results_df comes from elsewhere).
        if include_helix and 'patched_helix_pct' in mask_df.columns:
            print("\n" + "-"*60)
            print(f"ALPHA HELIX CONTENT ANALYSIS ({mask_mode.upper()})")
            print("-"*60)
            
            # Original helix content (should be same across all rows for a case)
            # .first() picks one representative value per case_idx, since this
            # value is constant across every block/patch_mode row for that case.
            orig_helix = mask_df.groupby('case_idx')['original_helix_pct'].first()
            print(f"\nOriginal structure helix content:")
            print(f"  Mean: {orig_helix.mean():.1f}%")
            print(f"  Range: {orig_helix.min():.1f}% - {orig_helix.max():.1f}%")
            
            print("\nMean helix change by patch mode:")
            for mode in ["sequence", "pairwise", "both"]:
                mode_df = mask_df[mask_df['patch_mode'] == mode]
                if len(mode_df) > 0 and mode_df['helix_absolute_change'].notna().any():
                    mean_change = mode_df['helix_absolute_change'].mean()
                    std_change = mode_df['helix_absolute_change'].std()
                    print(f"  {mode:12s}: {mean_change:+.2f} ± {std_change:.2f} pp")
                else:
                    print(f"  {mode:12s}: N/A")
            
            print("\nHelix change by block (averaged across patch modes):")
            block_helix = mask_df.groupby('block_idx')['helix_absolute_change'].mean()
            if len(block_helix) > 0 and block_helix.notna().any():
                print(f"  Most helix loss:    Block {block_helix.idxmin()} ({block_helix.min():+.2f} pp)")
                print(f"  Most helix gain:    Block {block_helix.idxmax()} ({block_helix.max():+.2f} pp)")
            
            # Correlation between hairpin and helix
            valid_df = mask_df.dropna(subset=['helix_absolute_change'])
            if len(valid_df) > 0:
                # .astype(float) turns True/False into 1.0/0.0 so .corr() (which
                # expects numeric data) can compute a Pearson correlation
                # between "hairpin formed" and the helix-content change.
                corr = valid_df['hairpin_found'].astype(float).corr(valid_df['helix_absolute_change'])
                print(f"\nCorrelation (hairpin found vs helix change): {corr:.3f}")
    
    # Compare mask modes
    if len(mask_modes) > 1:
        print("\n" + "="*60)
        print("COMPARISON ACROSS MASK MODES")
        print("="*60)
        
        # .unstack() pivots the last groupby level (patch_mode) from rows into
        # columns, turning the grouped Series into a mask_mode x patch_mode
        # table of success rates (equivalent to a pivot_table with
        # index='patch_mask_mode', columns='patch_mode').
        comparison = results_df.groupby(['patch_mask_mode', 'patch_mode'])['hairpin_found'].mean().unstack()
        print("\nFormation rates (rows=mask mode, cols=patch mode):")
        print(comparison.to_string())


def main():
    """CLI entry point: load results, print summary statistics, and generate plots."""
    parser = argparse.ArgumentParser(description="Plot ESMFold patching experiment results")
    parser.add_argument(
        # NOTE: possible bug -- this default is a hardcoded absolute path on a
        # specific machine/user's home directory, unlike module_patching.py /
        # block_patching.py's argparse defaults, which build paths relative to
        # a computed _PROJECT_ROOT. This default will not exist on any other
        # machine, so --results must be passed explicitly in practice.
        "--results", type=str, default="/share/u/kevin/ProteinFolding/base_patching_results/block_patching_results.parquet",
        help="Path to results parquet file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="plots/base_patching",
        help="Output directory for plots (default: same as results)"
    )
    parser.add_argument(
        "--include_helix", action="store_true",
        help="Include alpha helix analysis plots"
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results}...")
    results_df = pd.read_parquet(args.results)
    print(f"Loaded {len(results_df)} rows")
    
    # Output directory
    # Note: --output_dir's argparse default is "plots/base_patching" (not
    # None), so in practice this is always truthy and the os.path.dirname
    # fallback below is unreachable unless a caller explicitly passes
    # output_dir=None some other way.
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.dirname(args.results)
    os.makedirs(output_dir, exist_ok=True)
    
    # Auto-detect helix data
    # (includes helix analysis if explicitly requested via --include_helix, or
    # if the loaded results already contain helix columns -- produced by a
    # different upstream experiment/runner than this one; see the NOTE in
    # print_summary_statistics above)
    include_helix = args.include_helix or 'patched_helix_pct' in results_df.columns
    
    # Generate outputs
    print_summary_statistics(results_df, include_helix=include_helix)
    generate_all_plots(results_df, output_dir, include_helix=include_helix)
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()