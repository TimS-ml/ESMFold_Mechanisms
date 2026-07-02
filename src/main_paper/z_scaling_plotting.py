#!/usr/bin/env python
"""
Plot Z vs S Scaling Results
===========================

Creates a clean figure showing how scaling z (pairwise) vs s (sequence) 
representations affects mean pairwise distance.

Usage:
    python plot_z_vs_s_scaling.py --z_csv z_scaling_results.csv --s_csv s_scaling_results.csv --output figure.png
"""

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_z_vs_s_scaling(
    z_df: pd.DataFrame,
    s_df: pd.DataFrame,
    output_path: str = 'z_vs_s_scaling.png',
    metric: str = 'full_mean_ca_dist',
    figsize: Tuple[float, float] = (18, 6),
    dpi: int = 150,
    # Text size configuration
    axis_label_size: float = 30,
    tick_label_size: float = 28,
    legend_size: float = 28,
    # Colors
    z_color: str = '#1b9e77',  # Teal (pairwise)
    s_color: str = '#d95f02',  # Burnt orange (sequence)
):
    """
    Create a plot comparing z vs s scaling effects on change in mean pairwise distance.
    
    Args:
        z_df: DataFrame with z-scaling results
        s_df: DataFrame with s-scaling results
        output_path: Path to save the figure
        metric: Which metric to plot (default: full_mean_ca_dist)
        figsize: Figure size in inches
        dpi: Resolution for saved figure
        axis_label_size: Font size for axis labels
        tick_label_size: Font size for tick labels
        legend_size: Font size for legend
        z_color: Color for z (pairwise) line
        s_color: Color for s (sequence) line
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute change from baseline (scale=1.0) for each case
    # (each case is centered on its OWN scale=1.0 baseline, not a single global baseline - needed
    # since different cases are different protein sequences whose absolute metric values can
    # differ substantially; this makes deltas comparable/poolable across cases)
    # For z results
    z_baseline = z_df[z_df['scale'] == 1.0].set_index('case_idx')[metric]
    z_df = z_df.copy()
    # Row-wise apply (axis=1) since we need both this row's case_idx and its metric value at
    # once; .get(..., row[metric]) defensively falls back to delta=0 if a case's own baseline
    # row is somehow missing, instead of raising a KeyError.
    z_df['delta'] = z_df.apply(lambda row: row[metric] - z_baseline.get(row['case_idx'], row[metric]), axis=1)
    
    # For s results
    s_baseline = s_df[s_df['scale'] == 1.0].set_index('case_idx')[metric]
    s_df = s_df.copy()
    s_df['delta'] = s_df.apply(lambda row: row[metric] - s_baseline.get(row['case_idx'], row[metric]), axis=1)
    
    # Group by scale and compute mean and std of delta
    # (pools the per-case, baseline-centered deltas across ALL cases at each scale value;
    # reset_index() turns the 'scale' groupby key back into a plain column for the positional
    # .values access used below)
    z_grouped = z_df.groupby('scale')['delta'].agg(['mean', 'std']).reset_index()
    s_grouped = s_df.groupby('scale')['delta'].agg(['mean', 'std']).reset_index()
    
    scales_z = z_grouped['scale'].values
    scales_s = s_grouped['scale'].values
    
    z_mean = z_grouped['mean'].values
    z_std = z_grouped['std'].values
    s_mean = s_grouped['mean'].values
    s_std = s_grouped['std'].values
    
    # Plot with error bands
    ax.plot(scales_z, z_mean, 'o-', color=z_color, linewidth=2.5, markersize=8,
            label='Pairwise')
    ax.fill_between(scales_z, z_mean - z_std, z_mean + z_std, 
                    color=z_color, alpha=0.2)
    
    ax.plot(scales_s, s_mean, 's-', color=s_color, linewidth=2.5, markersize=8,
            label='Sequence')
    ax.fill_between(scales_s, s_mean - s_std, s_mean + s_std,
                    color=s_color, alpha=0.2)
    
    # Add interpolated marker at scale=1.75
    # (1.75 is the midpoint of the 1.5-2.0 gap, which - unlike the rest of the default scale
    # grid, spaced by 0.25 - is a wider 0.5 step; this linearly-interpolated point is purely a
    # cosmetic visual aid to fill that gap, not an actually-computed data point)
    if 1.5 in scales_z and 2.0 in scales_z:
        idx_15 = np.where(scales_z == 1.5)[0][0]
        idx_20 = np.where(scales_z == 2.0)[0][0]
        z_interp = z_mean[idx_15] + (z_mean[idx_20] - z_mean[idx_15]) * 0.5  # linear interpolation, halfway
        ax.plot(1.75, z_interp, 'o', color=z_color, markersize=8, zorder=5)
    
    if 1.5 in scales_s and 2.0 in scales_s:
        idx_15 = np.where(scales_s == 1.5)[0][0]
        idx_20 = np.where(scales_s == 2.0)[0][0]
        s_interp = s_mean[idx_15] + (s_mean[idx_20] - s_mean[idx_15]) * 0.5
        ax.plot(1.75, s_interp, 's', color=s_color, markersize=8, zorder=5)
    
    # Reference lines
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Styling
    ax.set_xlabel('Scale Factor', fontsize=axis_label_size)
    ax.set_ylabel('Δ Mean Pairwise Distance (Å)', fontsize=axis_label_size)
    ax.tick_params(axis='both', labelsize=tick_label_size)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    ax.legend(loc='best', fontsize=legend_size, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved figure to: {output_path}")
    return fig


def main():
    """CLI entry point: load z/s scaling result CSVs, plot the comparison, and print a summary."""
    parser = argparse.ArgumentParser(description='Plot z vs s scaling results')
    parser.add_argument('--z_csv', type=str, default='z_vs_s_scaling/z_scaling_results.csv',
                        help='Path to z_scaling_results.csv')
    parser.add_argument('--s_csv', type=str, default='z_vs_s_scaling/s_scaling_results.csv',
                        help='Path to s_scaling_results.csv')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for figure')
    parser.add_argument('--metric', type=str, default='full_mean_ca_dist',
                        choices=['full_mean_ca_dist', 'hairpin_mean_ca_dist', 'full_rg', 
                                 'hairpin_rg', 'strand_sep', 'rmsd_all'],
                        help='Metric to plot (default: full_mean_ca_dist)')
    parser.add_argument('--figsize', type=float, nargs=2, default=[7, 6],
                        help='Figure size (width height)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for saved figure')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading z results from {args.z_csv}...")
    z_df = pd.read_csv(args.z_csv)
    print(f"Loaded {len(z_df)} rows")
    
    print(f"Loading s results from {args.s_csv}...")
    s_df = pd.read_csv(args.s_csv)
    print(f"Loaded {len(s_df)} rows")
    
    # Print available scales
    print(f"\nScales in z results: {sorted(z_df['scale'].unique())}")
    print(f"Scales in s results: {sorted(s_df['scale'].unique())}")
    
    # Set output path
    if args.output is None:
        # os.path.dirname returns '' if z_csv is a bare filename with no directory component;
        # `or '.'` falls back to the current directory in that case.
        output_dir = os.path.dirname(args.z_csv) or '.'
        args.output = os.path.join(output_dir, f'z_vs_s_{args.metric}.png')
    
    # Create plot
    plot_z_vs_s_scaling(
        z_df=z_df,
        s_df=s_df,
        output_path=args.output,
        metric=args.metric,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
    )
    
    # Print summary statistics
    print("\n" + "="*50)
    print(f"Summary for Δ {args.metric}")
    print("="*50)
    
    # Compute deltas for summary
    # (repeats the same per-case-baseline-centered delta computation as plot_z_vs_s_scaling
    # above, since that function only returns the fig object, not its intermediate delta/grouped
    # values - recomputed here just to print the Z/S range comparison as text)
    z_baseline = z_df[z_df['scale'] == 1.0].set_index('case_idx')[args.metric]
    z_df_copy = z_df.copy()
    z_df_copy['delta'] = z_df_copy.apply(lambda row: row[args.metric] - z_baseline.get(row['case_idx'], row[args.metric]), axis=1)
    
    s_baseline = s_df[s_df['scale'] == 1.0].set_index('case_idx')[args.metric]
    s_df_copy = s_df.copy()
    s_df_copy['delta'] = s_df_copy.apply(lambda row: row[args.metric] - s_baseline.get(row['case_idx'], row[args.metric]), axis=1)
    
    z_grouped = z_df_copy.groupby('scale')['delta'].mean()
    s_grouped = s_df_copy.groupby('scale')['delta'].mean()
    
    z_range = z_grouped.max() - z_grouped.min()
    s_range = s_grouped.max() - s_grouped.min()
    
    print(f"Z scaling Δ range: {z_range:.2f} Å")
    print(f"S scaling Δ range: {s_range:.2f} Å")
    print(f"Z/S ratio: {z_range / (s_range + 1e-8):.2f}x")
    
    print("\nDone!")


if __name__ == '__main__':
    main()