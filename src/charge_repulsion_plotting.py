#!/usr/bin/env python
"""
Hairpin Disruption Plotting Script
==================================

Generates plots from hairpin disruption experiment results.
Main focus: mean distance change vs block position.

Usage:
    python charge_disruption_plots.py \
        --results results_disp_ws/hairpin_disruption_results.parquet \
        --output charge_disruption_plots/
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Number of folding-trunk blocks in ESMFold (matches EsmFoldConfig.trunk.num_blocks)
NUM_BLOCKS = 48


def load_results(path: str) -> pd.DataFrame:
    """Load results from parquet or CSV."""
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path)


def plot_charge_mode_comparison_ws10(
    results_df: pd.DataFrame,
    induction_df: pd.DataFrame,
    output_dir: str,
):
    """Plot mean cross-strand distance change vs. sliding-window start block, overlaying
    both disruption charge modes (Pos-Pos/Neg-Neg) and both induction polarities
    (Pos-Neg/Neg-Pos) with 95% CI bands, using a broken (top/bottom) y-axis.
    """
    # NOTE: possible bug -- despite the "_ws10" in the function name (and the output
    # filename below is "..._ws15.png"), the filters used throughout are window_size == 15,
    # not 10. This looks like a leftover from renaming/changing the fixed window size
    # without updating the function name to match.
    from matplotlib.gridspec import GridSpec

    # --- Disruption data: both_positive, both_negative ---
    # Fix a single representative (window_size, magnitude) slice for a clean comparison plot
    interventions = results_df[results_df['intervention_config'] != 'baseline']
    ws_data = interventions[
        (interventions['window_size'] == 15) & (interventions['magnitude'] == 0.5)
    ]

    # --- Induction data: pos_neg, neg_pos ---
    # Induction results use a different column name ('block_set') for the baseline marker
    # and a much larger representative magnitude (3.0 vs 0.5) -- the two experiments were
    # tuned independently, so these are not meant to be numerically comparable magnitudes.
    ind_interventions = induction_df[induction_df['block_set'] != 'baseline']
    ind_data = ind_interventions[
        (ind_interventions['window_size'] == 15) & (ind_interventions['magnitude'] == 3.0)
    ]

    # Broken y-axis layout: a short top panel (positive changes) stacked tightly (hspace=0.05)
    # on a taller bottom panel (negative changes), sharing the x-axis. Both panels get the
    # same data plotted; only their y-limits differ (set below) to create the "break".
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 1, height_ratios=[1, 2], hspace=0.05)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    axes = [ax_top, ax_bot]

    # Plot pos-pos and neg-neg from disruption data
    for charge_mode, color, label in [
        ('both_positive', 'blue', 'Pos-Pos'),
        ('both_negative', 'red', 'Neg-Neg'),
    ]:
        cm_data = ws_data[ws_data['charge_mode'] == charge_mode]
        if len(cm_data) == 0:
            continue
        mean_change = cm_data.groupby('window_start')['dist_change'].mean()
        # .sem() = standard error of the mean; fillna(0) handles window_start groups with
        # only 1 sample (SEM of a single point is NaN). 1.96x SEM = ~95% normal-approx CI.
        se_change = cm_data.groupby('window_start')['dist_change'].sem().fillna(0)
        ci = 1.96 * se_change.reindex(mean_change.index, fill_value=0)

        for ax in axes:
            # markevery=2: only draw a marker on every other point (dense x-axis, reduces clutter)
            ax.plot(mean_change.index, mean_change.values, 'o-',
                    color=color, linewidth=2.5, markersize=5,
                    label=label, alpha=0.9, markevery=2)
            ax.fill_between(mean_change.index,  # shaded 95% CI band
                            mean_change.values - ci.values,
                            mean_change.values + ci.values,
                            color=color, alpha=0.15)

    # Plot pos-neg and neg-pos from induction data (same mean/SEM/CI logic as above)
    for polarity, color, label in [
        ('pos_neg', 'green', 'Pos-Neg'),
        ('neg_pos', 'orange', 'Neg-Pos'),
    ]:
        pol_data = ind_data[ind_data['polarity'] == polarity]
        if len(pol_data) == 0:
            continue
        mean_change = pol_data.groupby('window_start')['dist_change'].mean()
        se_change = pol_data.groupby('window_start')['dist_change'].sem().fillna(0)
        ci = 1.96 * se_change.reindex(mean_change.index, fill_value=0)

        for ax in axes:
            ax.plot(mean_change.index, mean_change.values, 'o-',
                    color=color, linewidth=2.5, markersize=5,
                    label=label, alpha=0.9, markevery=2)
            ax.fill_between(mean_change.index,
                            mean_change.values - ci.values,
                            mean_change.values + ci.values,
                            color=color, alpha=0.15)

    # Set asymmetric y-limits: this is what actually creates the "break" -- the top panel
    # only displays [0, max] (positive changes) and the bottom only [min, 0] (negative
    # changes), so the two panels meet exactly at zero across the hspace=0.05 gap.
    all_ymin = min(ax_bot.get_ylim()[0], ax_top.get_ylim()[0])
    all_ymax = max(ax_bot.get_ylim()[1], ax_top.get_ylim()[1])
    ax_top.set_ylim(0, all_ymax)
    ax_bot.set_ylim(all_ymin, 0)

    for ax in axes:
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
        # x=16, x=32 divide the 48-block trunk into equal early/middle/late thirds
        ax.axvline(x=16, color='gray', linestyle=':', alpha=0.4)
        ax.axvline(x=32, color='gray', linestyle=':', alpha=0.4)
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=31)

    # Hide seam between axes: standard matplotlib "broken axis" trick -- drop the touching
    # spines/ticks (top panel's bottom, bottom panel's top) so the two panels read as one
    # continuous axis with a visual break at the shared y=0 line.
    ax_top.spines['bottom'].set_visible(False)
    ax_bot.spines['top'].set_visible(False)
    ax_top.tick_params(bottom=False, labelbottom=False)

    # Labels
    ax_bot.set_xlabel('Window Start Block', fontsize=38)
    # A single y-axis label spanning both stacked panels, placed via figure-relative
    # coordinates (ax.set_ylabel would only label one of the two panels).
    fig.text(0.02, 0.5, 'Strand Distance Δ (Å)', fontsize=38,
             va='center', rotation='vertical')

    # Legend only on top (avoids duplicating the same legend on both panels)
    ax_top.legend(loc='upper right', fontsize=30, ncols=2)

    # x-range sized for window_size=15 sliding windows (max valid start = NUM_BLOCKS - 15);
    # this hardcoded 15 must stay in sync with the window_size filters used above.
    ax_bot.set_xlim(-1, NUM_BLOCKS - 15 + 1)

    # Manually tuned margins for this specific (large-fontsize) publication figure
    fig.subplots_adjust(left=0.14, top=0.75, right=0.95, bottom=0.20)

    save_path = os.path.join(output_dir, 'dist_change_all_modes_ws15.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_dist_change_by_window_size(
    results_df: pd.DataFrame,
    output_dir: str,
):
    """
    Create one plot per window size showing mean distance change vs start block,
    with different lines for different magnitudes, separating pos/pos and neg/neg.
    """
    # Separate baseline and interventions
    baseline = results_df[results_df['intervention_config'] == 'baseline']  # computed but not used below
    interventions = results_df[results_df['intervention_config'] != 'baseline']
    
    window_sizes = sorted([ws for ws in interventions['window_size'].unique() if ws > 0])
    magnitudes = sorted(interventions['magnitude'].unique())
    charge_modes = sorted(interventions['charge_mode'].unique())
    
    # Color map for magnitudes: sample viridis between 15%-85% of its range (avoids the
    # very dark/very bright ends, which are hard to distinguish against a white background)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(magnitudes)))
    
    # Line styles for charge modes
    charge_linestyles = {
        'both_positive': '-',
        'both_negative': '--',
    }
    charge_markers = {
        'both_positive': 'o',
        'both_negative': 's',
    }
    charge_labels = {
        'both_positive': '++',
        'both_negative': '--',
    }
    
    for window_size in window_sizes:
        ws_data = interventions[interventions['window_size'] == window_size]
        
        # NOTE: possible bug -- figsize width of 147 (inches) looks like a typo; every
        # sibling plot in this file uses a width around 12-14 (e.g.
        # plot_hbond_change_by_window_size's figsize=(12, 6) below). This produces an
        # enormous, likely unusable image.
        fig, ax = plt.subplots(figsize=(147, 6))
        
        # Nested loop encodes magnitude via color and charge polarity via linestyle/marker,
        # so both dimensions are visible simultaneously in one legend.
        for mag, color in zip(magnitudes, colors):
            mag_data = ws_data[ws_data['magnitude'] == mag]
            
            for charge_mode in charge_modes:
                cm_data = mag_data[mag_data['charge_mode'] == charge_mode]
                
                if len(cm_data) == 0:
                    continue
                
                # Group by window_start and compute mean distance change
                mean_change = cm_data.groupby('window_start')['dist_change'].mean()
                
                linestyle = charge_linestyles.get(charge_mode, '-')
                marker = charge_markers.get(charge_mode, 'o')
                label_suffix = charge_labels.get(charge_mode, charge_mode)
                
                ax.plot(mean_change.index, mean_change.values, 
                        linestyle=linestyle, marker=marker,
                        color=color, linewidth=2, markersize=4, 
                        label=f'mag={mag} ({label_suffix})', alpha=0.8)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add vertical lines for block region boundaries (48 blocks total)
        ax.axvline(x=16, color='gray', linestyle=':', alpha=0.4)
        ax.axvline(x=32, color='gray', linestyle=':', alpha=0.4)
        
        ax.set_xlabel('Start Block', fontsize=13)
        ax.set_ylabel('Mean Distance Change (Å)', fontsize=13)
        ax.set_title(f'Cross-Strand Distance Change vs Block Position (Window Size = {window_size})\n(solid = ++, dashed = --, positive = disruption)', fontsize=14)
        ax.legend(loc='best', fontsize=8, ncol=4)
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)  # note: bottom spine hidden too (not just top), unlike most other plots in this file
        
        # Set x-axis limits: largest valid sliding-window start position is
        # NUM_BLOCKS - window_size (matches get_block_sets_for_sweep in charge_repulsion.py)
        max_start = NUM_BLOCKS - int(window_size)
        ax.set_xlim(-1, max_start + 1)
        
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'dist_change_ws{window_size}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")


def plot_hbond_change_by_window_size(
    results_df: pd.DataFrame,
    output_dir: str,
):
    """
    Create one plot per window size showing mean H-bond change vs start block,
    with different lines for different magnitudes.
    """
    baseline = results_df[results_df['intervention_config'] == 'baseline']  # computed but not used below
    interventions = results_df[results_df['intervention_config'] != 'baseline']
    
    window_sizes = sorted([ws for ws in interventions['window_size'].unique() if ws > 0])
    magnitudes = sorted(interventions['magnitude'].unique())
    
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(magnitudes)))
    
    for window_size in window_sizes:
        ws_data = interventions[interventions['window_size'] == window_size]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for mag, color in zip(magnitudes, colors):
            mag_data = ws_data[ws_data['magnitude'] == mag]
            
            # Group by window_start and compute mean H-bond change
            mean_change = mag_data.groupby('window_start')['hbond_change'].mean()
            std_change = mag_data.groupby('window_start')['hbond_change'].std()
            
            ax.plot(mean_change.index, mean_change.values, 'o-', 
                    color=color, linewidth=2, markersize=4, 
                    label=f'mag={mag}', alpha=0.8)
            
            # Shaded band is +/- 1 raw std dev here (not SEM/95% CI like
            # plot_charge_mode_comparison_ws10 above), so it's wider and less conservative
            ax.fill_between(mean_change.index, 
                            mean_change.values - std_change.values,
                            mean_change.values + std_change.values,
                            color=color, alpha=0.1)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=16, color='gray', linestyle=':', alpha=0.4)
        ax.axvline(x=32, color='gray', linestyle=':', alpha=0.4)
        
        ax.set_xlabel('Start Block', fontsize=13)
        ax.set_ylabel('Mean H-bond Change', fontsize=13)
        ax.set_title(f'H-bond Change vs Block Position (Window Size = {window_size})\n(negative = disruption)', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        
        max_start = NUM_BLOCKS - int(window_size)
        ax.set_xlim(-1, max_start + 1)
        
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'hbond_change_ws{window_size}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")


def plot_disruption_rate_by_window_size(
    results_df: pd.DataFrame,
    output_dir: str,
):
    """
    Create one plot per window size showing hairpin disruption rate vs start block,
    with different lines for different magnitudes.
    
    Disruption = baseline had hairpin but intervention doesn't.
    """
    # NOTE: possible bug -- this function (and the 'hairpin_found' column it relies on
    # below) does not match the disruption results schema produced by
    # run_hairpin_disruption_experiment in charge_repulsion.py, which has no
    # 'hairpin_found' column (see its case_results.append(...) dict keys). Calling this on
    # those results would raise a KeyError. Consistent with that, main() below has this
    # function commented out rather than called.
    baseline = results_df[results_df['intervention_config'] == 'baseline']
    interventions = results_df[results_df['intervention_config'] != 'baseline']
    
    # Get cases where baseline had hairpin
    baseline_with_hairpin = baseline[baseline['hairpin_found'] == True]['case_idx'].values
    
    # Filter interventions to only those cases
    interventions = interventions[interventions['case_idx'].isin(baseline_with_hairpin)]
    
    window_sizes = sorted([ws for ws in interventions['window_size'].unique() if ws > 0])
    magnitudes = sorted(interventions['magnitude'].unique())
    
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(magnitudes)))
    
    for window_size in window_sizes:
        ws_data = interventions[interventions['window_size'] == window_size]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for mag, color in zip(magnitudes, colors):
            mag_data = ws_data[ws_data['magnitude'] == mag]
            
            # Disruption rate = fraction where hairpin_found is False (was True at baseline).
            # `~x` inverts the boolean Series (True<->False); since these rows are already
            # restricted to baseline_with_hairpin cases, .mean() of the inverted series is
            # exactly the fraction where the hairpin was lost after intervention, as a percentage.
            disruption_rate = mag_data.groupby('window_start')['hairpin_found'].apply(
                lambda x: (~x).mean() * 100
            )
            
            ax.plot(disruption_rate.index, disruption_rate.values, 'o-', 
                    color=color, linewidth=2, markersize=4, 
                    label=f'mag={mag}', alpha=0.8)
        
        ax.axvline(x=16, color='gray', linestyle=':', alpha=0.4)
        ax.axvline(x=32, color='gray', linestyle=':', alpha=0.4)
        
        ax.set_xlabel('Start Block', fontsize=13)
        ax.set_ylabel('Disruption Rate (%)', fontsize=13)
        ax.set_title(f'Hairpin Disruption Rate vs Block Position (Window Size = {window_size})', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        
        max_start = NUM_BLOCKS - int(window_size)
        ax.set_xlim(-1, max_start + 1)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'disruption_rate_ws{window_size}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")


def plot_combined_summary(
    results_df: pd.DataFrame,
    output_dir: str,
):
    """
    Create a combined summary plot with all window sizes stacked.
    Shows distance change with different magnitude lines.
    """
    interventions = results_df[results_df['intervention_config'] != 'baseline']
    
    window_sizes = sorted([ws for ws in interventions['window_size'].unique() if ws > 0])
    
    fig, axes = plt.subplots(len(window_sizes), 1, figsize=(14, 5 * len(window_sizes)))
    # plt.subplots returns a bare Axes (not an array) when nrows=1; wrap it in a list so
    # axes[ws_idx] indexing below works uniformly regardless of how many window sizes there are
    if len(window_sizes) == 1:
        axes = [axes]
    
    magnitudes = sorted(interventions['magnitude'].unique())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(magnitudes)))
    
    for ws_idx, window_size in enumerate(window_sizes):
        ws_data = interventions[interventions['window_size'] == window_size]
        
        ax = axes[ws_idx]
        for mag, color in zip(magnitudes, colors):
            mag_data = ws_data[ws_data['magnitude'] == mag]
            mean_change = mag_data.groupby('window_start')['dist_change'].mean()
            ax.plot(mean_change.index, mean_change.values, 'o-', 
                    color=color, linewidth=2, markersize=3, 
                    label=f'mag={mag}', alpha=0.8)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=16, color='gray', linestyle=':', alpha=0.4)
        ax.axvline(x=32, color='gray', linestyle=':', alpha=0.4)
        ax.set_xlabel('Start Block', fontsize=11)
        ax.set_ylabel('Mean Distance Change (Å)', fontsize=11)
        ax.set_title(f'Distance Change (window_size={window_size})', fontsize=12)
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(-1, NUM_BLOCKS - int(window_size) + 1)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'combined_dist_change.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_heatmap(
    results_df: pd.DataFrame,
    output_dir: str,
):
    """
    Create heatmaps of distance change (window_start x magnitude) for each window size.
    """
    interventions = results_df[results_df['intervention_config'] != 'baseline']
    
    window_sizes = sorted([ws for ws in interventions['window_size'].unique() if ws > 0])
    
    for window_size in window_sizes:
        ws_data = interventions[interventions['window_size'] == window_size]
        
        # Pivot to create heatmap data: rows=magnitude, columns=window_start, cells=mean
        # dist_change -- exactly the 2D grid imshow needs below.
        pivot = ws_data.pivot_table(
            values='dist_change', 
            index='magnitude', 
            columns='window_start', 
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(14, 4))
        
        # Symmetric vmin/vmax around 0 so the diverging RdBu_r colormap is centered at
        # zero change (equal-magnitude positive/negative changes get equal color intensity)
        vmax = np.abs(pivot.values).max()
        im = ax.imshow(pivot.values, aspect='auto', cmap='RdBu_r',  # aspect='auto': let cells be non-square to fill figsize
                       vmin=-vmax, vmax=vmax)
        
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f'{m}' for m in pivot.index])
        
        # Show every 5th block on x-axis (avoids overcrowded tick labels across up to ~48 columns)
        xtick_positions = range(0, len(pivot.columns), 5)
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels([int(pivot.columns[i]) for i in xtick_positions])
        
        ax.set_xlabel('Start Block', fontsize=12)
        ax.set_ylabel('Magnitude', fontsize=12)
        ax.set_title(f'Distance Change Heatmap (Window Size = {window_size})\n(red = increased distance = disruption)', fontsize=13)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mean Distance Change (Å)', fontsize=11)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'heatmap_dist_ws{window_size}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")


def plot_by_charge_mode(
    results_df: pd.DataFrame,
    output_dir: str,
):
    """
    Create plots comparing different charge modes if multiple are present.
    """
    interventions = results_df[results_df['intervention_config'] != 'baseline']
    
    charge_modes = sorted(interventions['charge_mode'].unique())
    if len(charge_modes) < 2:
        # Nothing to compare with only a single charge mode in the results
        print("Only one charge mode, skipping charge mode comparison plots")
        return
    
    window_sizes = sorted([ws for ws in interventions['window_size'].unique() if ws > 0])
    
    charge_colors = {
        'both_positive': 'red',
        'both_negative': 'blue',
        'opposite': 'green',
    }
    
    for window_size in window_sizes:
        ws_data = interventions[interventions['window_size'] == window_size]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for charge_mode in charge_modes:
            cm_data = ws_data[ws_data['charge_mode'] == charge_mode]
            
            # Average across magnitudes
            mean_change = cm_data.groupby('window_start')['dist_change'].mean()
            std_change = cm_data.groupby('window_start')['dist_change'].std()
            
            color = charge_colors.get(charge_mode, 'gray')
            ax.plot(mean_change.index, mean_change.values, 'o-', 
                    color=color, linewidth=2, markersize=4, 
                    label=charge_mode, alpha=0.8)
            ax.fill_between(mean_change.index, 
                            mean_change.values - std_change.values,
                            mean_change.values + std_change.values,
                            color=color, alpha=0.1)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=16, color='gray', linestyle=':', alpha=0.4)
        ax.axvline(x=32, color='gray', linestyle=':', alpha=0.4)
        
        ax.set_xlabel('Start Block', fontsize=13)
        ax.set_ylabel('Mean Distance Change (Å)', fontsize=13)
        ax.set_title(f'Distance Change by Charge Mode (Window Size = {window_size})', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        
        max_start = NUM_BLOCKS - int(window_size)
        ax.set_xlim(-1, max_start + 1)
        
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'charge_mode_comparison_ws{window_size}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")


def main():
    """Parse CLI args, load disruption (and induction) results, and generate the comparison plot."""
    parser = argparse.ArgumentParser(description='Plot hairpin disruption results')
    # Default --results/--induction_results paths are absolute, machine-specific locations
    # from the original run; override them via CLI on any other machine.
    parser.add_argument('--results', type=str, default='/share/NFS/u/kevin/ProteinFolding-1/final_final_seperator_one_two_unfiltered_05/hairpin_disruption_results.parquet',
                        help='Path to results file (parquet or csv)')
    parser.add_argument('--output', type=str, default='charge_disruption_plots',
                        help='Output directory for plots')
    parser.add_argument('--induction_results', type=str,
                    default='/share/NFS/u/kevin/ProteinFolding-1/final_charge_three_ws_fifteen/hairpin_induction_results.parquet')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Loading results from {args.results}...")
    results_df = load_results(args.results)

# Load induction data for pos_neg / neg_pos
    induction_df = load_results(args.induction_results)

    # Currently the only plot actually generated by this script (see the commented-out
    # calls to the other plot_* functions further down)
    plot_charge_mode_comparison_ws10(results_df, induction_df, args.output)
    print(f"Loaded {len(results_df)} rows")
    print(f"Columns: {results_df.columns.tolist()}")
    
    # Print summary
    baseline = results_df[results_df['intervention_config'] == 'baseline']
    interventions = results_df[results_df['intervention_config'] != 'baseline']
    
    print(f"\nBaseline cases: {len(baseline)}")
    print(f"Intervention rows: {len(interventions)}")
    
    if len(interventions) > 0:
        print(f"Window sizes: {sorted([ws for ws in interventions['window_size'].unique() if ws > 0])}")
        print(f"Magnitudes: {sorted(interventions['magnitude'].unique())}")
        print(f"Charge modes: {sorted(interventions['charge_mode'].unique())}")
        print(f"Window starts: {sorted(interventions['window_start'].unique())[:5]}...{sorted(interventions['window_start'].unique())[-5:]}")
    
    print("\nGenerating plots...")
    
    # # Generate all plots
    # NOTE: possible bug -- this commented-out call passes only 2 positional args
    # (results_df, args.output), but plot_charge_mode_comparison_ws10 now requires 3
    # (results_df, induction_df, output_dir); uncommenting as-is would bind args.output to
    # the induction_df parameter and then raise a TypeError for the missing output_dir.
    # Likely stale from before the function was changed to also plot induction data.
    # plot_charge_mode_comparison_ws10(results_df, args.output)
    # plot_dist_change_by_window_size(results_df, args.output)
    # plot_hbond_change_by_window_size(results_df, args.output)
    # plot_disruption_rate_by_window_size(results_df, args.output)  # also see NOTE on this function re: 'hairpin_found'
    # plot_combined_summary(results_df, args.output)
    # plot_heatmap(results_df, args.output)
    # plot_by_charge_mode(results_df, args.output)
    
    print(f"\nAll plots saved to {args.output}/")


if __name__ == '__main__':
    main()