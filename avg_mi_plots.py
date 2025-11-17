#!/usr/bin/env python3
# Averaged MI contribution plots from per-file details CSVs produced earlier.
# python avg_mi_plots.py --details_dir mi_details --out_dir avg_plots [--summary mi_summary.csv]
import argparse, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def load_and_normalize(details_path: Path) -> pd.DataFrame:
    df = pd.read_csv(details_path)
    if {"stage","type","length","mi_contrib"}.issubset(df.columns) is False:
        raise ValueError(f"Missing columns in {details_path}")
    df["file"] = details_path.stem
    tot = df["mi_contrib"].sum()
    if tot > 0:
        df["mi_contrib_norm"] = df["mi_contrib"] / tot   # per-file normalization
    else:
        df["mi_contrib_norm"] = 0.0
    return df[["file","stage","type","length","mi_contrib_norm"]]

def pivot_stage_length(df: pd.DataFrame, which="both") -> pd.DataFrame:
    d = df if which=="both" else df[df["type"]==which]
    g = d.groupby(["file","stage","length"], as_index=False)["mi_contrib_norm"].sum()
    # average across files
    avg = g.groupby(["stage","length"], as_index=False)["mi_contrib_norm"].mean()
    pv = avg.pivot(index="stage", columns="length", values="mi_contrib_norm").fillna(0.0)
    pv = pv.reindex(sorted(pv.columns), axis=1)
    
    # Reorder stages in a logical sleep cycle order
    desired_stage_order = [
        'W-certain', 'transition-W-N1', 'N1-certain', 'transition-N1-N2',
        'N2-certain', 'transition-N2-N3', 'N3-certain', 'transition-N3-N2',
        'transition-N2-R', 'R-certain', 'transition-R-N2', 'transition-N1-W',
        'transition-N2-W', 'transition-R-W'
    ]
    # Only reindex with stages that actually exist in the data
    available_stages = [s for s in desired_stage_order if s in pv.index]
    pv = pv.reindex(available_stages)
    
    return pv

def bar_stage(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["file","stage"], as_index=False)["mi_contrib_norm"].sum()
    avg = g.groupby("stage", as_index=False)["mi_contrib_norm"].mean()
    return avg

def bar_length_by_type(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["file","type","length"], as_index=False)["mi_contrib_norm"].sum()
    avg = g.groupby(["type","length"], as_index=False)["mi_contrib_norm"].mean()
    return avg

def plot_heatmap(pv: pd.DataFrame, title: str, out: Path):
    fig, ax = plt.subplots(figsize=(10,5))
    im = ax.imshow(pv.to_numpy(), aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(pv.shape[1])); ax.set_xticklabels(list(pv.columns))
    ax.set_yticks(np.arange(pv.shape[0])); ax.set_yticklabels(list(pv.index))
    ax.set_xlabel("Run length"); ax.set_ylabel("Stage"); ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax); cbar.set_label("Avg MI contribution (fraction)")
    fig.tight_layout(); fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)

def plot_combined_heatmaps(pv_D: pd.DataFrame, pv_A: pd.DataFrame, pv_both: pd.DataFrame, out: Path):
    """Plot all three heatmaps in a single figure with shared dimensions."""
    # Set larger font sizes for publication quality
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'figure.titlesize': 20
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 5))
    
    # Find global min/max for consistent colorbar across ALL heatmaps
    vmin = min(pv_D.min().min(), pv_A.min().min(), pv_both.min().min())
    vmax = max(pv_D.max().max(), pv_A.max().max(), pv_both.max().max())
    
    # Force vmin to 0 for better visual consistency (contributions are non-negative)
    vmin = 0.0
    
    print(f"Shared color scale: vmin={vmin:.4f}, vmax={vmax:.4f}")
    
    heatmaps_data = [
        (pv_D, "D only", axes[0], True),
        (pv_A, "A only", axes[1], False),
        (pv_both, "A+D", axes[2], False)
    ]
    
    # Store the last image for colorbar (all use same scale, so any will work)
    last_im = None
    for pv, title, ax, show_labels in heatmaps_data:
        im = ax.imshow(pv.to_numpy(), aspect="auto", interpolation="nearest", 
                      vmin=vmin, vmax=vmax, cmap='viridis')
        last_im = im  # Keep reference for colorbar
        ax.set_xticks(np.arange(pv.shape[1]))
        ax.set_yticks(np.arange(pv.shape[0]))
        
        # All heatmaps get x-axis labels (run lengths)
        ax.set_xticklabels(list(pv.columns), fontsize=16)
        ax.set_xlabel("Run length", fontsize=18)
        
        if show_labels:
            # Only first heatmap gets y-axis labels (sleep stages)
            ax.set_yticklabels(list(pv.index), fontsize=16)
            ax.set_ylabel("Stage", fontsize=18)
        else:
            # Other heatmaps: no y-axis labels
            ax.set_yticklabels([])
        
        ax.set_title(title, fontsize=20)
    
    # Add single colorbar for all three heatmaps using the shared scale
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label("Avg MI contribution (fraction)", fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Reset to default font sizes
    plt.rcParams.update(plt.rcParamsDefault)

def plot_stage_bars(avg_stage: pd.DataFrame, title: str, out: Path):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(avg_stage["stage"], avg_stage["mi_contrib_norm"])
    ax.set_ylabel("Avg total MI fraction"); ax.set_xlabel("Stage"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)

def plot_len_bars(avg_len_type: pd.DataFrame, title: str, out: Path):
    lengths = sorted(avg_len_type["length"].unique())
    A = avg_len_type.query("type=='A'").set_index("length").reindex(lengths)["mi_contrib_norm"].fillna(0.0)
    D = avg_len_type.query("type=='D'").set_index("length").reindex(lengths)["mi_contrib_norm"].fillna(0.0)
    x = np.arange(len(lengths)); w=0.4
    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(x-w/2, A.values, width=w, label="A")
    ax.bar(x+w/2, D.values, width=w, label="D")
    ax.set_xticks(x); ax.set_xticklabels(lengths)
    ax.set_xlabel("Run length"); ax.set_ylabel("Avg MI fraction"); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Average MI contribution plots over many recordings.")
    ap.add_argument("--details_dir", required=True, help="Folder with *_mi_details.csv files")
    ap.add_argument("--out_dir", required=True, help="Output folder for averaged plots")
    ap.add_argument("--glob", default="*_mi_details.csv", help="Glob for detail files (default: *_mi_details.csv)")
    ap.add_argument("--summary", default=None, help="Optional: path to mi_summary.csv for statistical annotations")
    ap.add_argument("--combined", action="store_true", help="Plot all heatmaps in a single figure instead of separate files")
    args = ap.parse_args()

    details_dir = Path(args.details_dir); out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(details_dir.glob(args.glob))
    if not files:
        raise SystemExit("No detail CSVs found.")

    all_df = pd.concat([load_and_normalize(p) for p in files], ignore_index=True)

    # Load summary statistics if provided
    summary_df = None
    if args.summary:
        summary_path = Path(args.summary)
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            print(f"\nLoaded summary statistics from {args.summary}")
            print(f"Files with p < 0.05: {(summary_df['p_value'] < 0.05).sum()}/{len(summary_df)}")
            if 'mi_corrected_bits' in summary_df.columns:
                valid = summary_df.dropna(subset=['mi_corrected_bits'])
                print(f"Mean MI (corrected, bits): {valid['mi_corrected_bits'].mean():.4f} ± {valid['mi_corrected_bits'].std():.4f}")
                if 'mi_normalized' in summary_df.columns:
                    print(f"Mean MI (normalized): {valid['mi_normalized'].mean():.4f} ± {valid['mi_normalized'].std():.4f}")

    # Averaged heatmaps
    pv_both = pivot_stage_length(all_df, "both")
    pv_A    = pivot_stage_length(all_df, "A")
    pv_D    = pivot_stage_length(all_df, "D")
    
    if args.combined:
        # Plot all three heatmaps in a single figure (D, A, both)
        plot_combined_heatmaps(pv_D, pv_A, pv_both, out_dir/"avg_mi_heatmap_combined.png")
    else:
        # Plot each heatmap in a separate file (original behavior)
        plot_heatmap(pv_both, "Averaged MI contributions (A+D)", out_dir/"avg_mi_heatmap_both.png")
        plot_heatmap(pv_A,    "Averaged MI contributions (A only)", out_dir/"avg_mi_heatmap_A.png")
        plot_heatmap(pv_D,    "Averaged MI contributions (D only)", out_dir/"avg_mi_heatmap_D.png")

    # Averaged barplots
    avg_stage = bar_stage(all_df)
    plot_stage_bars(avg_stage, "Averaged total MI by stage", out_dir/"avg_mi_by_stage.png")

    avg_len_type = bar_length_by_type(all_df)
    plot_len_bars(avg_len_type, "Averaged MI by length (A vs D)", out_dir/"avg_mi_by_length_type.png")

    # Also save the aggregated tables
    pv_both.to_csv(out_dir/"avg_heatmap_both_table.csv")
    pv_A.to_csv(out_dir/"avg_heatmap_A_table.csv")
    pv_D.to_csv(out_dir/"avg_heatmap_D_table.csv")
    avg_stage.to_csv(out_dir/"avg_stage_totals.csv", index=False)
    avg_len_type.to_csv(out_dir/"avg_length_by_type.csv", index=False)
    
    print(f"\n✓ Averaged plots and tables saved to {out_dir}")

if __name__ == "__main__":
    main()
