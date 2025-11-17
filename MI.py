"""
Mutual Information Analysis for Heart Rate Acceleration (HRA) Runs vs. Sleep Stages

This module computes mutual information between HRA run categories (type + length) 
and sleep stages, with sophisticated statistical enhancements:

1. **Miller-Madow Bias Correction**: Corrects finite-sample bias in plug-in MI estimator
   MI_corrected = MI_plugin - (K-1)/(2N*ln(2))
   where K = number of non-zero cells, N = total beats

2. **Bootstrap Confidence Intervals**: Subject-level resampling (default 1000 iterations)
   to quantify uncertainty in MI estimates (95% CI by default)

3. **Permutation Testing**: Within-subject shuffling of stage labels while preserving
   stage counts to test significance against null hypothesis (default 1000 permutations)

4. **Normalized MI**: MI / min(H(runs), H(stages)) for interpretable [0,1] scale showing
   proportion of uncertainty explained

5. **Aggregate Statistics**: Cross-subject mean, SD, coefficient of variation (CV),
   median, and interquartile range (IQR)

All MI values are computed with natural logarithms (nats) and converted to bits 
(divide by ln(2)) for reporting.

Usage:
    python MI.py
    # Processes all .xlsx/.csv files in 'data/' directory
    # Outputs: mi_summary.csv (per-subject) and mi_summary_statistics.csv (aggregate)
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import matplotlib.pyplot as plt
from scipy import stats

# ---------- Helpers to detect columns ----------

def _find_run_columns_df(
    df: pd.DataFrame,
    ar_regex: str = r'^AR[_ ]?(\d+)$',
    dr_regex: str = r'^DR[_ ]?(\d+)$',
) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Detect acceleration (AR) and deceleration (DR) run-length columns.
    Regex must capture the run length as group(1).
    Returns two dicts: {length -> column_name}.
    """
    ar_cols: Dict[int, str] = {}
    dr_cols: Dict[int, str] = {}
    for col in df.columns:
        s = str(col)
        m = re.match(ar_regex, s)
        if m:
            ar_cols[int(m.group(1))] = col
            continue
        m = re.match(dr_regex, s)
        if m:
            dr_cols[int(m.group(1))] = col
    if not ar_cols and not dr_cols:
        raise ValueError("No AR/DR columns found. Adjust ar_regex/dr_regex to your headers.")
    return dict(sorted(ar_cols.items())), dict(sorted(dr_cols.items()))


def _find_stage_columns_df(
    df: pd.DataFrame,
    stage_prefixes: Tuple[str, ...] = (
        "W-certain",
        "N2-certain", 
        "transition-N1-N2", 
        "transition-W-N1",
        "transition-N1-W", 
        "transition-N2-W",
        "transition-R-W",  
        "N1-certain", 
        "transition-N2-N3", 
        "N3-certain", 
        "transition-N3-N2", 
        "transition-N2-R", 
        "transition-N3-R", 
        "transition-R-N2", 
        "R-certain"),
    require_binary: bool = True
) -> List[str]:
    """
    Detect stage/transition columns by prefix (e.g., 'Stage-W', 'Stage-N2', 'Stage-R').
    Optionally require that they are binary 0/1 columns.
    """
    cand = [c for c in df.columns if any(str(c).startswith(p) for p in stage_prefixes)]
    if require_binary:
        def is_binary(col):
            vals = pd.unique(df[col].dropna())
            return set(vals).issubset({0, 1}) and len(vals) <= 2
        cand = [c for c in cand if is_binary(c)]
    if not cand:
        # fallback: any binary column
        for c in df.columns:
            vals = pd.unique(df[c].dropna())
            if set(vals).issubset({0, 1}) and len(vals) <= 2:
                cand.append(c)
    if not cand:
        raise ValueError("No stage columns detected. Check prefixes or ensure 0/1 coding.")
    return sorted(cand)


# ---------- Core MI from a single DataFrame ----------

def mutual_information_run_category_vs_stage_df(
    df: pd.DataFrame,
    ar_regex: str = r'^AR[_ ]?(\d+)$',
    dr_regex: str = r'^DR[_ ]?(\d+)$',
    stage_prefixes: Tuple[str, ...] = ("Stage_", "Stage-", "Stage "),
    drop_ambiguous_rows: bool = True,
    eps: float = 0.0
) -> Tuple[float, pd.DataFrame, Dict]:
    """
    Compute MI between run category (type+length) and stage from a single DataFrame.

    Uses per-beat proportions p_{i,k,s} built from run counts * length.
    Returns:
      - mi (float): I((K,L);S) in nats (uncorrected plug-in). (Divide by np.log(2) for bits.)
      - details (DataFrame): per-stage/per-category contributions and marginals.
      - metadata (Dict): additional info like total_beats, K (non-zero cells), for bias correction
    """
    # Detect columns
    ar_cols, dr_cols = _find_run_columns_df(df, ar_regex, dr_regex)
    stage_cols = _find_stage_columns_df(df, stage_prefixes, require_binary=True)

    # Keep only windows with exactly one active stage (unless told otherwise)
    stage_matrix = df[stage_cols].to_numpy()
    active_counts = stage_matrix.sum(axis=1)
    if drop_ambiguous_rows:
        mask = active_counts == 1
        df = df.loc[mask].copy()
        stage_matrix = stage_matrix[mask]
    else:
        if not np.all(active_counts == 1):
            raise ValueError("Some rows have 0 or >1 active stages. "
                             "Set drop_ambiguous_rows=True to drop them.")

    # Row-wise stage labels
    stage_labels = [stage_cols[idx] for idx in stage_matrix.argmax(axis=1)]
    df = df.assign(__stage__=stage_labels)

    # Aggregate beats contributed by each (length, type) within each stage
    rows = []
    for s in sorted(df["__stage__"].unique()):
        df_s = df.loc[df["__stage__"] == s]
        for length, col in ar_cols.items():
            beats = float(df_s[col].sum()) * length
            rows.append((s, "A", length, beats))
        for length, col in dr_cols.items():
            beats = float(df_s[col].sum()) * length
            rows.append((s, "D", length, beats))

    details = pd.DataFrame(rows, columns=["stage", "type", "length", "beats"])

    total_beats = details["beats"].sum()
    if total_beats <= 0:
        raise ValueError("Total beats from AR/DR runs is zero. Check your inputs.")
    details["p_i_k_s"] = details["beats"] / total_beats

    # Marginals
    p_i_k = details.groupby(["type", "length"], as_index=False)["p_i_k_s"] \
                   .sum().rename(columns={"p_i_k_s": "p_i_k"})
    p_s = details.groupby("stage", as_index=False)["p_i_k_s"] \
                 .sum().rename(columns={"p_i_k_s": "p_s"})
    details = details.merge(p_i_k, on=["type", "length"], how="left") \
                     .merge(p_s, on="stage", how="left")

    # MI sum p_{i,k,s} * log( p_{i,k,s} / (p_{i,k} * p_s) )
    pis = details["p_i_k_s"].to_numpy()
    pik = details["p_i_k"].to_numpy()
    ps = details["p_s"].to_numpy()

    if eps > 0.0:
        pis = pis + eps
        pik = pik + eps
        ps = ps + eps
        pis = pis / pis.sum()

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(pis > 0, pis / (pik * ps), 1.0)   # safe when pis==0
        mi_contrib = np.where(pis > 0, pis * np.log(ratio), 0.0)

    details["mi_contrib"] = mi_contrib
    mi = float(mi_contrib.sum())
    
    # Count non-zero cells for Miller-Madow correction
    K = int((pis > 0).sum())
    
    # Compute entropies for normalized MI (using unique marginal values to avoid duplicates)
    # Get unique marginal probabilities
    p_i_k_unique = details.groupby(["type", "length"], as_index=False)["p_i_k"].first()["p_i_k"].to_numpy()
    p_s_unique = details.groupby("stage", as_index=False)["p_s"].first()["p_s"].to_numpy()
    
    # Compute entropies safely
    with np.errstate(divide='ignore', invalid='ignore'):
        H_runs = -np.sum(np.where(p_i_k_unique > 0, p_i_k_unique * np.log(p_i_k_unique), 0.0))
        H_stages = -np.sum(np.where(p_s_unique > 0, p_s_unique * np.log(p_s_unique), 0.0))
    
    metadata = {
        'total_beats': int(total_beats),
        'K': K,
        'H_runs': float(H_runs),
        'H_stages': float(H_stages)
    }
    
    return mi, details, metadata


def bootstrap_mi_ci(
    df: pd.DataFrame,
    ar_regex: str = r'^AR[_ ]?(\d+)$',
    dr_regex: str = r'^DR[_ ]?(\d+)$',
    stage_prefixes: Tuple[str, ...] = ("Stage_", "Stage-", "Stage "),
    drop_ambiguous_rows: bool = True,
    eps: float = 0.0,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence intervals for MI by resampling rows (windows) with replacement.
    
    Returns:
      - ci_lower (float): lower bound of CI in nats
      - ci_upper (float): upper bound of CI in nats
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_rows = len(df)
    mi_samples = []
    
    for _ in range(n_bootstrap):
        # Resample rows with replacement
        idx = np.random.choice(n_rows, size=n_rows, replace=True)
        df_boot = df.iloc[idx].copy()
        
        try:
            mi_boot, _, _ = mutual_information_run_category_vs_stage_df(
                df_boot, ar_regex, dr_regex, stage_prefixes, drop_ambiguous_rows, eps
            )
            mi_samples.append(mi_boot)
        except (ValueError, ZeroDivisionError):
            # Skip failed bootstrap samples (e.g., if all one stage)
            continue
    
    if len(mi_samples) < 10:
        # Not enough successful samples
        return np.nan, np.nan
    
    mi_samples = np.array(mi_samples)
    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(mi_samples, 100 * alpha / 2))
    ci_upper = float(np.percentile(mi_samples, 100 * (1 - alpha / 2)))
    
    return ci_lower, ci_upper


def permutation_test_mi(
    df: pd.DataFrame,
    ar_regex: str = r'^AR[_ ]?(\d+)$',
    dr_regex: str = r'^DR[_ ]?(\d+)$',
    stage_prefixes: Tuple[str, ...] = ("Stage_", "Stage-", "Stage "),
    drop_ambiguous_rows: bool = True,
    eps: float = 0.0,
    n_permutations: int = 1000,
    random_seed: Optional[int] = None
) -> Tuple[float, np.ndarray]:
    """
    Perform within-subject permutation test for MI significance.
    
    Stage labels are randomly shuffled within the subject while preserving stage counts.
    
    Returns:
      - p_value (float): proportion of null MI >= observed MI
      - null_distribution (array): MI values from permuted data
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Compute observed MI
    try:
        mi_obs, _, _ = mutual_information_run_category_vs_stage_df(
            df, ar_regex, dr_regex, stage_prefixes, drop_ambiguous_rows, eps
        )
    except (ValueError, ZeroDivisionError):
        return np.nan, np.array([])
    
    # Detect stage columns for permutation
    stage_cols = _find_stage_columns_df(df, stage_prefixes, require_binary=True)
    
    null_mis = []
    for _ in range(n_permutations):
        df_perm = df.copy()
        
        # Shuffle stage assignments (permute which stage column is active for each row)
        # while preserving the total count of each stage
        stage_matrix = df[stage_cols].to_numpy()
        active_counts = stage_matrix.sum(axis=1)
        
        # Only permute unambiguous rows
        if drop_ambiguous_rows:
            mask = active_counts == 1
        else:
            mask = np.ones(len(df), dtype=bool)
        
        if mask.sum() == 0:
            continue
            
        # Get which stage is active for each unambiguous row
        stage_indices = stage_matrix[mask].argmax(axis=1)
        
        # Permute these indices
        perm_indices = np.random.permutation(stage_indices)
        
        # Create new stage matrix
        new_stage_matrix = np.zeros_like(stage_matrix[mask])
        for i, stage_idx in enumerate(perm_indices):
            new_stage_matrix[i, stage_idx] = 1
        
        # Replace in dataframe
        for j, col in enumerate(stage_cols):
            df_perm.loc[mask, col] = new_stage_matrix[:, j]
        
        try:
            mi_perm, _, _ = mutual_information_run_category_vs_stage_df(
                df_perm, ar_regex, dr_regex, stage_prefixes, drop_ambiguous_rows, eps
            )
            null_mis.append(mi_perm)
        except (ValueError, ZeroDivisionError):
            continue
    
    if len(null_mis) < 10:
        return np.nan, np.array([])
    
    null_mis = np.array(null_mis)
    p_value = float((null_mis >= mi_obs).sum() / len(null_mis))
    
    return p_value, null_mis


# ---------- Batch processing for many files ----------

def process_files_for_mi(
    paths: Iterable[Path],
    ar_regex: str = r'^AR[_ ]?(\d+)$',
    dr_regex: str = r'^DR[_ ]?(\d+)$',
    stage_prefixes: Tuple[str, ...] = ("Stage_", "Stage-", "Stage "),
    drop_ambiguous_rows: bool = True,
    eps: float = 0.0,
    save_details_dir: Optional[Path] = None,
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute MI((K,L);Stage) for many files with the same schema.
    Includes Miller-Madow bias correction, bootstrap CIs, and permutation testing.

    Args:
      paths: iterable of file paths (Excel or CSV).
      save_details_dir: if provided, saves per-file 'details' CSVs there.
      n_bootstrap: number of bootstrap samples for CI (default 1000).
      n_permutations: number of permutations for significance test (default 1000).
      confidence_level: confidence level for bootstrap CI (default 0.95).
      random_seed: random seed for reproducibility.

    Returns:
      summary DataFrame with columns:
        ['file', 'mi_nats', 'mi_bits', 'mi_corrected_nats', 'mi_corrected_bits',
         'mi_normalized', 'ci_lower_nats', 'ci_upper_nats', 'ci_lower_bits', 
         'ci_upper_bits', 'p_value', 'n_windows_used', 'n_windows_total', 
         'total_beats', 'K_cells', 'H_runs_nats', 'H_stages_nats']
      (plus an 'error' column if any file fails to load)
    """
    rows = []
    for p in paths:
        p = Path(p)
        try:
            # Try Excel, then CSV
            try:
                df = pd.read_excel(p)
            except Exception:
                df = pd.read_csv(p)

            # Count windows pre/post filtering for transparency
            stage_cols = _find_stage_columns_df(df, stage_prefixes, require_binary=True)
            stage_matrix = df[stage_cols].to_numpy()
            active_counts = stage_matrix.sum(axis=1)
            total = int(len(df))
            used = int((active_counts == 1).sum()) if drop_ambiguous_rows else total

            # Compute MI with metadata
            mi, details, metadata = mutual_information_run_category_vs_stage_df(
                df, ar_regex, dr_regex, stage_prefixes, drop_ambiguous_rows, eps
            )
            
            # Miller-Madow bias correction
            K = metadata['K']
            N = metadata['total_beats']
            mi_corrected = mi - (K - 1) / (2 * N * np.log(2))  # correction in bits, convert
            mi_corrected_nats = mi_corrected * np.log(2)  # back to nats
            
            # Normalized MI
            H_runs = metadata['H_runs']
            H_stages = metadata['H_stages']
            mi_normalized = mi / min(H_runs, H_stages) if min(H_runs, H_stages) > 0 else 0.0
            
            # Bootstrap CI
            print(f"  Computing bootstrap CI for {p.stem}...")
            ci_lower, ci_upper = bootstrap_mi_ci(
                df, ar_regex, dr_regex, stage_prefixes, drop_ambiguous_rows, eps,
                n_bootstrap, confidence_level, random_seed
            )
            
            # Permutation test
            print(f"  Computing permutation test for {p.stem}...")
            p_value, _ = permutation_test_mi(
                df, ar_regex, dr_regex, stage_prefixes, drop_ambiguous_rows, eps,
                n_permutations, random_seed
            )

            if save_details_dir is not None:
                save_details_dir.mkdir(parents=True, exist_ok=True)
                out_csv = save_details_dir / f"{p.stem}_mi_details.csv"
                details.to_csv(out_csv, index=False)

            # plotting - only if we have meaningful MI contributions
            if mi > 0 and not details.empty and details["mi_contrib"].sum() > 0:
                plot_all_for_file(details, Path("mi_plots"), p.stem)
            
            rows.append({
                "file": str(p.name),
                "mi_nats": mi,
                "mi_bits": mi / np.log(2.0),
                "mi_corrected_nats": mi_corrected_nats,
                "mi_corrected_bits": mi_corrected,
                "mi_normalized": mi_normalized,
                "ci_lower_nats": ci_lower,
                "ci_upper_nats": ci_upper,
                "ci_lower_bits": ci_lower / np.log(2.0) if not np.isnan(ci_lower) else np.nan,
                "ci_upper_bits": ci_upper / np.log(2.0) if not np.isnan(ci_upper) else np.nan,
                "p_value": p_value,
                "n_windows_used": used,
                "n_windows_total": total,
                "total_beats": N,
                "K_cells": K,
                "H_runs_nats": H_runs,
                "H_stages_nats": H_stages
            })

        except Exception as e:
            rows.append({
                "file": str(p.name),
                "mi_nats": np.nan,
                "mi_bits": np.nan,
                "mi_corrected_nats": np.nan,
                "mi_corrected_bits": np.nan,
                "mi_normalized": np.nan,
                "ci_lower_nats": np.nan,
                "ci_upper_nats": np.nan,
                "ci_lower_bits": np.nan,
                "ci_upper_bits": np.nan,
                "p_value": np.nan,
                "n_windows_used": np.nan,
                "n_windows_total": np.nan,
                "total_beats": np.nan,
                "K_cells": np.nan,
                "H_runs_nats": np.nan,
                "H_stages_nats": np.nan,
                "error": str(e)
            })
    return pd.DataFrame(rows)


def compute_summary_statistics(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute aggregate statistics across subjects from the summary DataFrame.
    
    Returns a DataFrame with summary statistics including mean, SD, CV, median, IQR, etc.
    """
    # Filter out rows with errors
    valid = summary_df.dropna(subset=['mi_bits'])
    
    if len(valid) == 0:
        return pd.DataFrame({"statistic": ["No valid data"]})
    
    stats_rows = []
    
    # MI (bits) - corrected
    mi_bits_corrected = valid['mi_corrected_bits'].dropna()
    if len(mi_bits_corrected) > 0:
        stats_rows.append({
            "metric": "MI_corrected (bits)",
            "mean": mi_bits_corrected.mean(),
            "std": mi_bits_corrected.std(),
            "cv": mi_bits_corrected.std() / mi_bits_corrected.mean() if mi_bits_corrected.mean() > 0 else np.nan,
            "median": mi_bits_corrected.median(),
            "q25": mi_bits_corrected.quantile(0.25),
            "q75": mi_bits_corrected.quantile(0.75),
            "min": mi_bits_corrected.min(),
            "max": mi_bits_corrected.max(),
            "n": len(mi_bits_corrected)
        })
    
    # MI normalized
    mi_norm = valid['mi_normalized'].dropna()
    if len(mi_norm) > 0:
        stats_rows.append({
            "metric": "MI_normalized",
            "mean": mi_norm.mean(),
            "std": mi_norm.std(),
            "cv": mi_norm.std() / mi_norm.mean() if mi_norm.mean() > 0 else np.nan,
            "median": mi_norm.median(),
            "q25": mi_norm.quantile(0.25),
            "q75": mi_norm.quantile(0.75),
            "min": mi_norm.min(),
            "max": mi_norm.max(),
            "n": len(mi_norm)
        })
    
    # P-values
    p_vals = valid['p_value'].dropna()
    if len(p_vals) > 0:
        n_significant = (p_vals < 0.05).sum()
        stats_rows.append({
            "metric": "p_value",
            "mean": p_vals.mean(),
            "std": p_vals.std(),
            "cv": np.nan,
            "median": p_vals.median(),
            "q25": p_vals.quantile(0.25),
            "q75": p_vals.quantile(0.75),
            "min": p_vals.min(),
            "max": p_vals.max(),
            "n": len(p_vals),
            "n_significant_p<0.05": n_significant
        })
    
    return pd.DataFrame(stats_rows)

# ---------- Visualization helper ----------
def _pivot_mi(details: pd.DataFrame, which: str = "both") -> pd.DataFrame:
    """
    Build a stage x length pivot of MI contributions.
    which: 'both' (sum A and D), 'A', or 'D'
    """
    d = details.copy()
    if which in ("A", "D"):
        d = d.loc[d["type"] == which]
    pv = d.groupby(["stage", "length"], as_index=False)["mi_contrib"].sum()
    pv = pv.pivot(index="stage", columns="length", values="mi_contrib")
    pv = pv.reindex(sorted(pv.columns), axis=1)
    return pv.fillna(0.0)

def plot_mi_heatmap(details: pd.DataFrame, which: str = "both",
                    title: str = None, save_path: Path = None):
    """
    Heatmap of MI contributions with rows=stage, cols=run length.
    """
    pv = _pivot_mi(details, which)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pv.to_numpy(), aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(pv.shape[1]))
    ax.set_xticklabels(list(pv.columns))
    ax.set_yticks(np.arange(pv.shape[0]))
    ax.set_yticklabels(list(pv.index))
    ax.set_xlabel("Run length")
    ax.set_ylabel("Stage")
    ax.set_title(title if title else f"MI contributions heatmap ({which})")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("MI contribution (nats)")
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)  # Close figure to prevent memory warnings
    return fig, ax

def plot_stage_mi_bars(details: pd.DataFrame, title: str = None,
                       save_path: Path = None):
    """
    Bar chart of total MI contribution per stage.
    """
    stage_sum = details.groupby("stage", as_index=False)["mi_contrib"].sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(stage_sum["stage"], stage_sum["mi_contrib"])
    ax.set_ylabel("Total MI contribution (nats)")
    ax.set_xlabel("Stage")
    ax.set_title(title if title else "Total MI contribution by stage")
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)  # Close figure to prevent memory warnings
    return fig, ax

def plot_length_mi_bars_by_type(details: pd.DataFrame, title: str = None,
                                save_path: Path = None):
    """
    Bar chart of MI contribution by run length, split by type (A vs D).
    """
    grp = details.groupby(["type", "length"], as_index=False)["mi_contrib"].sum()
    lengths = sorted(grp["length"].unique())
    A = grp.loc[grp["type"] == "A"].set_index("length").reindex(lengths)["mi_contrib"].fillna(0.0)
    D = grp.loc[grp["type"] == "D"].set_index("length").reindex(lengths)["mi_contrib"].fillna(0.0)

    x = np.arange(len(lengths))
    w = 0.4

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w/2, A.values, width=w, label="A")
    ax.bar(x + w/2, D.values, width=w, label="D")
    ax.set_xticks(x)
    ax.set_xticklabels(lengths)
    ax.set_xlabel("Run length")
    ax.set_ylabel("MI contribution (nats)")
    ax.set_title(title if title else "MI contribution by run length (A vs D)")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)  # Close figure to prevent memory warnings
    return fig, ax

def plot_all_for_file(details: pd.DataFrame, outdir: Path, file_stem: str):
    """
    Produce a standard set of plots for one recording.
    Saves PNGs to outdir.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    plot_mi_heatmap(details, "both", f"{file_stem} — MI contributions (A+D)",
                    save_path=outdir / f"{file_stem}_mi_heatmap_both.png")
    plot_mi_heatmap(details, "A", f"{file_stem} — MI contributions (A only)",
                    save_path=outdir / f"{file_stem}_mi_heatmap_A.png")
    plot_mi_heatmap(details, "D", f"{file_stem} — MI contributions (D only)",
                    save_path=outdir / f"{file_stem}_mi_heatmap_D.png")
    plot_stage_mi_bars(details, f"{file_stem} — MI by stage",
                       save_path=outdir / f"{file_stem}_mi_by_stage.png")
    plot_length_mi_bars_by_type(details, f"{file_stem} — MI by length (A vs D)",
                                save_path=outdir / f"{file_stem}_mi_by_length_type.png")

# ---------- Example usage ----------
if __name__ == "__main__":
    # 1) Glob a directory of recordings (both .xlsx and .csv supported)
    data_dir = Path("data")  
    files = list(data_dir.glob("*.xlsx")) + list(data_dir.glob("*.csv"))
    print(f"Found {len(files)} files.")
    
    # 2) Process all files and save per-file contribution tables
    # This now includes Miller-Madow correction, bootstrap CIs, and permutation tests
    print("\nProcessing files with statistical analyses...")
    summary = process_files_for_mi(
        files,
        # Adjust these if your headers differ:
        ar_regex=r'^AR[_ ]?(\d+)$',
        dr_regex=r'^DR[_ ]?(\d+)$',
        stage_prefixes=(
            "W-certain",
            "N2-certain", 
            "transition-N1-N2", 
            "transition-W-N1",
            "transition-N1-W", 
            "transition-N2-W",
            "transition-R-W",  
            "N1-certain", 
            "transition-N2-N3", 
            "N3-certain", 
            "transition-N3-N2", 
            "transition-N2-R", 
            "transition-N3-R", 
            "transition-R-N2", 
            "R-certain"
        ),
        drop_ambiguous_rows=True,
        eps=0.0,
        save_details_dir=Path("mi_details"),
        n_bootstrap=1000,
        n_permutations=1000,
        confidence_level=0.95,
        random_seed=42  # for reproducibility
    )

    # 3) Save summary for all patients
    summary.to_csv("mi_summary.csv", index=False)
    print("\n" + "="*80)
    print("INDIVIDUAL SUBJECT RESULTS:")
    print("="*80)
    print(summary.to_string())
    
    # 4) Compute and display aggregate statistics
    print("\n" + "="*80)
    print("AGGREGATE STATISTICS ACROSS SUBJECTS:")
    print("="*80)
    stats = compute_summary_statistics(summary)
    print(stats.to_string(index=False))
    stats.to_csv("mi_summary_statistics.csv", index=False)
    
    print(f"\n✓ Results saved to 'mi_summary.csv' and 'mi_summary_statistics.csv'")
