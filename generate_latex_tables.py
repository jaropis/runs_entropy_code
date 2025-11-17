"""
Generate LaTeX tables from MI summary CSV files.

Usage:
    python generate_latex_tables.py
    
Outputs:
    - table1_summary_stats.tex (aggregate statistics for main text)
    - table2_per_subject.tex (detailed per-subject results for supplement)
"""

import pandas as pd
from pathlib import Path

def generate_summary_table(stats_csv: Path, output_tex: Path):
    """Generate Table 1: Summary statistics across all subjects."""
    
    df = pd.read_csv(stats_csv)
    
    # Extract key statistics
    mi_mean = df.loc[df['metric'] == 'MI_corrected (bits)', 'mean'].values[0]
    mi_std = df.loc[df['metric'] == 'MI_corrected (bits)', 'std'].values[0]
    mi_median = df.loc[df['metric'] == 'MI_corrected (bits)', 'median'].values[0]
    mi_q25 = df.loc[df['metric'] == 'MI_corrected (bits)', 'q25'].values[0]
    mi_q75 = df.loc[df['metric'] == 'MI_corrected (bits)', 'q75'].values[0]
    mi_cv = df.loc[df['metric'] == 'MI_corrected (bits)', 'cv'].values[0]
    mi_min = df.loc[df['metric'] == 'MI_corrected (bits)', 'min'].values[0]
    mi_max = df.loc[df['metric'] == 'MI_corrected (bits)', 'max'].values[0]
    
    norm_mean = df.loc[df['metric'] == 'MI_normalized', 'mean'].values[0]
    norm_std = df.loc[df['metric'] == 'MI_normalized', 'std'].values[0]
    norm_median = df.loc[df['metric'] == 'MI_normalized', 'median'].values[0]
    norm_q25 = df.loc[df['metric'] == 'MI_normalized', 'q25'].values[0]
    norm_q75 = df.loc[df['metric'] == 'MI_normalized', 'q75'].values[0]
    norm_cv = df.loc[df['metric'] == 'MI_normalized', 'cv'].values[0]
    
    n_total = int(df.loc[df['metric'] == 'MI_corrected (bits)', 'n'].values[0])
    n_sig = int(df.loc[df['metric'] == 'p_value', 'n_significant_p<0.05'].values[0])
    pct_sig = 100 * n_sig / n_total
    
    p_median = df.loc[df['metric'] == 'p_value', 'median'].values[0]
    p_q25 = df.loc[df['metric'] == 'p_value', 'q25'].values[0]
    p_q75 = df.loc[df['metric'] == 'p_value', 'q75'].values[0]
    
    # Generate LaTeX table
    latex = r"""\begin{table}[htbp]
\centering
\caption{Summary statistics for mutual information between HRA run categories and sleep stages across 31 subjects.}
\label{tab:mi_summary}
\begin{tabular}{lcc}
\hline
\textbf{Metric} & \textbf{MI Corrected (bits)} & \textbf{MI Normalized} \\
\hline
Mean $\pm$ SD & """ + f"{mi_mean:.4f} $\\pm$ {mi_std:.4f}" + r""" & """ + f"{norm_mean:.4f} $\\pm$ {norm_std:.4f}" + r""" \\
Median [IQR] & """ + f"{mi_median:.4f} [{mi_q25:.4f}--{mi_q75:.4f}]" + r""" & """ + f"{norm_median:.4f} [{norm_q25:.4f}--{norm_q75:.4f}]" + r""" \\
Coefficient of variation & """ + f"{mi_cv:.2f}" + r""" & """ + f"{norm_cv:.2f}" + r""" \\
Range (min--max) & """ + f"{mi_min:.4f}--{mi_max:.4f}" + r""" & --- \\
\hline
\multicolumn{3}{l}{\textbf{Significance Testing}} \\
\hline
Subjects with $p < 0.05$ & \multicolumn{2}{c}{""" + f"{n_sig}/{n_total} ({pct_sig:.1f}\\%)" + r"""} \\
Median $p$-value [IQR] & \multicolumn{2}{c}{""" + f"{p_median:.3f} [{p_q25:.3f}--{p_q75:.3f}]" + r"""} \\
\hline
\end{tabular}
\end{table}"""
    
    # Write to file
    with open(output_tex, 'w') as f:
        f.write(latex)
    
    print(f"Summary table written to: {output_tex}")


def generate_per_subject_table(summary_csv: Path, output_tex: Path):
    """Generate Table 2: Per-subject results for supplementary material."""
    
    df = pd.read_csv(summary_csv)
    
    # Sort by MI corrected (descending)
    df = df.sort_values('mi_corrected_bits', ascending=False)
    
    # Start LaTeX table
    latex = r"""\begin{table}[htbp]
\centering
\caption{Per-subject mutual information results. Subjects are sorted by MI (corrected) in descending order. Asterisks denote significance: *** $p < 0.001$, ** $p < 0.01$, * $p < 0.05$.}
\label{tab:mi_per_subject}
\small
\begin{tabular}{lccccr}
\hline
\textbf{Subject} & \textbf{MI (bits)} & \textbf{95\% CI} & \textbf{$p$-value} & \textbf{MI norm.} & \textbf{$N$} \\
\hline
"""
    
    # Add rows
    for _, row in df.iterrows():
        # Clean up filename
        subject = row['file'].replace('.xlsx', '').replace('_', ' ')
        
        # Format MI with CI
        mi = row['mi_corrected_bits']
        ci_lower = row['ci_lower_bits']
        ci_upper = row['ci_upper_bits']
        
        # Format p-value with significance stars
        p = row['p_value']
        if p < 0.001:
            p_str = f"{p:.3f}***"
        elif p < 0.01:
            p_str = f"{p:.3f}**"
        elif p < 0.05:
            p_str = f"{p:.3f}*"
        else:
            p_str = f"{p:.3f}"
        
        # Format normalized MI
        mi_norm = row['mi_normalized']
        
        # Number of windows
        n_windows = int(row['n_windows_used'])
        
        latex += f"{subject} & {mi:.4f} & [{ci_lower:.4f}, {ci_upper:.4f}] & {p_str} & {mi_norm:.4f} & {n_windows} \\\\\n"
    
    # Close table
    latex += r"""\hline
\end{tabular}
\end{table}"""
    
    # Write to file
    with open(output_tex, 'w') as f:
        f.write(latex)
    
    print(f"Per-subject table written to: {output_tex}")


def main():
    """Generate both LaTeX tables."""
    
    # File paths
    stats_csv = Path("mi_summary_statistics.csv")
    summary_csv = Path("mi_summary.csv")
    
    table1_tex = Path("table1_summary_stats.tex")
    table2_tex = Path("table2_per_subject.tex")
    
    # Check if input files exist
    if not stats_csv.exists():
        print(f"Error: {stats_csv} not found!")
        return
    if not summary_csv.exists():
        print(f"Error: {summary_csv} not found!")
        return
    
    # Generate tables
    print("Generating LaTeX tables...")
    generate_summary_table(stats_csv, table1_tex)
    generate_per_subject_table(summary_csv, table2_tex)
    print("\nDone! You can now \\input{} these files in your LaTeX document.")


if __name__ == "__main__":
    main()