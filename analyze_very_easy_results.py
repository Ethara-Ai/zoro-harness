#!/usr/bin/env python3
"""
Very Easy Experiment Results Analyzer

Analyzes the results from Very Easy environment experiments and generates
summary statistics and comparison tables.

Usage:
    python analyze_very_easy_results.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

try:
    import pandas as pd
    import numpy as np
    from scipy import stats
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  Warning: pandas/scipy not available. Limited analysis only.")

sys.path.insert(0, str(Path(__file__).parent))
try:
    from util.stats_utils import (
        bootstrap_mean_ci,
        cohens_d_paired,
        format_ci,
        holm_correction,
        paired_bootstrap_diff_ci,
        paired_permutation_test,
        pairwise_win_rate_matrix,
    )
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False
    print("⚠️  Warning: util.stats_utils not importable. CI/paired-test output disabled.")


def load_results(results_dir="experiments/very_easy/results"):
    """Load experiment results from directory."""
    results_path = Path(results_dir) / "very_easy_summary.json"

    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        print("Please run experiments first: python run_very_easy_experiment.py")
        sys.exit(1)

    with open(results_path, 'r') as f:
        data = json.load(f)

    return data["results"]


def _collect_paired_profits(success_df):
    per_model_profits = {}
    for model_key in sorted(success_df["model_key"].unique()):
        model_data = success_df[success_df["model_key"] == model_key].sort_values("seed")
        per_model_profits[model_key] = model_data["avg_daily_profit"].tolist()
    return per_model_profits


def _print_pairwise_table(contrasts, adjusted_pvals):
    headers = ["A vs B", "n", "delta_mean", "95% CI", "raw p", "Holm p", "Cohen's d"]
    rows = []
    for (a_key, b_key, mean_d, lo, hi, p_val, d, n_paired), adj_p in zip(contrasts, adjusted_pvals):
        rows.append([
            f"{a_key} vs {b_key}",
            str(n_paired),
            f"{mean_d:+.2f}",
            f"[{lo:+.2f}, {hi:+.2f}]",
            f"{p_val:.4f}",
            f"{adj_p:.4f}",
            f"{d:+.2f}",
        ])
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-"*w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def _print_win_rate_matrix(matrix):
    model_keys = list(matrix.keys())
    col_width = max(max(len(k) for k in model_keys), 8)
    header = " " * (col_width + 2) + "  ".join(f"{k:>{col_width}}" for k in model_keys)
    print(header)
    for row_key in model_keys:
        cells = []
        for col_key in model_keys:
            v = matrix[row_key][col_key]
            cells.append(f"{'--':>{col_width}}" if v != v else f"{v:>{col_width}.2f}")
        print(f"  {row_key:<{col_width}}  " + "  ".join(cells))


def analyze_results(results):
    """Analyze experiment results."""
    print("="*70)
    print("VERY EASY ENVIRONMENT EXPERIMENT RESULTS")
    print("="*70)
    print(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    if not PANDAS_AVAILABLE:
        print("⚠️  pandas not available, showing basic summary only")
        for result in results:
            if result["status"] == "success":
                print(f"{result['model']} (seed {result['seed']}): "
                      f"Profit={result['avg_daily_profit']:.2f}, "
                      f"Days={result['days_survived']}")
        return

    df = pd.DataFrame(results)
    success_df = df[df["status"] == "success"]

    if len(success_df) == 0:
        print("❌ No successful runs to analyze!")
        return

    print("="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    print(f"Total runs: {len(df)}")
    print(f"Successful: {len(success_df)}")
    print(f"Failed: {len(df) - len(success_df)}")
    print(f"Success rate: {len(success_df)/len(df)*100:.1f}%")
    print()

    per_model_profits = _collect_paired_profits(success_df)

    ci_label = "mean [95% bootstrap CI]" if STATS_AVAILABLE else "mean ± std"
    print("="*70)
    print(f"PER-MODEL STATISTICS ({ci_label})")
    print("="*70)

    for model_key in sorted(success_df["model_key"].unique()):
        model_data = success_df[success_df["model_key"] == model_key].sort_values("seed")
        model_name = model_data["model"].iloc[0]
        profits = per_model_profits[model_key]
        days = model_data["days_survived"].tolist()

        print(f"\n{model_name}:")
        print(f"  Successful runs: {len(model_data)}")

        if STATS_AVAILABLE and len(profits) >= 1:
            m, lo, hi = bootstrap_mean_ci(profits)
            print(f"  Daily profit:  {format_ci(m, lo, hi)}")
            m, lo, hi = bootstrap_mean_ci(days)
            print(f"  Days survived: {format_ci(m, lo, hi, digits=1)} / 30")
        else:
            print(f"  Daily profit:  {np.mean(profits):.2f} ± {np.std(profits, ddof=1) if len(profits)>1 else 0.0:.2f}")
            print(f"  Days survived: {np.mean(days):.1f} / 30")

        perfect = int((model_data["days_survived"] == 30).sum())
        print(f"  Perfect runs:  {perfect}/{len(model_data)}")
        print(f"  Expiry ratio:  {model_data['expiry_ratio'].mean():.2%}")
        print(f"  Return ratio:  {model_data['return_ratio'].mean():.2%}")
        if "total_cost" in model_data.columns:
            print(f"  Total cost:    ${model_data['total_cost'].sum():.2f}")
        if "inventory_turnover" in model_data.columns:
            print(f"  Inv. turnover: {model_data['inventory_turnover'].mean():.3f}")
        if "holding_units_days" in model_data.columns:
            print(f"  Holding u·day: {model_data['holding_units_days'].mean():.1f}")

    contrasts = []
    adjusted_pvals = []
    win_matrix = {}
    if STATS_AVAILABLE and len(per_model_profits) >= 2:
        print("\n" + "="*70)
        print("PAIRWISE COMPARISONS (paired across seeds; A - B)")
        print("="*70)

        model_keys = list(per_model_profits.keys())
        raw_pvals = []
        for i in range(len(model_keys)):
            for j in range(i + 1, len(model_keys)):
                a_key, b_key = model_keys[i], model_keys[j]
                a_vals, b_vals = per_model_profits[a_key], per_model_profits[b_key]
                n_paired = min(len(a_vals), len(b_vals))
                if n_paired < 2:
                    continue
                a_p, b_p = a_vals[:n_paired], b_vals[:n_paired]
                mean_d, lo, hi = paired_bootstrap_diff_ci(a_p, b_p)
                p_val = paired_permutation_test(a_p, b_p)
                d = cohens_d_paired(a_p, b_p)
                contrasts.append((a_key, b_key, mean_d, lo, hi, p_val, d, n_paired))
                raw_pvals.append(p_val)

        if contrasts:
            adjusted_pvals = holm_correction(raw_pvals)
            _print_pairwise_table(contrasts, adjusted_pvals)
        else:
            print("  (need >=2 paired seeds; skipped)")

        print("\n" + "="*70)
        print("PAIRWISE WIN-RATE MATRIX (row > col, ties=0.5)")
        print("="*70)
        win_matrix = pairwise_win_rate_matrix(per_model_profits)
        _print_win_rate_matrix(win_matrix)

    print("\n" + "="*70)
    print("COMPARISON TABLE (for paper)")
    print("="*70)

    summary_data = []
    seed_counts = []
    for model_key in sorted(success_df["model_key"].unique()):
        model_data = success_df[success_df["model_key"] == model_key].sort_values("seed")
        model_name = model_data["model"].iloc[0]
        profits = per_model_profits[model_key]
        seed_counts.append(len(profits))

        if STATS_AVAILABLE and len(profits) >= 1:
            m, lo, hi = bootstrap_mean_ci(profits)
            profit_str = format_ci(m, lo, hi)
        else:
            profit_str = f"{np.mean(profits):.2f}"

        summary_data.append({
            "Model": model_name,
            "n": len(profits),
            "Avg. Profit [95% CI]": profit_str,
            "Days": f"{model_data['days_survived'].mean():.1f}",
            "Survival": f"{(model_data['days_survived']==30).sum()}/{len(model_data)}",
            "Expiry": f"{model_data['expiry_ratio'].mean():.1%}",
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    output_dir = Path("experiments/very_easy/results")
    summary_df.to_csv(output_dir / "comparison_table.csv", index=False)
    print(f"\n✅ Comparison table saved: {output_dir / 'comparison_table.csv'}")

    print("\n" + "="*70)
    print("LATEX TABLE FORMAT")
    print("="*70)

    max_seeds = max(seed_counts) if seed_counts else 0
    latex_table = "\\begin{table}[t]\n"
    latex_table += "\\centering\n"
    latex_table += f"\\caption{{Performance in Very Easy environment (up to {max_seeds} paired seeds/model). Profit reported as mean [95\\% percentile-bootstrap CI].}}\n"
    latex_table += "\\label{tab:very_easy_results}\n"
    latex_table += "\\begin{tabular}{lccccc}\n"
    latex_table += "\\toprule\n"
    latex_table += "\\textbf{Model} & \\textbf{n} & \\textbf{Avg. Profit [95\\% CI]} & \\textbf{Days} & \\textbf{Survival} & \\textbf{Expiry} \\\\\n"
    latex_table += "\\midrule\n"
    for _, row in summary_df.iterrows():
        latex_table += f"{row['Model']} & {row['n']} & {row['Avg. Profit [95% CI]']} & {row['Days']} & {row['Survival']} & {row['Expiry']} \\\\\n"
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}"

    print(latex_table)
    with open(output_dir / "latex_table.tex", 'w') as f:
        f.write(latex_table)
    print(f"\n✅ LaTeX table saved: {output_dir / 'latex_table.tex'}")

    if STATS_AVAILABLE:
        stats_dump = {
            "per_model": {},
            "pairwise_contrasts": [],
            "win_rate_matrix": {},
        }
        for model_key, profits in per_model_profits.items():
            if not profits:
                continue
            m, lo, hi = bootstrap_mean_ci(profits)
            stats_dump["per_model"][model_key] = {
                "n": len(profits),
                "profits": profits,
                "mean": m,
                "ci_low": lo,
                "ci_high": hi,
            }
        for (a_key, b_key, mean_d, lo, hi, p_val, d, n_paired), adj_p in zip(contrasts, adjusted_pvals):
            stats_dump["pairwise_contrasts"].append({
                "a": a_key,
                "b": b_key,
                "n_paired": n_paired,
                "mean_diff": mean_d,
                "ci_low": lo,
                "ci_high": hi,
                "p_raw": p_val,
                "p_holm": adj_p,
                "cohens_d": d,
            })
        stats_dump["win_rate_matrix"] = {
            k: {kk: (None if vv != vv else vv) for kk, vv in row.items()}
            for k, row in win_matrix.items()
        }
        paired_stats_path = output_dir / "paired_stats.json"
        with open(paired_stats_path, "w") as f:
            json.dump(stats_dump, f, indent=2)
        print(f"\n✅ Paired-stats JSON saved: {paired_stats_path}")


def generate_comparison_figure(results):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n⚠️  matplotlib not available, skipping figure generation")
        return

    df = pd.DataFrame(results)
    success_df = df[df["status"] == "success"]
    if len(success_df) == 0:
        return

    model_keys_sorted = sorted(success_df["model_key"].unique())
    per_model_profits = {}
    per_model_names = {}
    for model_key in model_keys_sorted:
        model_data = success_df[success_df["model_key"] == model_key].sort_values("seed")
        per_model_profits[model_key] = model_data["avg_daily_profit"].tolist()
        per_model_names[model_key] = model_data["model"].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    models = [per_model_names[k] for k in model_keys_sorted]
    means, lows, highs = [], [], []
    for k in model_keys_sorted:
        vals = per_model_profits[k]
        if STATS_AVAILABLE and len(vals) >= 2:
            m, lo, hi = bootstrap_mean_ci(vals)
        else:
            m = float(np.mean(vals)) if vals else float("nan")
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            lo, hi = m - std, m + std
        means.append(m)
        lows.append(m - lo)
        highs.append(hi - m)

    axes[0].bar(models, means, yerr=[lows, highs], capsize=5, alpha=0.75)
    axes[0].set_ylabel('Avg. Daily Profit')
    axes[0].set_title('Profit with 95% Bootstrap CI' if STATS_AVAILABLE else 'Profit ± std')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].tick_params(axis='x', rotation=20)

    days_data = [
        success_df[success_df["model_key"] == k]["days_survived"].values
        for k in model_keys_sorted
    ]
    axes[1].boxplot(days_data, tick_labels=models)
    axes[1].set_ylabel('Days Survived')
    axes[1].set_title('Episode Length Distribution')
    axes[1].axhline(y=30, color='r', linestyle='--', label='Target (30 days)')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].tick_params(axis='x', rotation=20)

    seeds_sorted = sorted(success_df["seed"].unique())
    profit_by_seed = {
        seed: success_df[success_df["seed"] == seed].set_index("model_key")["avg_daily_profit"].to_dict()
        for seed in seeds_sorted
    }
    for k in model_keys_sorted:
        ranks = []
        xs = []
        for seed in seeds_sorted:
            seed_map = profit_by_seed[seed]
            present = {mk: seed_map[mk] for mk in model_keys_sorted if mk in seed_map}
            if k not in present:
                continue
            ranked = sorted(present.items(), key=lambda kv: kv[1], reverse=True)
            rank = [i for i, (mk, _) in enumerate(ranked) if mk == k][0] + 1
            ranks.append(rank)
            xs.append(seed)
        if ranks:
            axes[2].plot(xs, ranks, marker='o', label=per_model_names[k])
    axes[2].invert_yaxis()
    axes[2].set_xlabel('Seed')
    axes[2].set_ylabel('Rank (1 = best)')
    axes[2].set_title('Rank Stability Across Seeds')
    axes[2].set_yticks(range(1, len(model_keys_sorted) + 1))
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8, loc='best')

    plt.tight_layout()
    output_dir = Path("experiments/very_easy/results")
    fig_path = output_dir / "comparison_figure.pdf"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "comparison_figure.png", dpi=200, bbox_inches='tight')
    print(f"\n✅ Figure saved: {fig_path}")


def main():
    """Main function."""
    results = load_results()
    analyze_results(results)
    generate_comparison_figure(results)


if __name__ == "__main__":
    main()
