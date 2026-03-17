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

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Separate successful runs
    success_df = df[df["status"] == "success"]

    if len(success_df) == 0:
        print("❌ No successful runs to analyze!")
        return

    # Overall summary
    print("="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    print(f"Total runs: {len(df)}")
    print(f"Successful: {len(success_df)}")
    print(f"Failed: {len(df) - len(success_df)}")
    print(f"Success rate: {len(success_df)/len(df)*100:.1f}%")
    print()

    # Per-model statistics
    print("="*70)
    print("PER-MODEL STATISTICS")
    print("="*70)

    for model_key in sorted(success_df["model_key"].unique()):
        model_data = success_df[success_df["model_key"] == model_key]
        model_name = model_data["model"].iloc[0]

        print(f"\n{model_name}:")
        print(f"  Successful runs: {len(model_data)}/3")

        if len(model_data) > 0:
            # Profit statistics
            profit_mean = model_data["avg_daily_profit"].mean()
            profit_std = model_data["avg_daily_profit"].std()
            profit_min = model_data["avg_daily_profit"].min()
            profit_max = model_data["avg_daily_profit"].max()

            print(f"\n  Daily Profit:")
            print(f"    Mean:  {profit_mean:.2f}")
            print(f"    Std:   {profit_std:.2f}")
            print(f"    Range: [{profit_min:.2f}, {profit_max:.2f}]")

            # Days survived
            days_mean = model_data["days_survived"].mean()
            days_survived = (model_data["days_survived"] == 30).sum()

            print(f"\n  Days Survived:")
            print(f"    Mean:  {days_mean:.1f} / 30")
            print(f"    Perfect: {days_survived}/{len(model_data)} ({days_survived/len(model_data)*100:.0f}%)")

            # Other metrics
            print(f"\n  Other Metrics:")
            print(f"    Expiry ratio:  {model_data['expiry_ratio'].mean():.2%}")
            print(f"    Return ratio:  {model_data['return_ratio'].mean():.2%}")

            # Cost
            if "total_cost" in model_data.columns:
                total_cost = model_data["total_cost"].sum()
                print(f"    Total cost:    ${total_cost:.2f}")

    # Comparison table
    print("\n" + "="*70)
    print("COMPARISON TABLE (for paper)")
    print("="*70)

    # Create summary table
    summary_data = []
    for model_key in sorted(success_df["model_key"].unique()):
        model_data = success_df[success_df["model_key"] == model_key]
        model_name = model_data["model"].iloc[0]

        summary_data.append({
            "Model": model_name,
            "Avg. Profit": f"{model_data['avg_daily_profit'].mean():.2f} ± {model_data['avg_daily_profit'].std():.2f}",
            "Days": f"{model_data['days_survived'].mean():.1f}",
            "Survival": f"{(model_data['days_survived']==30).sum()}/{len(model_data)}",
            "Expiry": f"{model_data['expiry_ratio'].mean():.1%}",
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Save comparison table as CSV
    output_dir = Path("experiments/very_easy/results")
    summary_df.to_csv(output_dir / "comparison_table.csv", index=False)
    print(f"\n✅ Comparison table saved: {output_dir / 'comparison_table.csv'}")

    # LaTeX table format
    print("\n" + "="*70)
    print("LATEX TABLE FORMAT")
    print("="*70)

    latex_table = "\\begin{table}[t]\n"
    latex_table += "\\centering\n"
    latex_table += "\\caption{Performance in Very Easy environment (n=3).}\n"
    latex_table += "\\label{tab:very_easy_results}\n"
    latex_table += "\\begin{tabular}{lcccc}\n"
    latex_table += "\\toprule\n"
    latex_table += "\\textbf{Model} & \\textbf{Avg. Profit} & \\textbf{Days} & \\textbf{Survival} & \\textbf{Expiry} \\\\\\n"
    latex_table += "\\midrule\n"

    for _, row in summary_df.iterrows():
        latex_table += f"{row['Model']} & ${row['Avg. Profit']}$ & {row['Days']} & {row['Survival']} & {row['Expiry']} \\\\\n"

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}"

    print(latex_table)

    # Save LaTeX table
    with open(output_dir / "latex_table.tex", 'w') as f:
        f.write(latex_table)
    print(f"\n✅ LaTeX table saved: {output_dir / 'latex_table.tex'}")


def generate_comparison_figure(results):
    """Generate comparison figure (if matplotlib available)."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.DataFrame(results)
        success_df = df[df["status"] == "success"]

        if len(success_df) == 0:
            return

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot 1: Average daily profit
        models = []
        profits = []
        errors = []

        for model_key in sorted(success_df["model_key"].unique()):
            model_data = success_df[success_df["model_key"] == model_key]
            models.append(model_data["model"].iloc[0])
            profits.append(model_data["avg_daily_profit"].mean())
            errors.append(model_data["avg_daily_profit"].std())

        axes[0].bar(models, profits, yerr=errors, capsize=5, alpha=0.7)
        axes[0].set_ylabel('Average Daily Profit')
        axes[0].set_title('Performance in Very Easy Environment')
        axes[0].grid(axis='y', alpha=0.3)

        # Plot 2: Days survived
        days_data = []
        for model_key in sorted(success_df["model_key"].unique()):
            model_data = success_df[success_df["model_key"] == model_key]
            days_data.append(model_data["days_survived"].values)

        axes[1].boxplot(days_data, labels=models)
        axes[1].set_ylabel('Days Survived')
        axes[1].set_title('Episode Length Distribution')
        axes[1].axhline(y=30, color='r', linestyle='--', label='Target (30 days)')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()

        # Save figure
        output_dir = Path("experiments/very_easy/results")
        fig_path = output_dir / "comparison_figure.pdf"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Figure saved: {fig_path}")

    except ImportError:
        print("\n⚠️  matplotlib not available, skipping figure generation")


def main():
    """Main function."""
    results = load_results()
    analyze_results(results)
    generate_comparison_figure(results)


if __name__ == "__main__":
    main()
