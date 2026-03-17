#!/usr/bin/env python3
"""
Very Easy Environment Experiment Runner

Purpose: Evaluate 3 models (DeepSeek-V3.2, GLM-4.6, Kimi-K2) on Very Easy environment
with 3 rollouts each using different random seeds.

Usage:
    python run_very_easy_experiment.py

Expected output:
    - experiments/very_easy/results/very_easy_summary.json
    - experiments/very_easy/results/[model]_seed[N]_trajectory.json
    - experiments/very_easy/results/very_easy_results.csv
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import your existing modules
from util.very_easy_config import VERY_EASY_CONFIG
from run_env import run_single_episode  # Assuming this function exists


# Model configurations
MODELS = {
    "deepseek-v3.2": {
        "name": "DeepSeek-V3.2",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "model": "deepseek-v3.2-exp"
    },
    "glm-4.6": {
        "name": "GLM-4.6",
        "api_key": os.getenv("GLM_API_KEY"),
        "base_url": os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        "model": "glm-4.6"
    },
    "kimi-k2-thinking": {
        "name": "Kimi-K2 (Thinking)",
        "api_key": os.getenv("KIMI_API_KEY"),
        "base_url": os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
        "model": "kimi-k2-thinking"
    }
}

# Experiment configuration
EXPERIMENT_CONFIG = {
    "environment": "very_easy",
    "framework": "proposed",  # Use your Proposed framework
    "max_days": 30,
    "seeds": [0, 1, 2],  # 3 different random seeds
    "output_dir": "experiments/very_easy/results"
}


def create_output_directory():
    """Create output directory if it doesn't exist."""
    output_dir = Path(EXPERIMENT_CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_single_experiment(model_key, seed):
    """
    Run a single experiment with specified model and seed.

    Args:
        model_key: Key from MODELS dict
        seed: Random seed for environment initialization

    Returns:
        dict: Experiment results
    """
    model_config = MODELS[model_key]
    print(f"\n{'='*70}")
    print(f"Running: {model_config['name']} | Seed: {seed}")
    print(f"{'='*70}")

    try:
        # Import the run_env module function
        # This assumes you have a function that can run a single episode
        result = run_single_episode(
            model=model_config["model"],
            api_key=model_config["api_key"],
            base_url=model_config["base_url"],
            config_type=EXPERIMENT_CONFIG["environment"],
            seed=seed,
            max_days=EXPERIMENT_CONFIG["max_days"],
            framework=EXPERIMENT_CONFIG["framework"],
            output_dir=EXPERIMENT_CONFIG["output_dir"]
        )

        # Extract key metrics
        metrics = {
            "model": model_config["name"],
            "model_key": model_key,
            "seed": seed,
            "environment": EXPERIMENT_CONFIG["environment"],
            "framework": EXPERIMENT_CONFIG["framework"],

            # Performance metrics
            "final_net_worth": result.get("final_net_worth", 0),
            "avg_daily_profit": result.get("avg_daily_profit", 0),
            "avg_daily_sales": result.get("avg_daily_sales", 0),
            "avg_daily_income": result.get("avg_daily_income", 0),
            "expiry_ratio": result.get("expiry_ratio", 0),
            "return_ratio": result.get("return_ratio", 0),
            "days_survived": result.get("days_survived", 0),

            # Token usage
            "total_tokens": result.get("total_tokens", 0),
            "total_cost": result.get("total_cost", 0.0),
            "total_api_calls": result.get("total_api_calls", 0),

            # Status
            "status": "success",
            "error": None
        }

        # Save detailed trajectory
        trajectory_path = Path(EXPERIMENT_CONFIG["output_dir"]) / f"{model_key}_seed{seed}_trajectory.json"
        with open(trajectory_path, 'w') as f:
            json.dump(result.get("trajectory", []), f, indent=2)

        print(f"✅ Success: Profit={metrics['avg_daily_profit']:.2f}, Days={metrics['days_survived']}/{EXPERIMENT_CONFIG['max_days']}")

        return metrics

    except Exception as e:
        print(f"❌ Error: {e}")

        # Return error result
        return {
            "model": model_config["name"],
            "model_key": model_key,
            "seed": seed,
            "environment": EXPERIMENT_CONFIG["environment"],
            "framework": EXPERIMENT_CONFIG["framework"],
            "status": "error",
            "error": str(e),
            # All metrics as NaN or 0
            "final_net_worth": 0,
            "avg_daily_profit": 0,
            "avg_daily_sales": 0,
            "days_survived": 0,
            "total_tokens": 0,
            "total_cost": 0.0
        }


def main():
    """Main experiment runner."""
    print("="*70)
    print("VERY EASY ENVIRONMENT EXPERIMENT")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Create output directory
    output_dir = create_output_directory()
    print(f"Output directory: {output_dir}")

    # Print experiment configuration
    print("\n" + "="*70)
    print("EXPERIMENT CONFIGURATION")
    print("="*70)
    print(f"Environment: {EXPERIMENT_CONFIG['environment']}")
    print(f"Framework: {EXPERIMENT_CONFIG['framework']}")
    print(f"Max days: {EXPERIMENT_CONFIG['max_days']}")
    print(f"Seeds: {EXPERIMENT_CONFIG['seeds']}")
    print(f"Models:")
    for key, config in MODELS.items():
        print(f"  - {config['name']} ({key})")
    print(f"\nTotal runs: {len(MODELS) * len(EXPERIMENT_CONFIG['seeds'])}")

    # Run experiments
    results = []

    for model_key in MODELS.keys():
        for seed in EXPERIMENT_CONFIG["seeds"]:
            result = run_single_experiment(model_key, seed)
            results.append(result)

    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    # Save as JSON
    summary_path = output_dir / "very_easy_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "config": EXPERIMENT_CONFIG,
            "models": {k: v["name"] for k, v in MODELS.items()},
            "results": results,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    print(f"Summary saved: {summary_path}")

    # Save as CSV (requires pandas)
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        csv_path = output_dir / "very_easy_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"CSV saved: {csv_path}")
    except ImportError:
        print("⚠️  pandas not available, skipping CSV export")

    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    try:
        import pandas as pd
        df = pd.DataFrame(results)

        for model_key in MODELS.keys():
            model_data = df[df["model_key"] == model_key]
            model_name = MODELS[model_key]["name"]

            print(f"\n{model_name}:")
            print(f"  Runs: {len(model_data)}")
            print(f"  Success rate: {(model_data['status']=='success').sum()}/{len(model_data)}")

            if (model_data['status']=='success').sum() > 0:
                success_data = model_data[model_data['status']=='success']
                print(f"  Avg daily profit: {success_data['avg_daily_profit'].mean():.2f} ± {success_data['avg_daily_profit'].std():.2f}")
                print(f"  Days survived: {success_data['days_survived'].mean():.1f} / {EXPERIMENT_CONFIG['max_days']}")
                print(f"  Survival rate: {(success_data['days_survived']==EXPERIMENT_CONFIG['max_days']).sum()}/{len(success_data)}")
                print(f"  Total cost: ${success_data['total_cost'].sum():.2f}")

    except ImportError:
        print("⚠️  pandas not available, skipping statistics")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Very Easy environment experiments")
    parser.add_argument("--models", nargs='+', choices=list(MODELS.keys()),
                       help="Models to run (default: all)")
    parser.add_argument("--seeds", type=int, nargs='+',
                       help="Seeds to run (default: 0 1 2)")

    args = parser.parse_args()

    if args.models:
        # Filter models
        MODELS = {k: v for k, v in MODELS.items() if k in args.models}

    if args.seeds:
        # Override seeds
        EXPERIMENT_CONFIG["seeds"] = args.seeds

    main()
