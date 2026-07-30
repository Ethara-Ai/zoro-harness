"""Harbor-format task writer for generate_datasets.py --format harbor mode.

Given a (task_id, archetype, env_config), emits a directory conforming to the
retailbench harbor task layout the pipeline downstream expects:

    <out_root>/<task_id>/
    ├── task.toml
    ├── instruction.md
    ├── environment/dataset.json
    ├── solution/solve.sh
    └── tests/test.sh

Reference shape: /Users/apple/zoro-harness/48df3c8b-f46f-50e6-983e-d583fda86fa2/
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_SUITE = "retailbench"
_HORIZON_DAYS = 180
_ORACLE_SAMPLE_SIZE = 2
_ORACLE_BULK_QTY_MULTIPLIER = 4
_ALLOW_INTERNET = "true"
_VERIFIER_TIMEOUT_SEC = 1800.0

_TIER_NAMES: dict[int, str] = {
    5000:  "micro",
    10000: "small",
    18000: "medium",
    30000: "standard",
    50000: "large",
    80000: "flagship",
}


TASK_TOML_TEMPLATE = """\
version = "1.0"

# RetailBench-specific fields live under [metadata] (Harbor's free-form dict[str, Any]).
# The [environment] / [verifier] / [solution] tables contain ONLY Harbor TaskConfig schema
# fields, so this task validates against a strict Harbor loader. solution/solve.sh and
# tests/test.sh are discovered by convention (not declared here).

[metadata]
id              = "{task_id}"
suite           = "{suite}"
archetype       = "{archetype}"       # {archetype_note}
description     = "Operate store #{store_id} for {horizon_days} days; maximize terminal net worth (cash + inventory at depreciated cost + in-transit)."
# provenance (mirrors environment/dataset.json; horizon is a runner constant, absent from dataset):
horizon_days    = {horizon_days}
economic_tier   = "{economic_tier}"                # funds {initial_funds} / rent {everyday_rent} / capacity {inventory_capacity}
category_count  = {category_count}
category_effect = {category_effect}                  # cross-SKU cannibalization coefficient (NOT price elasticity)
enable_new      = {enable_new_lower}
enable_review   = {enable_review_lower}
# reference-solution (oracle) policy knobs — held constant across the suite so goldens compare:
oracle_policy   = {{ sample_size = {sample_size}, bulk_qty_multiplier = {bulk_qty_multiplier} }}
# grading conventions consumed by tests/test.sh (paths, not Harbor schema fields):
world_config    = "environment/dataset.json"
submission      = "tool_calls.jsonl"

[environment]
# Graded on-host by evaluation/run_eval.py; Harbor's container engine is not used, so there is
# no Dockerfile and docker_image is intentionally omitted (defaults to None). allow_internet
# documents that the LLM council reaches judge vendors.
allow_internet = {allow_internet}

[verifier]
timeout_sec = {verifier_timeout_sec}                     # council fan-out dominates wall-clock
"""


INSTRUCTION_MD_TEMPLATE = """\
# Store Operations

You run a retail store for a fixed period of **{days} days**. Your goal is to **maximize the store's net worth at the end of day {days}**.

Net worth is your cash on hand, plus the value of the inventory you are holding (counted at what you paid for it, declining as the goods age), plus the value of any stock you have ordered but not yet received. A fixed rent of **${rent} is charged every day**, whatever your sales, so you must stay solvent.

## How the store works

- You stock products across several categories. Products in the same category compete for the same shoppers, so adding more products to a category can eat into the sales of the others.
- A product's unit sales fall as you raise its shelf price and rise as you lower it. Total shopper traffic is set by the market and is outside your control — you compete for a share of it.
- Past sales history is available and is your main signal for future demand.{review_bullet}{news_bullet}
- Suppliers differ in price, in quality, and in how long they take to deliver. You pay for an order when you place it, and it arrives after a delay of a few days.
- Inventory is perishable: stock has a short shelf life, and units that expire unsold are cleared at a loss — so ordering more than you can sell in time is wasteful.
- The warehouse holds a limited number of units. Stock ordered beyond that limit cannot go on the shelf until space frees up, so it sits idle.

## What you do

Each day you can review the store's current data — inventory, funds, sales history, supplier prices and quality{reviews_and_clause} — and then decide what to order from which supplier and what shelf prices to set. Your decisions carry over from day to day, so plan across the whole {days} days rather than optimizing a single day in isolation.
"""

_REVIEW_BULLET = "\n- Customer reviews affect demand over time, and each product's return rate depends on the quality of the supplier it was bought from — higher-quality goods are returned less often. A return refunds the customer the full sale price."
_NEWS_BULLET = "\n- News can temporarily raise or lower demand for particular products or categories, and can move supplier costs. Watch the news and factor it into your ordering and pricing."


SOLVE_SH_TEMPLATE = """#!/usr/bin/env bash
# Reference solution = the RetailBench Oracle (rule-based, non-LLM shopkeeper).
# Deterministic: the environment seeds random + np.random from global_random_seed ({seed}),
# so this reproduces byte-identically (see tools/verify_oracle_reproducibility.py).
#
# The committed artifacts (tool_calls.jsonl, golden.json, metadata.json) in this directory
# were produced by exactly this command; solve.sh is the reproducible recipe + Harbor's
# oracle-agent sanity check. Threshold calibration additionally sweeps the policy knobs
# (sample_size x bulk_qty_multiplier) at the SAME seed — see evaluation/thresholds.py.
set -euo pipefail
HARNESS=${{RETAILBENCH_HARNESS:-/opt/retailbench}}   # path to the harness install
TASK_DIR="$(cd "$(dirname "$0")/.." && pwd)"

python "$HARNESS/tools/run_oracle.py" \\
    --task "$TASK_DIR/environment/dataset.json" \\
    --days {horizon_days} \\
    --sample_size {sample_size} \\
    --bulk_qty_multiplier {bulk_qty_multiplier} \\
    --out "$TASK_DIR/solution"
# Emits: solution/tool_calls.jsonl, solution/golden.json, solution/metadata.json
"""


TEST_SH_TEMPLATE = """#!/usr/bin/env bash
# Harbor verifier entry point. Orchestrates the three verifier artifacts and writes the
# Harbor reward contract. Verifier LOGIC lives in the shared evaluation library; this file
# is the thin, task-local composition point.
set -uo pipefail

TASK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TESTS="$TASK_DIR/tests"
# The graded submission: the agent's canonical trajectory (tool_calls.jsonl).
SUB="${{SUBMISSION:?set SUBMISSION to the agent run dir or its tool_calls.jsonl}}"
OUT="${{VERIFIER_OUT:-$TASK_DIR/verifier}}"
mkdir -p "$OUT"

# ARTIFACT 1 — deterministic outcome + process pytest (imports evaluation fixtures).
# Test failures are DATA, not errors -> never abort the verifier on non-zero pytest exit.
pytest "$TESTS/test_outputs.py" -q \\
    --task-id {task_id} \\
    --traj-path "$SUB" \\
    --dataset-path "$TASK_DIR/environment/dataset.json" \\
    --thresholds-path "$TESTS/thresholds.json" \\
    --json-out "$OUT/test_results.json" || true

# ARTIFACT 2 — non-deterministic LLM council (our own driver). Reads task-specific rubrics
# from rubrics.json; the judge PANEL is the suite-wide shared config bundled with the package
# (evaluation/council/judges.yaml) — not duplicated per task.
if [ "${{NO_COUNCIL:-0}}" != "1" ]; then
  python -m evaluation.council.run_council \\
      --rubrics "$TESTS/rubrics.json" \\
      --truth   "$TASK_DIR/TRUTH.md" \\
      --traj-path "$SUB" \\
      --dataset-path "$TASK_DIR/environment/dataset.json" \\
      --out "$OUT/council_results.json" || echo "WARN: council failed; aggregate will run pytest-only"
fi

# ARTIFACT 3 — aggregate to the Harbor reward contract (reward.json + scalar reward.txt).
python -m evaluation.aggregate \\
    --test-results    "$OUT/test_results.json" \\
    --council-results "$OUT/council_results.json" \\
    --weights "$TESTS/thresholds.json" \\
    --reward-json "$OUT/reward.json" \\
    --reward-txt  "$OUT/reward.txt"
"""


def _derive_category_effect(env_config: dict) -> float:
    effects = env_config["category_effects"]
    selected = env_config["selected_categories"]
    spaced = [c.replace("_", " ") for c in selected]
    values = [effects[c] for c in spaced]
    return float(statistics.median(values))


def _derive_economic_tier(env_config: dict) -> str:
    return _TIER_NAMES.get(env_config["initial_funds"], "custom")


def _archetype_note(archetype: str, category_count: int) -> str:
    data_mode = "dynamic" if "dynamic" in archetype else "still"
    if "hard" in archetype:
        return f"{data_mode} data_dir + 20 categories (>10 => hard)"
    return f"{data_mode} data_dir + {category_count} categories (<=10 => middle)"


def _render_task_toml(task_id: str, archetype: str, env_config: dict) -> str:
    return TASK_TOML_TEMPLATE.format(
        task_id=task_id,
        suite=_SUITE,
        archetype=archetype,
        archetype_note=_archetype_note(archetype, len(env_config["selected_categories"])),
        store_id=env_config["store_id"],
        horizon_days=_HORIZON_DAYS,
        economic_tier=_derive_economic_tier(env_config),
        initial_funds=env_config["initial_funds"],
        everyday_rent=env_config["everyday_rent"],
        inventory_capacity=env_config["inventory_capacity"],
        category_count=len(env_config["selected_categories"]),
        category_effect=_derive_category_effect(env_config),
        enable_new_lower=str(bool(env_config["enable_new"])).lower(),
        enable_review_lower=str(bool(env_config["enable_review"])).lower(),
        sample_size=_ORACLE_SAMPLE_SIZE,
        bulk_qty_multiplier=_ORACLE_BULK_QTY_MULTIPLIER,
        allow_internet=_ALLOW_INTERNET,
        verifier_timeout_sec=_VERIFIER_TIMEOUT_SEC,
    )


def _render_instruction_md(env_config: dict) -> str:
    review = bool(env_config["enable_review"])
    news = bool(env_config["enable_new"])
    return INSTRUCTION_MD_TEMPLATE.format(
        days=_HORIZON_DAYS,
        rent=env_config["everyday_rent"],
        review_bullet=_REVIEW_BULLET if review else "",
        news_bullet=_NEWS_BULLET if news else "",
        reviews_and_clause=", and customer reviews" if review else "",
    )


def _render_solve_sh(env_config: dict) -> str:
    return SOLVE_SH_TEMPLATE.format(
        seed=env_config["global_random_seed"],
        horizon_days=_HORIZON_DAYS,
        sample_size=_ORACLE_SAMPLE_SIZE,
        bulk_qty_multiplier=_ORACLE_BULK_QTY_MULTIPLIER,
    )


def _render_test_sh(task_id: str) -> str:
    return TEST_SH_TEMPLATE.format(task_id=task_id)


def write_harbor_task(
    task_id: str,
    archetype: str,
    env_config: dict,
    out_root: Path,
    run_oracle: bool = False,
) -> Path:
    task_dir = out_root / task_id
    (task_dir / "environment").mkdir(parents=True, exist_ok=True)
    (task_dir / "solution").mkdir(parents=True, exist_ok=True)
    (task_dir / "tests").mkdir(parents=True, exist_ok=True)

    (task_dir / "task.toml").write_text(_render_task_toml(task_id, archetype, env_config))
    (task_dir / "instruction.md").write_text(_render_instruction_md(env_config))
    (task_dir / "environment" / "dataset.json").write_text(json.dumps(env_config, indent=2))

    solve_path = task_dir / "solution" / "solve.sh"
    solve_path.write_text(_render_solve_sh(env_config))
    solve_path.chmod(0o755)

    test_path = task_dir / "tests" / "test.sh"
    test_path.write_text(_render_test_sh(task_id))
    test_path.chmod(0o755)

    if run_oracle:
        _invoke_oracle(task_dir)

    return task_dir


def _invoke_oracle(task_dir: Path) -> None:
    dataset_path = task_dir / "environment" / "dataset.json"
    out_path = task_dir / "solution"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_oracle.py"),
        "--task", str(dataset_path),
        "--days", str(_HORIZON_DAYS),
        "--sample_size", str(_ORACLE_SAMPLE_SIZE),
        "--bulk_qty_multiplier", str(_ORACLE_BULK_QTY_MULTIPLIER),
        "--out", str(out_path),
    ]
    print(f"[harbor] running oracle: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)
