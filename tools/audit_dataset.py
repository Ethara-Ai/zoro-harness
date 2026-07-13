#!/usr/bin/env python3
"""
audit_dataset.py — End-to-end QC/audit for RetailBench dataset tasks.

Per-task checks:
  C1  Schema completeness (top-level keys + env_config key set)
  C2  task_id UUID integrity (recomputed from identity tuple)
  C3  Archetype locked-key alignment (15 locked keys match template exactly)
  C4  validate_dataset Gates 1–3, 5–8 always; Gate 4 paper-run exclusion when paper_data_dir present
  C6  Archetype directory consistency (dir name vs infer_archetype)
  C7  Variable keys all present (12 variable keys)
  C8  No extraneous env_config keys (only locked + variable keys allowed)

Cross-task checks:
  X1  All task_ids globally unique
  X2  No two tasks share identical env_config (redundancy / copy-paste detection)
  X3  No two tasks share identical identity tuple (archetype, seed, categories, ep, tier)

Usage:
    python zoro/harness/tools/audit_dataset.py --dataset-dir zoro/harness/dataset
    python zoro/harness/tools/audit_dataset.py --dataset-dir zoro/harness/dataset --verbose
    python zoro/harness/tools/audit_dataset.py --dataset-dir zoro/harness/dataset --archetype dynamic_hard
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_TOOLS_DIR = Path(__file__).resolve().parent
_HARNESS_DIR = _TOOLS_DIR.parent
sys.path.insert(0, str(_HARNESS_DIR))

from generator.templates.archetypes import ARCHETYPES
from generator.validators import (
    EXPECTED_KEYS,
    PAPER_DATA_DIR,
    _paper_run_ids_from_dir,
    compute_task_id,
    infer_archetype,
    match_elasticity_profile,
    match_economic_tier,
    validate_dataset,
)

# ---------------------------------------------------------------------------
# Rich / fallback console
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rich.text import Text

    _RICH = True
    console = Console()

    def _ok(msg: str) -> str:
        return f"[green]✓[/green] {msg}"

    def _fail(msg: str) -> str:
        return f"[red]✗[/red] {msg}"

    def _warn(msg: str) -> str:
        return f"[yellow]![/yellow] {msg}"

    def _bold(msg: str) -> str:
        return f"[bold]{msg}[/bold]"

    def _dim(msg: str) -> str:
        return f"[dim]{msg}[/dim]"

except ImportError:
    _RICH = False
    console = None  # type: ignore

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    def _ok(msg: str) -> str:
        return f"{GREEN}✓{RESET} {msg}"

    def _fail(msg: str) -> str:
        return f"{RED}✗{RESET} {msg}"

    def _warn(msg: str) -> str:
        return f"{YELLOW}!{RESET} {msg}"

    def _bold(msg: str) -> str:
        return f"{BOLD}{msg}{RESET}"

    def _dim(msg: str) -> str:
        return f"{DIM}{msg}{RESET}"


def _print(msg: str = "") -> None:
    if _RICH:
        console.print(msg)
    else:
        # Strip rich markup for plain output
        import re
        clean = re.sub(r"\[/?[^\]]+\]", "", msg)
        print(clean)


# ---------------------------------------------------------------------------
# Variable key manifest (12 keys stored in every task)
# ---------------------------------------------------------------------------
VARIABLE_KEYS: frozenset[str] = frozenset({
    "global_random_seed",
    "selected_categories",
    "category_effects",
    "initial_funds",
    "everyday_rent",
    "inventory_capacity",
    "review_ratio",
    "news_impact_base_scale",
    "news_sample_ratios",
    "news_daily_count",
    "news_random_seed",
    "initial_inventory",
})

LOCKED_KEYS: frozenset[str] = frozenset(ARCHETYPES["dynamic_hard"].keys())
ALLOWED_KEYS: frozenset[str] = LOCKED_KEYS | VARIABLE_KEYS

# Guard: all archetypes must share the same 15 key names (values differ per archetype)
assert all(
    ARCHETYPES[a].keys() == ARCHETYPES["dynamic_hard"].keys() for a in ARCHETYPES
), "Archetype key-name divergence — update LOCKED_KEYS derivation in audit_dataset.py"


# ---------------------------------------------------------------------------
# Per-task check result
# ---------------------------------------------------------------------------
class CheckResult:
    def __init__(self, name: str, passed: bool, details: str = ""):
        self.name = name
        self.passed = passed
        self.details = details

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"{self.name}:{status} {self.details}"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_c1_schema(ds: dict) -> CheckResult:
    """C1: Flat config key set completeness — must contain exactly EXPECTED_KEYS."""
    missing = EXPECTED_KEYS - set(ds.keys())
    if missing:
        return CheckResult("C1", False, f"missing keys: {sorted(missing)}")
    extra = set(ds.keys()) - EXPECTED_KEYS
    if extra:
        return CheckResult("C1", False, f"extra keys: {sorted(extra)}")
    return CheckResult("C1", True, "all expected keys present")


def check_c2_uuid(ds: dict, archetype: str, stored_id: str) -> CheckResult:
    """C2: Recompute task_id from identity tuple and verify it matches filename stem."""
    cfg = ds  # flat schema
    seed = cfg.get("global_random_seed")
    cats = cfg.get("selected_categories", [])
    effects = cfg.get("category_effects", {})
    ep = match_elasticity_profile(effects)
    tier = match_economic_tier(cfg)

    try:
        recomputed = compute_task_id(archetype, seed, cats, ep, tier)
    except Exception as exc:
        return CheckResult("C2", False, f"compute_task_id raised: {exc}")

    if recomputed != stored_id:
        return CheckResult(
            "C2", False,
            f"UUID mismatch: filename={stored_id!r} recomputed={recomputed!r} "
            f"(ep={ep!r}, tier={tier!r})"
        )
    return CheckResult("C2", True, f"UUID verified (ep={ep!r}, tier={tier!r})")


def check_c3_locked_keys(ds: dict, archetype: str) -> CheckResult:
    """C3: All 15 locked keys in env_config must exactly match the archetype template."""
    cfg = ds  # flat schema
    template = ARCHETYPES[archetype]
    mismatches: list[str] = []

    for key, expected in template.items():
        actual = cfg.get(key, _SENTINEL := object())
        if actual is _SENTINEL:
            mismatches.append(f"{key!r} missing")
        elif actual != expected:
            mismatches.append(f"{key!r}: expected {expected!r}, got {actual!r}")

    if mismatches:
        return CheckResult("C3", False, "; ".join(mismatches))
    return CheckResult("C3", True, f"all {len(template)} locked keys match")


def check_c4_validate(ds: dict, paper_run_ids: set[str] | None, stored_id: str) -> CheckResult:
    """C4: Run validate_dataset (Gates 1–3, 5–8 always; Gate 4 when paper_run_ids provided)."""
    errors = validate_dataset(ds, paper_run_ids=paper_run_ids, task_id=stored_id)
    if errors:
        return CheckResult("C4", False, " | ".join(errors))
    gate4_note = "incl. Gate 4" if paper_run_ids is not None else "Gate 4 skipped (no paper_data)"
    return CheckResult("C4", True, f"all gates pass ({gate4_note})")




def check_c6_archetype_consistency(ds: dict, dir_archetype: str) -> CheckResult:
    """C6: Archetype inferred from env_config must match the directory name."""
    cfg = ds  # flat schema
    inferred = infer_archetype(cfg)
    if inferred != dir_archetype:
        return CheckResult(
            "C6", False,
            f"inferred archetype={inferred!r} ≠ directory archetype={dir_archetype!r}"
        )
    return CheckResult("C6", True, f"archetype={dir_archetype!r} consistent")


def check_c7_variable_keys(ds: dict) -> CheckResult:
    """C7: All 12 variable keys must be present in env_config."""
    cfg = ds  # flat schema
    missing = VARIABLE_KEYS - set(cfg.keys())
    if missing:
        return CheckResult("C7", False, f"missing variable keys: {sorted(missing)}")
    return CheckResult("C7", True, f"all {len(VARIABLE_KEYS)} variable keys present")


def check_c8_no_extra_keys(ds: dict) -> CheckResult:
    """C8: env_config must contain only allowed keys (locked ∪ variable)."""
    cfg = ds  # flat schema (check root keys)
    extra = set(cfg.keys()) - ALLOWED_KEYS
    if extra:
        return CheckResult("C8", False, f"unexpected env_config keys: {sorted(extra)}")
    return CheckResult("C8", True, "no extra keys")


# ---------------------------------------------------------------------------
# Audit a single task file
# ---------------------------------------------------------------------------

def audit_task(task_path: Path, archetype: str, paper_run_ids: set[str] | None) -> tuple[dict, list[CheckResult]]:
    """Load and audit one task file. Returns (dataset_dict, [CheckResult, ...])."""
    ds = json.loads(task_path.read_text())
    stored_id = task_path.stem  # task_id is the filename (not stored inside file)
    checks = [
        check_c1_schema(ds),
        check_c2_uuid(ds, archetype, stored_id),
        check_c3_locked_keys(ds, archetype),
        check_c4_validate(ds, paper_run_ids, stored_id),
        check_c6_archetype_consistency(ds, archetype),
        check_c7_variable_keys(ds),
        check_c8_no_extra_keys(ds),
    ]
    return ds, checks


# ---------------------------------------------------------------------------
# Cross-task checks
# ---------------------------------------------------------------------------

def cross_check_x1_unique_ids(all_tasks: list[tuple[str, str, dict]]) -> list[str]:
    """X1: All task_ids must be globally unique."""
    seen: dict[str, list[str]] = {}
    for task_id, path, _ in all_tasks:
        seen.setdefault(task_id, []).append(path)
    return [
        f"task_id={tid!r} appears {len(paths)} times: {paths}"
        for tid, paths in seen.items()
        if len(paths) > 1
    ]


def cross_check_x2_unique_configs(all_tasks: list[tuple[str, str, dict]]) -> list[str]:
    """X2: No two tasks may have identical env_config (modulo order_record_dir)."""
    seen: dict[str, list[str]] = {}
    for task_id, path, ds in all_tasks:
        cfg = dict(ds)  # flat schema
        cfg.pop("order_record_dir", None)  # runtime-injected; exclude from comparison
        key = json.dumps(cfg, sort_keys=True)
        seen.setdefault(key, []).append(f"{task_id} ({path})")

    return [
        f"duplicate env_config across: {paths}"
        for paths in seen.values()
        if len(paths) > 1
    ]


def cross_check_x3_unique_identity_tuples(
    all_tasks: list[tuple[str, str, dict]],
    archetype_map: dict[str, str],
) -> list[str]:
    """X3: No two tasks may share the same identity tuple (archetype, seed, cats, ep, tier)."""
    seen: dict[tuple, list[str]] = {}
    for task_id, path, ds in all_tasks:
        cfg = ds  # flat schema
        archetype = archetype_map.get(task_id, "unknown")
        seed = cfg.get("global_random_seed")
        cats = tuple(sorted(cfg.get("selected_categories", [])))
        ep = match_elasticity_profile(cfg.get("category_effects", {}))
        tier = match_economic_tier(cfg)
        key = (archetype, seed, cats, ep, tier)
        seen.setdefault(key, []).append(f"{task_id} ({path})")

    return [
        f"duplicate identity tuple {key}: {paths}"
        for key, paths in seen.items()
        if len(paths) > 1
    ]


# ---------------------------------------------------------------------------
# Rich output helpers
# ---------------------------------------------------------------------------

def _print_task_header(task_id: str, archetype: str, path: str) -> None:
    short_id = task_id[:8]
    _print(f"\n{_bold(f'[{archetype}] {short_id}...')} {_dim(path)}")


def _print_check(cr: CheckResult, verbose: bool) -> None:
    if cr.passed:
        if verbose:
            _print(f"  {_ok(cr.name + ': ' + cr.details)}")
    else:
        _print(f"  {_fail(cr.name + ': ' + cr.details)}")


def _print_cross_header(label: str) -> None:
    _print(f"\n{_bold(label)}")


def _print_cross_result(label: str, errors: list[str]) -> None:
    if not errors:
        _print(f"  {_ok(label + ': no issues')}")
    else:
        _print(f"  {_fail(label + f': {len(errors)} issue(s)')}")
        for e in errors:
            _print(f"    {_dim(e)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="RetailBench dataset QC/audit")
    ap.add_argument(
        "--dataset-dir", default="zoro/harness/dataset",
        help="Root dataset directory containing {archetype}/{task_id}.json layout",
    )
    ap.add_argument(
        "--archetype",
        choices=list(ARCHETYPES.keys()),
        default=None,
        help="Limit audit to one archetype (default: all)",
    )
    ap.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show passing checks too (default: only failures)",
    )
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        _print(f"{_fail('Dataset directory not found:')} {dataset_dir}")
        sys.exit(1)

    # Load paper-run fingerprints for Gate 4 (None if paper_data_dir absent)
    if PAPER_DATA_DIR.exists():
        paper_run_ids: set[str] | None = _paper_run_ids_from_dir(PAPER_DATA_DIR)
        _print(f"Paper-run IDs : {len(paper_run_ids)} loaded from {PAPER_DATA_DIR}")
    else:
        paper_run_ids = None
        _print(_warn(f"paper_data_dir not found ({PAPER_DATA_DIR}) — Gate 4 disabled"))

    # ------------------------------------------------------------------
    # Discover task files
    # ------------------------------------------------------------------
    archetypes_to_scan = [args.archetype] if args.archetype else list(ARCHETYPES.keys())
    task_files: list[tuple[str, Path]] = []  # (archetype, path)

    for arch in archetypes_to_scan:
        arch_dir = dataset_dir / arch
        if not arch_dir.exists():
            _print(_warn(f"Archetype directory not found, skipping: {arch_dir}"))
            continue
        for f in sorted(arch_dir.glob("*.json")):
            if f.name != "manifest.json":
                task_files.append((arch, f))

    if not task_files:
        _print(_warn("No task files found."))
        sys.exit(0)

    _print(f"\n{_bold('RetailBench Dataset Audit')}")
    _print(f"Dataset dir : {dataset_dir.resolve()}")
    _print(f"Tasks found : {len(task_files)}")
    _print(f"Archetypes  : {', '.join(archetypes_to_scan)}")

    # ------------------------------------------------------------------
    # Per-task audit
    # ------------------------------------------------------------------
    _print(f"\n{_bold('── Per-task checks ──────────────────────────────────────────')}")

    all_tasks: list[tuple[str, str, dict]] = []   # (task_id, rel_path, ds)
    archetype_map: dict[str, str] = {}            # task_id → archetype
    total_tasks = 0
    failed_tasks = 0

    for arch, path in task_files:
        total_tasks += 1
        rel_path = str(path.relative_to(dataset_dir.parent.parent)
                       if dataset_dir.parent.parent.exists() else path)

        try:
            ds, checks = audit_task(path, arch, paper_run_ids)
        except Exception as exc:
            _print(f"\n{_fail(f'[{arch}] {path.name}: failed to load/parse — {exc}')}")
            failed_tasks += 1
            continue

        task_id = path.stem  # flat schema: task_id is the filename
        archetype_map[task_id] = arch
        all_tasks.append((task_id, str(path), ds))

        task_passed = all(c.passed for c in checks)
        if not task_passed:
            failed_tasks += 1

        _print_task_header(task_id, arch, path.name)

        passed_count = sum(1 for c in checks if c.passed)
        status = _ok(f"{passed_count}/{len(checks)} checks") if task_passed else _fail(f"{passed_count}/{len(checks)} checks")
        _print(f"  {status}")

        for cr in checks:
            _print_check(cr, verbose=args.verbose)

    # ------------------------------------------------------------------
    # Cross-task checks
    # ------------------------------------------------------------------
    _print(f"\n{_bold('── Cross-task checks ────────────────────────────────────────')}")

    x1_errors = cross_check_x1_unique_ids(all_tasks)
    x2_errors = cross_check_x2_unique_configs(all_tasks)
    x3_errors = cross_check_x3_unique_identity_tuples(all_tasks, archetype_map)

    _print_cross_result("X1 unique task_ids", x1_errors)
    _print_cross_result("X2 unique env_configs", x2_errors)
    _print_cross_result("X3 unique identity tuples", x3_errors)

    cross_failed = bool(x1_errors or x2_errors or x3_errors)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _print(f"\n{_bold('── Summary ──────────────────────────────────────────────────')}")
    _print(f"  Tasks audited    : {total_tasks}")
    _print(f"  Tasks passed     : {total_tasks - failed_tasks}")
    _print(f"  Tasks failed     : {failed_tasks}")
    _print(f"  Cross-task issues: {len(x1_errors) + len(x2_errors) + len(x3_errors)}")

    if failed_tasks == 0 and not cross_failed:
        _print(f"\n{_ok(_bold('ALL CHECKS PASSED'))}")
        sys.exit(0)
    else:
        _print(f"\n{_fail(_bold('AUDIT FAILED — see failures above'))}")
        sys.exit(1)


if __name__ == "__main__":
    main()
