#!/usr/bin/env python3
"""Tier-2 end-to-end test for run_parallel.py orchestration.

Uses a stub script in place of the real run_env.py to validate:
  * cross-product of --datasets and --models spawns N*M subprocesses
  * output tree is nested as <out_dir>/<task_id>/<model_short>/run_N/
  * silent-zero-exit guard fires when subprocess exits 0 but writes no run_env.json
  * --fail_fast cancels pending futures on first failure
  * --log_dir captures per-task stdout+stderr with collision-safe suffixes
  * exit codes: 0 = all pass, 1 = any fail

The stub reads run_parallel.py's argv (--dataset, --model, --out_dir), extracts
task_id from the dataset JSON, mimics run_env.py's dataset-mode nesting, writes
run_env.json (or skips it based on env var), and exits.

Exits 0 on all scenarios passing, non-zero on any failure.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
RUN_PARALLEL = HARNESS_DIR / "run_parallel.py"

STUB_SUCCESS = textwrap.dedent(
    """
    #!/usr/bin/env python3
    import argparse, json, os, sys, re
    from pathlib import Path

    _MODEL_SHORT_MAP = {
        "opus": "claude-opus-4.8",
        "sonnet": "claude-sonnet-4.5",
        "haiku": "claude-haiku-4.5",
        "gpt-5.6": "gpt-5.6",
    }

    def _resolve_model_short(model):
        key = (model or "").strip().lower()
        if not key:
            return "unknown-model"
        if key in _MODEL_SHORT_MAP:
            return _MODEL_SHORT_MAP[key]
        return re.sub(r"[^A-Za-z0-9._\\-]+", "_", key)

    def _pick_next_run_dir(model_dir: Path) -> Path:
        model_dir.mkdir(parents=True, exist_ok=True)
        used = []
        for child in model_dir.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if not name.startswith("run_"):
                continue
            suffix = name[len("run_"):]
            if suffix.isdigit():
                used.append(int(suffix))
        next_n = (max(used) + 1) if used else 1
        while True:
            candidate = model_dir / f"run_{next_n}"
            try:
                candidate.mkdir(exist_ok=False)
                return candidate
            except FileExistsError:
                next_n += 1

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out_dir", required=True)
    args, _rest = p.parse_known_args()

    with open(args.dataset, encoding="utf-8") as f:
        ds = json.load(f)
    task_id = ds["task_id"]
    model_short = _resolve_model_short(args.model)
    model_dir = Path(args.out_dir) / str(task_id) / model_short
    out = _pick_next_run_dir(model_dir)

    print(f"[stub] task={task_id} model={args.model} out={out}", flush=True)

    behavior = os.environ.get("STUB_BEHAVIOR", "success")
    if behavior == "success":
        (out / "run_env.json").write_text(json.dumps({"ok": True, "task_id": task_id}))
        sys.exit(0)
    elif behavior == "silent_zero":
        # exits 0 but writes NO run_env.json (triggers silent-zero-exit guard)
        sys.exit(0)
    elif behavior == "fail":
        print("[stub] simulated failure", file=sys.stderr, flush=True)
        sys.exit(7)
    elif behavior == "selective_fail":
        # fail only for task_2, succeed for others
        if "task_2" in str(task_id):
            print("[stub] task_2 failed", file=sys.stderr, flush=True)
            sys.exit(9)
        (out / "run_env.json").write_text(json.dumps({"ok": True, "task_id": task_id}))
        sys.exit(0)
    else:
        sys.exit(99)
    """
).strip()


def _make_datasets(root: Path, task_ids: list[str]) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for tid in task_ids:
        p = root / f"{tid}.json"
        p.write_text(json.dumps({"task_id": tid, "config_type": "still_middle"}))
        paths.append(p)
    return paths


def _run_launcher(args: list[str], env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(RUN_PARALLEL), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def _scenario_success_cross_product(tmp: Path) -> list[str]:
    """3 datasets x 2 models = 6 tasks, all succeed."""
    failures: list[str] = []
    ds_dir = tmp / "ds_success"
    out_dir = tmp / "out_success"
    log_dir = tmp / "logs_success"
    stub = tmp / "stub_success.py"
    stub.write_text(STUB_SUCCESS)

    tids = ["task_a", "task_b", "task_c"]
    ds_paths = _make_datasets(ds_dir, tids)
    models = ["opus", "gpt-5.6"]

    r = _run_launcher(
        [
            "--datasets", *[str(p) for p in ds_paths],
            "--models", *models,
            "--out_dir", str(out_dir),
            "--max_workers", "4",
            "--log_dir", str(log_dir),
            "--python", sys.executable,
            "--run_env", str(stub),
        ],
        env_extra={"STUB_BEHAVIOR": "success"},
    )

    if r.returncode != 0:
        failures.append(f"[success] exit={r.returncode} expected 0; stderr:\n{r.stderr}")

    expected_shorts = {"opus": "claude-opus-4.8", "gpt-5.6": "gpt-5.6"}
    for tid in tids:
        for model, short in expected_shorts.items():
            run_env_json = out_dir / tid / short / "run_1" / "run_env.json"
            if not run_env_json.exists():
                failures.append(f"[success] missing {run_env_json}")

    if log_dir.exists():
        log_files = list(log_dir.glob("*.log"))
        if len(log_files) != len(tids) * len(models):
            failures.append(f"[success] expected {len(tids) * len(models)} log files, got {len(log_files)}")
    else:
        failures.append(f"[success] log_dir {log_dir} not created")

    return failures


def _scenario_silent_zero_exit(tmp: Path) -> list[str]:
    """Stub exits 0 but writes no run_env.json -> launcher must report failure."""
    failures: list[str] = []
    ds_dir = tmp / "ds_silent"
    out_dir = tmp / "out_silent"
    stub = tmp / "stub_silent.py"
    stub.write_text(STUB_SUCCESS)

    ds_paths = _make_datasets(ds_dir, ["task_silent"])

    r = _run_launcher(
        [
            "--datasets", *[str(p) for p in ds_paths],
            "--models", "opus",
            "--out_dir", str(out_dir),
            "--python", sys.executable,
            "--run_env", str(stub),
        ],
        env_extra={"STUB_BEHAVIOR": "silent_zero"},
    )

    if r.returncode == 0:
        failures.append(
            f"[silent_zero] launcher exited 0 but should have detected missing run_env.json; stdout:\n{r.stdout}"
        )
    if "no run_env.json produced" not in r.stdout and "no run_env.json produced" not in r.stderr:
        failures.append(
            f"[silent_zero] expected 'no run_env.json produced' in output; stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        )
    return failures


def _scenario_fail_fast(tmp: Path) -> list[str]:
    """One task fails -> --fail_fast should cancel remaining."""
    failures: list[str] = []
    ds_dir = tmp / "ds_ff"
    out_dir = tmp / "out_ff"
    stub = tmp / "stub_ff.py"
    stub.write_text(STUB_SUCCESS)

    tids = [f"task_{i}" for i in range(1, 5)]
    ds_paths = _make_datasets(ds_dir, tids)

    r = _run_launcher(
        [
            "--datasets", *[str(p) for p in ds_paths],
            "--models", "opus",
            "--out_dir", str(out_dir),
            "--max_workers", "1",
            "--fail_fast",
            "--python", sys.executable,
            "--run_env", str(stub),
        ],
        env_extra={"STUB_BEHAVIOR": "selective_fail"},
    )

    if r.returncode == 0:
        failures.append(f"[fail_fast] launcher exited 0 but selective_fail should return non-zero")

    return failures


def _scenario_log_collision(tmp: Path) -> list[str]:
    """Same (task_id, model) submitted twice via duplicate dataset -> log suffix increments."""
    failures: list[str] = []
    ds_dir = tmp / "ds_collide"
    out_dir = tmp / "out_collide"
    log_dir = tmp / "logs_collide"
    stub = tmp / "stub_collide.py"
    stub.write_text(STUB_SUCCESS)

    # Two dataset files with SAME task_id
    ds_dir.mkdir(parents=True)
    for name in ("a", "b"):
        (ds_dir / f"{name}.json").write_text(json.dumps({"task_id": "task_dup"}))

    r = _run_launcher(
        [
            "--datasets", str(ds_dir / "a.json"), str(ds_dir / "b.json"),
            "--models", "opus",
            "--out_dir", str(out_dir),
            "--max_workers", "2",
            "--log_dir", str(log_dir),
            "--python", sys.executable,
            "--run_env", str(stub),
        ],
        env_extra={"STUB_BEHAVIOR": "success"},
    )

    if r.returncode != 0:
        failures.append(f"[log_collision] exit={r.returncode} expected 0; stderr:\n{r.stderr}")

    log_files = sorted(p.name for p in log_dir.glob("task_dup*.log")) if log_dir.exists() else []
    # First = task_dup_claude-opus-4.8.log; second = task_dup_claude-opus-4.8_1.log
    expected = {"task_dup_claude-opus-4.8.log", "task_dup_claude-opus-4.8_1.log"}
    if set(log_files) != expected:
        failures.append(f"[log_collision] expected log names {expected}, got {log_files}")

    # Both runs should nest into same model_dir -> run_1 and run_2
    run_dirs = sorted((out_dir / "task_dup" / "claude-opus-4.8").iterdir()) if (out_dir / "task_dup" / "claude-opus-4.8").exists() else []
    run_names = sorted(p.name for p in run_dirs if p.is_dir())
    if run_names != ["run_1", "run_2"]:
        failures.append(f"[log_collision] expected run_1+run_2, got {run_names}")

    return failures


def _scenario_repeat_invocation_silent_zero(tmp: Path) -> list[str]:
    """Oracle C2 regression: repeat invocation with silent_zero must NOT be verified by stale run_1.

    1st: STUB_BEHAVIOR=success  -> creates run_1/run_env.json, exit 0.
    2nd: STUB_BEHAVIOR=silent_zero on same (out_dir, task_id, model)
         -> creates run_2/ but no run_env.json, launcher's pre_n snapshot
            must reject run_1 as stale -> non-zero exit.
    """
    failures: list[str] = []
    ds_dir = tmp / "ds_repeat"
    out_dir = tmp / "out_repeat"
    stub = tmp / "stub_repeat.py"
    stub.write_text(STUB_SUCCESS)

    ds_paths = _make_datasets(ds_dir, ["task_repeat"])

    r1 = _run_launcher(
        [
            "--datasets", *[str(p) for p in ds_paths],
            "--models", "opus",
            "--out_dir", str(out_dir),
            "--python", sys.executable,
            "--run_env", str(stub),
        ],
        env_extra={"STUB_BEHAVIOR": "success"},
    )
    if r1.returncode != 0:
        failures.append(f"[repeat] 1st invocation exit={r1.returncode} expected 0; stderr:\n{r1.stderr}")
    run1_json = out_dir / "task_repeat" / "claude-opus-4.8" / "run_1" / "run_env.json"
    if not run1_json.exists():
        failures.append(f"[repeat] 1st invocation did not create {run1_json}")

    r2 = _run_launcher(
        [
            "--datasets", *[str(p) for p in ds_paths],
            "--models", "opus",
            "--out_dir", str(out_dir),
            "--python", sys.executable,
            "--run_env", str(stub),
        ],
        env_extra={"STUB_BEHAVIOR": "silent_zero"},
    )
    if r2.returncode == 0:
        failures.append(
            f"[repeat] 2nd invocation exited 0 but should have detected missing run_2/run_env.json; "
            f"stdout:\n{r2.stdout}"
        )
    if "no run_env.json produced" not in r2.stdout and "no run_env.json produced" not in r2.stderr:
        failures.append(
            f"[repeat] 2nd invocation missing 'no run_env.json produced' guard; "
            f"stdout:\n{r2.stdout}\nstderr:\n{r2.stderr}"
        )

    model_dir = out_dir / "task_repeat" / "claude-opus-4.8"
    run_dirs = sorted(p.name for p in model_dir.iterdir() if p.is_dir()) if model_dir.exists() else []
    if run_dirs != ["run_1", "run_2"]:
        failures.append(f"[repeat] expected run_1+run_2 on disk, got {run_dirs}")

    if not run1_json.exists():
        failures.append("[repeat] run_1/run_env.json disappeared after 2nd invocation")

    run2_json = model_dir / "run_2" / "run_env.json"
    if run2_json.exists():
        failures.append("[repeat] run_2/run_env.json unexpectedly exists after silent_zero invocation")

    return failures


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="run_parallel_e2e_") as tmp_str:
        tmp = Path(tmp_str)
        all_failures: dict[str, list[str]] = {}

        for name, fn in [
            ("success_cross_product", _scenario_success_cross_product),
            ("silent_zero_exit", _scenario_silent_zero_exit),
            ("fail_fast", _scenario_fail_fast),
            ("log_collision", _scenario_log_collision),
            ("repeat_invocation_silent_zero", _scenario_repeat_invocation_silent_zero),
        ]:
            print(f"[e2e] running scenario: {name}")
            fails = fn(tmp)
            if fails:
                all_failures[name] = fails
                for f in fails:
                    print(f"  FAIL: {f}")
            else:
                print(f"  PASS")

        if all_failures:
            print(f"\n[e2e] {len(all_failures)} scenarios FAILED: {list(all_failures.keys())}")
            return 1
        print("\n[e2e] PASS \u2014 all 5 scenarios passed")
        return 0


if __name__ == "__main__":
    sys.exit(main())
