"""Parallel launcher for harness/run_env.py in dataset mode.

Fires N `run_env.py --dataset <D> --model <M>` subprocesses concurrently under a
ThreadPoolExecutor. Each subprocess writes to its own
`<out_dir>/<task_id>/<model_short>/run_N/` tree, so different (dataset, model)
pairs never collide on disk. Concurrent invocations of the *same*
(dataset, model) pair are also safe because `_pick_next_run_dir` in run_env.py
atomically claims a fresh `run_N` via `mkdir(exist_ok=False)`.

Guardrails Oracle flagged that tiers 1+2 do not catch:
  - SIGINT propagation: Ctrl+C on the launcher forwards SIGINT to every live
    subprocess before shutting the pool down. Without this, subprocesses keep
    running and burning API budget after the launcher exits.
  - Silent zero-exit on failure: after subprocess exits rc=0, we assert that
    the expected `run_env.json` artifact exists inside the latest run_N dir.
    A run that swallows an error and returns 0 without producing output is
    counted as failure.

API keys are inherited via `env=os.environ.copy()`. Never passed on argv.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_MODEL_SHORT_MAP: dict[str, str] = {
    "opus": "claude-opus-4.8",
    "claude-opus-4-8": "claude-opus-4.8",
    "claude-opus-4.8": "claude-opus-4.8",
    "sonnet": "claude-sonnet-4.5",
    "claude-sonnet-4-5": "claude-sonnet-4.5",
    "claude-sonnet-4.5": "claude-sonnet-4.5",
    "haiku": "claude-haiku-4.5",
    "claude-haiku-4-5": "claude-haiku-4.5",
    "claude-haiku-4.5": "claude-haiku-4.5",
    "sol": "gpt-5.6",
    "terra": "gpt-5.6",
    "luna": "gpt-5.6",
    "gpt-5": "gpt-5.6",
    "gpt5": "gpt-5.6",
    "gpt-5.6": "gpt-5.6",
    "gpt-5.6-sol": "gpt-5.6",
    "codex": "gpt-5.6",
    "codex-cc": "gpt-5.6",
    "codex-mini": "gpt-5.6",
    "gpt5-codex-cc": "gpt-5.6",
}


def _resolve_model_short(model: str) -> str:
    key = (model or "").strip().lower()
    if not key:
        return "unknown-model"
    if key in _MODEL_SHORT_MAP:
        return _MODEL_SHORT_MAP[key]
    return re.sub(r"[^A-Za-z0-9._\-]+", "_", key)


def _load_task_id(dataset_path: Path) -> str:
    with open(dataset_path, encoding="utf-8") as f:
        ds = json.load(f)
    task_id = ds.get("task_id")
    if not task_id:
        task_id = dataset_path.stem
    return str(task_id)


_LIVE_PROCS: set[subprocess.Popen] = set()
_PROCS_LOCK = threading.Lock()

_LOG_COUNTERS: dict[tuple[str, str], int] = {}
_LOG_COUNTER_LOCK = threading.Lock()


def _register(p: subprocess.Popen) -> None:
    with _PROCS_LOCK:
        _LIVE_PROCS.add(p)


def _unregister(p: subprocess.Popen) -> None:
    with _PROCS_LOCK:
        _LIVE_PROCS.discard(p)


def _terminate_all_live(sig: int = signal.SIGINT) -> None:
    with _PROCS_LOCK:
        procs = list(_LIVE_PROCS)
    for p in procs:
        with contextlib.suppress(ProcessLookupError, OSError):
            p.send_signal(sig)


def _allocate_log_path(log_dir: Path, task_id: str, model_short: str) -> Path:
    key = (task_id, model_short)
    with _LOG_COUNTER_LOCK:
        n = _LOG_COUNTERS.get(key, 0)
        _LOG_COUNTERS[key] = n + 1
    stem = f"{task_id}_{model_short}"
    return log_dir / (f"{stem}.log" if n == 0 else f"{stem}_{n}.log")


def _max_existing_run_n(out_dir: Path, task_id: str, model_short: str) -> int:
    """Return the largest existing run_N under <out_dir>/<task_id>/<model_short>, else 0."""
    model_dir = out_dir / task_id / model_short
    if not model_dir.is_dir():
        return 0
    max_n = 0
    for child in model_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("run_"):
            continue
        suffix = child.name[len("run_"):]
        if suffix.isdigit():
            n = int(suffix)
            if n > max_n:
                max_n = n
    return max_n


def _verify_run_output(
    out_dir: Path, task_id: str, model_short: str, min_n: int = 0
) -> Optional[Path]:
    """Return the newest run_N dir under <out_dir>/<task_id>/<model_short> that
    contains run_env.json AND has N > min_n, else None.

    min_n=0 accepts any run_N. Passing the pre-launch max lets callers reject
    stale artifacts from earlier invocations that silently exited 0."""
    model_dir = out_dir / task_id / model_short
    if not model_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for child in model_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("run_"):
            continue
        suffix = child.name[len("run_"):]
        if suffix.isdigit():
            candidates.append((int(suffix), child))
    for n, rd in sorted(candidates, reverse=True):
        if n > min_n and (rd / "run_env.json").exists():
            return rd
    return None


@dataclass
class TaskSpec:
    dataset: Path
    model: str
    task_id: str
    model_short: str


@dataclass
class TaskResult:
    spec: TaskSpec
    returncode: int
    duration_s: float
    output_dir: Optional[Path]
    output_verified: bool
    log_path: Optional[Path]
    error: Optional[str]

    @property
    def success(self) -> bool:
        return self.returncode == 0 and self.output_verified


def _run_one(
    spec: TaskSpec,
    *,
    python: str,
    run_env: Path,
    out_dir: Path,
    log_dir: Optional[Path],
    timeout: Optional[float],
    extra: list[str],
    cancel_event: threading.Event,
) -> TaskResult:
    if cancel_event.is_set():
        return TaskResult(
            spec=spec, returncode=-2, duration_s=0.0, output_dir=None,
            output_verified=False, log_path=None, error="cancelled before start",
        )

    argv = [
        python, str(run_env),
        "--dataset", str(spec.dataset),
        "--model", spec.model,
        "--out_dir", str(out_dir),
        *extra,
    ]
    env = os.environ.copy()
    pre_n = _max_existing_run_n(out_dir, spec.task_id, spec.model_short)

    log_path: Optional[Path] = None
    log_handle = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = _allocate_log_path(log_dir, spec.task_id, spec.model_short)
        log_handle = open(log_path, "w", encoding="utf-8", buffering=1)
        log_handle.write(f"# argv: {argv}\n# pid: pending\n")
        log_handle.flush()

    stdout_target = log_handle if log_handle is not None else None
    stderr_target = subprocess.STDOUT if log_handle is not None else None

    start = time.monotonic()
    proc: Optional[subprocess.Popen] = None
    error: Optional[str] = None
    returncode: int = -1
    try:
        proc = subprocess.Popen(
            argv,
            env=env,
            stdout=stdout_target,
            stderr=stderr_target,
        )
        _register(proc)
        if log_handle is not None:
            log_handle.write(f"# pid: {proc.pid}\n")
            log_handle.flush()
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            error = f"timeout after {timeout}s"
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError, OSError):
                    proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=5)
            returncode = proc.returncode if proc.returncode is not None else -1
    except Exception as e:
        error = f"launch failure: {e!r}"
    finally:
        if proc is not None:
            _unregister(proc)
        if log_handle is not None:
            with contextlib.suppress(Exception):
                log_handle.close()

    duration = time.monotonic() - start
    output_dir = _verify_run_output(out_dir, spec.task_id, spec.model_short, min_n=pre_n) if returncode == 0 else None
    output_verified = output_dir is not None
    if returncode == 0 and not output_verified and error is None:
        error = "subprocess exited 0 but no run_env.json produced"

    return TaskResult(
        spec=spec,
        returncode=returncode,
        duration_s=duration,
        output_dir=output_dir,
        output_verified=output_verified,
        log_path=log_path,
        error=error,
    )


def _build_task_specs(datasets: list[Path], models: list[str]) -> list[TaskSpec]:
    task_ids: dict[Path, str] = {}
    for d in datasets:
        task_ids[d] = _load_task_id(d)
    specs: list[TaskSpec] = []
    for d in datasets:
        for m in models:
            specs.append(TaskSpec(
                dataset=d,
                model=m,
                task_id=task_ids[d],
                model_short=_resolve_model_short(m),
            ))
    return specs


def _format_result(r: TaskResult) -> str:
    status = "OK " if r.success else "FAIL"
    out = str(r.output_dir) if r.output_dir else "-"
    err = f" err={r.error}" if r.error else ""
    return (
        f"[{status}] task={r.spec.task_id} model={r.spec.model_short} "
        f"rc={r.returncode} t={r.duration_s:.1f}s out={out}{err}"
    )


def _install_sigint_handler(cancel_event: threading.Event) -> None:
    def _handler(_signum, _frame):
        if cancel_event.is_set():
            return
        cancel_event.set()
        print("[run_parallel] SIGINT received; forwarding to live subprocesses",
              file=sys.stderr, flush=True)
        _terminate_all_live(signal.SIGINT)

    signal.signal(signal.SIGINT, _handler)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parallel launcher for harness/run_env.py in dataset mode",
    )
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="Dataset JSON files (one --dataset per subprocess call).")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model names; cross-product with --datasets.")
    parser.add_argument("--out_dir", required=True,
                        help="Shared root; each subprocess writes to <out_dir>/<task_id>/<model_short>/run_N/")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--fail_fast", action="store_true",
                        help="On first task failure, cancel pending and terminate live subprocesses.")
    parser.add_argument("--log_dir", default=None,
                        help="Capture per-task stdout+stderr to <log_dir>/<task_id>_<model_short>.log")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Per-subprocess wall-clock timeout in seconds.")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter to launch subprocesses with.")
    parser.add_argument("--run_env",
                        default=str(Path(__file__).resolve().parent / "run_env.py"),
                        help="Path to run_env.py.")
    parser.add_argument("extra", nargs=argparse.REMAINDER,
                        help="Extra args passed through to run_env.py (after --).")
    args = parser.parse_args()

    datasets = [Path(d).resolve() for d in args.datasets]
    for d in datasets:
        if not d.is_file():
            print(f"[run_parallel] dataset not found: {d}", file=sys.stderr)
            return 2

    run_env = Path(args.run_env).resolve()
    if not run_env.is_file():
        print(f"[run_parallel] run_env.py not found: {run_env}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir).resolve() if args.log_dir else None

    extra = list(args.extra or [])
    if extra and extra[0] == "--":
        extra = extra[1:]

    if any(a == "--config_type" for a in extra):
        print("[run_parallel] ERROR: --config_type is unsafe under parallelism; use --dataset mode only",
              file=sys.stderr)
        return 2

    specs = _build_task_specs(datasets, args.models)
    if not specs:
        print("[run_parallel] no tasks to run", file=sys.stderr)
        return 2

    cancel_event = threading.Event()
    _install_sigint_handler(cancel_event)

    print(f"[run_parallel] launching {len(specs)} tasks with max_workers={args.max_workers}",
          flush=True)

    results: list[TaskResult] = []
    exit_code = 0
    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            future_to_spec: dict[Future[TaskResult], TaskSpec] = {}
            for spec in specs:
                fut = pool.submit(
                    _run_one, spec,
                    python=args.python, run_env=run_env, out_dir=out_dir,
                    log_dir=log_dir, timeout=args.timeout, extra=extra,
                    cancel_event=cancel_event,
                )
                future_to_spec[fut] = spec

            for fut in list(future_to_spec):
                try:
                    r = fut.result()
                except Exception as e:
                    spec = future_to_spec[fut]
                    r = TaskResult(
                        spec=spec, returncode=-1, duration_s=0.0,
                        output_dir=None, output_verified=False,
                        log_path=None, error=f"future error: {e!r}",
                    )
                results.append(r)
                print(_format_result(r), flush=True)
                if not r.success and args.fail_fast and not cancel_event.is_set():
                    cancel_event.set()
                    print("[run_parallel] fail_fast: cancelling pending and terminating live",
                          file=sys.stderr, flush=True)
                    for pending in future_to_spec:
                        if not pending.done():
                            pending.cancel()
                    _terminate_all_live(signal.SIGTERM)
    except KeyboardInterrupt:
        exit_code = 130

    n_ok = sum(1 for r in results if r.success)
    n_fail = len(results) - n_ok
    print(f"[run_parallel] done: {n_ok} ok, {n_fail} fail, {len(specs) - len(results)} skipped",
          flush=True)

    if exit_code == 0 and (n_fail > 0 or len(results) < len(specs)):
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
