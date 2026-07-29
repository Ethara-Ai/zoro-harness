"""run_council.py — LLM council driver (hardened).

Reads council_inputs.json (per-rubric prompt + metadata from prepare_inputs.py), fans out
each rubric to the 3 judges over the local Claude/Codex bridges in a thread pool, aggregates
per rubric (confidence floor -> majority -> tie=0 -> all-null=null), and writes
council_results.json with the full audit trail. See .sisyphus/plans/retailbench-llm-council.md.

Resolves the review-ledger issues:
  D1  no-opportunity rubrics (status="no_opportunity") are short-circuited -> null, no judge call.
  D3  validate_panel() preflight fails LOUD; --allow-degraded is explicit and surfaced.
  D4  judge model ids pinned in judges.yaml; preflight ping confirms they resolve.
  D5  robust JSON extraction (fences + last balanced object) + verdict coercion.
  D6  council_results carries is_positive/score/status + per-judge abstain_rate.
  D7  per-bridge BoundedSemaphore so the 2-Claude/1-GPT split can't overload one subscription.
  D8  per-judge abstain_rate recorded and surfaced.
  #4  optional on-disk verdict cache keyed by (prompt, model) -> stable, free re-grades.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

# Network/config deps are LAZY so the pure-logic functions (extract_json, coerce_verdict,
# aggregate_rubric, run_council with an injected call_one) import and test on stdlib alone.
try:
    import requests
except Exception:  # pragma: no cover
    requests = None

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

_HERE = Path(__file__).resolve().parent
DEFAULT_PANEL = _HERE / "judges.yaml"

# reasoning-model prefixes that REJECT the `temperature` param (Codex bridge server.py:52,
# verified live: gpt-5.5 + temperature => HTTP 400). No temperature is sent to these.
_REASONING_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4", "codex")


def _is_reasoning_model(model: str) -> bool:
    m = (model or "").lower()
    return any(m.startswith(p) for p in _REASONING_PREFIXES)


# ------------------------------------------------------------------ parsing (D5)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(content: str) -> dict:
    """Strip code fences, then take the LAST complete top-level {...} object. Robust to
    reasoning prose and stray braces (D5)."""
    if not content or not content.strip():
        raise ValueError("empty content")
    m = _FENCE.search(content)
    text = m.group(1) if m else content
    depth = 0
    start = -1
    last: Optional[str] = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                last = text[start : i + 1]
    if last is None:
        raise ValueError(f"no JSON object in: {content[:200]!r}")
    return json.loads(last)


def coerce_verdict(v: Any) -> Optional[int]:
    """Models stringify; don't silently drop the vote (D5)."""
    if v in (0, 1, None):
        return v
    if isinstance(v, bool):
        return int(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "present", "met"):
        return 1
    if s in ("0", "false", "no", "absent", "unmet"):
        return 0
    if s in ("null", "none", "", "n/a", "unassessed"):
        return None
    raise ValueError(f"uncoercible verdict {v!r}")


# ------------------------------------------------------------------ config

def load_panel(path: Path = DEFAULT_PANEL) -> dict:
    if yaml is None:
        raise RuntimeError("pyyaml is required to read judges.yaml")
    cfg = yaml.safe_load(path.read_text())
    for req in ("panel", "bridges"):
        if req not in cfg:
            raise ValueError(f"judges.yaml missing '{req}'")
    return cfg


def _bridge_base(b: dict) -> str:
    return os.environ.get(b.get("base_url_env", ""), b["default_base_url"])


# ------------------------------------------------------------------ per-bridge concurrency (D7)

_SEM_LOCK = threading.Lock()
_SEM: dict[str, threading.BoundedSemaphore] = {}


def _sem(bridge: str, n: int) -> threading.BoundedSemaphore:
    with _SEM_LOCK:
        if bridge not in _SEM:
            _SEM[bridge] = threading.BoundedSemaphore(max(1, n))
        return _SEM[bridge]


# ------------------------------------------------------------------ cache (#4)

class VerdictCache:
    """Optional on-disk cache keyed by (prompt, model) so re-grades are free & stable."""

    def __init__(self, path: Optional[Path]):
        self.path = path
        self._d: dict[str, dict] = {}
        if path and path.exists():
            try:
                self._d = json.loads(path.read_text())
            except Exception:
                self._d = {}

    @staticmethod
    def key(prompt: str, model: str) -> str:
        return hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()

    def get(self, prompt: str, model: str) -> Optional[dict]:
        return self._d.get(self.key(prompt, model)) if self.path else None

    def put(self, prompt: str, model: str, rec: dict) -> None:
        if not self.path or rec.get("error") is not None:
            return  # never cache failures
        self._d[self.key(prompt, model)] = rec

    def flush(self) -> None:
        if self.path:
            self.path.write_text(json.dumps(self._d))


# ------------------------------------------------------------------ single judge call

def call_one(number: str, judge: dict, prompt: str, panel: dict, bridges: dict,
             cache: Optional[VerdictCache] = None) -> dict:
    """POST one (rubric, judge) prompt. NEVER raises: any failure -> verdict=null (abstain)."""
    if cache is not None:
        hit = cache.get(prompt, judge["id"])
        if hit is not None:
            return {**hit, "number": number, "model": judge["id"], "cached": True}

    b = bridges[judge["bridge"]]
    base = _bridge_base(b)
    secret = os.environ.get(b["secret_env"], "")
    headers = {
        "x-zoro-bridge-secret": secret,
        "Authorization": f"Bearer {secret}",
        "Content-Type": "application/json",
    }
    body = {
        "model": judge["model"],
        "stream": False,                        # non-streaming so real HTTP errors surface
        "messages": [{"role": "user", "content": prompt}],
    }
    # temperature: Claude models want temp=0; GPT-5 / o-series REASONING models 400 on it
    # (verified live: gpt-5.5 + temperature => HTTP 400). Send temp only when the model accepts it.
    # Per-judge `send_temperature` overrides the family heuristic.
    send_temp = judge.get("send_temperature")
    if send_temp is None:
        send_temp = not _is_reasoning_model(judge["model"])
    if send_temp:
        body["temperature"] = panel.get("temperature", 0.0)
    url = f"{base}/chat/completions"
    to = (panel.get("connect_timeout_sec", 10), panel.get("read_timeout_sec", 180))
    retries = panel.get("max_retries", 2)
    backoff = panel.get("retry_backoff_sec", 3)
    per_bridge = panel.get("per_bridge_workers", 4)

    if requests is None:   # missing dep -> abstain, never raise (keeps call_one total)
        return {"number": number, "model": judge["id"], "verdict": None, "confidence": 0.0,
                "evidence": "", "error": "requests package not installed"}
    last = ""
    for attempt in range(retries + 1):
        try:
            with _sem(judge["bridge"], per_bridge):
                r = requests.post(url, json=body, headers=headers, timeout=to)
            if r.status_code == 200:
                content = (r.json().get("choices") or [{}])[0].get("message", {}).get("content") or ""
                obj = extract_json(content)
                rec = {
                    "number": number, "model": judge["id"],
                    "verdict": coerce_verdict(obj.get("verdict")),
                    "confidence": float(obj.get("confidence", 0.0)),
                    "evidence": str(obj.get("evidence", ""))[:500],
                    "error": None,
                }
                if cache is not None:
                    cache.put(prompt, judge["id"], {k: rec[k] for k in
                                                    ("verdict", "confidence", "evidence", "error")})
                return rec
            last = f"http {r.status_code}: {r.text[:200]}"
            if r.status_code == 429:               # honor bridge cap Retry-After
                time.sleep(float(r.headers.get("Retry-After", backoff)))
                continue
        except Exception as e:  # noqa: BLE001 - transport/parse/anything -> abstain
            last = f"{type(e).__name__}: {e}"
        if attempt < retries:
            time.sleep(backoff * (attempt + 1))
    return {"number": number, "model": judge["id"], "verdict": None,
            "confidence": 0.0, "evidence": "", "error": last}


# ------------------------------------------------------------------ preflight (D3/D4)

def validate_panel(cfg: dict) -> list[str]:
    """Return a list of problems; empty == healthy. Checks health, secret, and a 1-token
    ping per judge (confirms the pinned model id resolves)."""
    if requests is None:
        return ["'requests' package not installed (pip install requests)"]
    problems: list[str] = []
    for name, b in cfg["bridges"].items():
        if not os.environ.get(b["secret_env"]):
            problems.append(f"{b['secret_env']} not set (bridge '{name}')")
        try:
            hz = requests.get(b["healthz"], timeout=5).json()
            if hz.get("ok") is not True:
                problems.append(f"bridge '{name}' /healthz not ok: {hz}")
        except Exception as e:  # noqa: BLE001
            problems.append(f"bridge '{name}' unreachable: {e}")
    if problems:
        return problems  # don't ping models if bridges/secrets are already broken
    ping = 'Reply with exactly this JSON and nothing else: {"verdict":1,"confidence":1.0,"evidence":"ping"}'
    for j in cfg["panel"]["judges"]:
        rec = call_one("ping", j, ping, cfg["panel"], cfg["bridges"])
        if rec["error"] is not None:
            problems.append(f"judge '{j['id']}' ({j['model']}) ping failed: {rec['error']}")
    return problems


# ------------------------------------------------------------------ aggregation

def aggregate_rubric(subs: list[dict], floor: float) -> Optional[int]:
    """eval-plan §3.5: confidence<floor -> null; majority of survivors; tie -> 0; all-null -> null."""
    votes = [s["verdict"] for s in subs
             if s.get("verdict") in (0, 1) and s.get("confidence", 0.0) >= floor]
    if not votes:
        return None
    return 1 if votes.count(1) > votes.count(0) else 0


# ------------------------------------------------------------------ orchestration

def run_council(inputs: dict, cfg: dict, allow_degraded: bool = False,
                skip_preflight: bool = False, cache_path: Optional[Path] = None) -> dict:
    panel, bridges = cfg["panel"], cfg["bridges"]
    judges = panel["judges"]
    floor = panel.get("confidence_floor", 0.5)
    rubrics = inputs["rubrics"]

    degraded_reason = None
    if not skip_preflight:
        problems = validate_panel(cfg)
        if problems:
            msg = "COUNCIL PREFLIGHT FAILED:\n  - " + "\n  - ".join(problems)
            if not allow_degraded:
                raise SystemExit(msg)
            degraded_reason = "; ".join(problems)
            print("WARN (--allow-degraded): " + msg, flush=True)

    cache = VerdictCache(cache_path) if cache_path else None
    results: list[dict] = []
    abstain = {j["id"]: 0 for j in judges}
    graded = {j["id"]: 0 for j in judges}

    if degraded_reason is not None:
        # every rubric unassessable; no judge calls
        for r in rubrics:
            results.append(_result_row(r, [], None, "council_unavailable"))
        return _finalize(inputs, results, abstain, graded, degraded_reason)

    # D1: short-circuit no-opportunity rubrics (no judge call, verdict=null)
    live = [r for r in rubrics if r.get("status") != "no_opportunity"]
    tasks = [(r["number"], j, r["prompt"]) for r in live for j in judges]

    per: dict[str, list[dict]] = {r["number"]: [] for r in rubrics}
    with ThreadPoolExecutor(max_workers=panel.get("max_workers", 6)) as pool:
        fut: dict[Future, tuple] = {
            pool.submit(call_one, num, j, prompt, panel, bridges, cache): (num, j)
            for (num, j, prompt) in tasks
        }
        for f in list(fut):
            rec = f.result()  # call_one never raises
            per[rec["number"]].append(rec)
            if rec["verdict"] is None:
                abstain[rec["model"]] += 1
            else:
                graded[rec["model"]] += 1

    by_num = {r["number"]: r for r in rubrics}
    for num, subs in per.items():
        r = by_num[num]
        if r.get("status") == "no_opportunity":
            results.append(_result_row(r, [], None, "no_opportunity"))
        else:
            results.append(_result_row(r, subs, aggregate_rubric(subs, floor), "assessed"))

    if cache is not None:
        cache.flush()
    return _finalize(inputs, results, abstain, graded, degraded_reason)


def _result_row(r: dict, subs: list[dict], verdict: Optional[int], status: str) -> dict:
    return {
        "number": r["number"],
        "type": r.get("type"),
        "is_positive": r.get("is_positive"),
        "score": r.get("score"),
        "status": status,
        "verdict": verdict,
        "judges": [{k: s.get(k) for k in ("model", "verdict", "confidence", "evidence", "error")}
                   for s in subs],
    }


def _finalize(inputs: dict, results: list[dict], abstain: dict, graded: dict,
              degraded_reason: Optional[str]) -> dict:
    abstain_rate = {m: round(abstain[m] / t, 3) if (t := abstain[m] + graded[m]) else None
                    for m in abstain}
    warnings = [f"judge {m} abstained on {r:.0%} of graded rubrics" for m, r in abstain_rate.items()
                if r is not None and r >= 0.5]
    n_assessed = sum(1 for x in results if x["status"] == "assessed" and x["verdict"] is not None)
    return {
        "task_id": inputs.get("task_id"),
        "traj_path": inputs.get("traj_path"),
        "council_status": "unavailable" if degraded_reason else "ok",
        "degraded_reason": degraded_reason,
        "abstain_rate": abstain_rate,
        "warnings": warnings,
        "n_met": sum(1 for x in results if x["verdict"] == 1),
        "n_unmet": sum(1 for x in results if x["verdict"] == 0),
        "n_unassessed": sum(1 for x in results if x["verdict"] is None),
        "n_assessed": n_assessed,
        "rubrics": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM council driver")
    ap.add_argument("--inputs", required=True, help="council_inputs.json from prepare_inputs.py")
    ap.add_argument("--out", required=True, help="council_results.json output path")
    ap.add_argument("--panel", default=str(DEFAULT_PANEL))
    ap.add_argument("--allow-degraded", action="store_true",
                    help="on preflight failure, emit council_unavailable instead of aborting")
    ap.add_argument("--skip-preflight", action="store_true", help="offline/testing only")
    ap.add_argument("--cache", default=None, help="optional verdict cache json path")
    args = ap.parse_args()

    cfg = load_panel(Path(args.panel))
    inputs = json.loads(Path(args.inputs).read_text())
    out = run_council(inputs, cfg, allow_degraded=args.allow_degraded,
                      skip_preflight=args.skip_preflight,
                      cache_path=Path(args.cache) if args.cache else None)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    for w in out["warnings"]:
        print("WARN:", w, flush=True)
    print(f"council_status={out['council_status']} met={out['n_met']} unmet={out['n_unmet']} "
          f"unassessed={out['n_unassessed']} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
