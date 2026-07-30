#!/usr/bin/env python3
"""evaluation.aggregate — blend the pytest signals and the LLM-council rubric into the
final money-primary reward for one RetailBench run, per tests/thresholds.json.

Reads (paths from the task's test.sh):
  --test-results     test_results.json     (pytest via evaluation.conftest)
  --council-results  council_results.json  (the council; optional — pytest-only if absent/failed)
  --weights          thresholds.json       (config + scoring formula)
Writes:
  --reward-json      reward.json           (full breakdown)
  --reward-txt       reward.txt            (the scalar reward, or empty on EXCLUDE)

Signals:
  O  money outcome  = the continuous outcome recorded by the money_net_worth_outcome check.
  P  process        = weighted pass-fraction of the kind="process" checks.
  R  rubric         = clip(max(0, (sum met-positive score - sum triggered-negative |score|)
                            / sum all-positive score), 0, 1) over the assessed council rubrics;
                      None when the council is unavailable (then Q = P, pytest-only).
  integrity gate    = EXCLUDE if any category="integrity" check failed (or the money outcome
                      is unreadable): reported separately, no per-run score, counted as the
                      loss-band floor 0.0 in any mean.

Bands (Q = 0.7*P + 0.3*R, or Q = P when R is None):
  EXCLUDE                                  -> no score
  made money (horizon_pinned_nw > funds)   -> 0.55 + 0.40*O + 0.05*Q       [0.55 - 1.00]
  lost money                               -> 0.40*clip(horizon_pinned_nw/funds,0,1) + 0.05*Q  [0.00 - 0.45]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _process_fraction(tests: list) -> float:
    scored = [t for t in tests if t.get("kind") == "process"]
    den = sum(float(t.get("weight", 0.0)) for t in scored)
    num = sum(float(t.get("weight", 0.0)) for t in scored if t.get("verdict") == 1)
    return (num / den) if den else 0.0


def _integrity_excluded(tests: list) -> tuple[bool, str | None]:
    for t in tests:
        if t.get("category") == "integrity" and t.get("verdict") == 0:
            return True, t.get("name")
    return False, None


def _money_props(tests: list) -> dict:
    m = next((t for t in tests if t.get("category") == "money"), None)
    return (m or {}).get("properties", {}) or {}


def _rubric_R(council: dict | None) -> tuple[float | None, str]:
    if not council or council.get("council_status") != "ok":
        return None, "council_unavailable"
    positive_total = 0.0
    got = 0.0
    for r in council.get("rubrics", []):
        if r.get("status") != "assessed":
            continue
        verdict = r.get("verdict")
        if verdict not in (0, 1):
            continue
        mag = abs(int(r.get("score") or 0))
        if r.get("is_positive"):
            positive_total += mag
            if verdict == 1:
                got += mag
        else:  # negative rubric: present (verdict 1) subtracts
            if verdict == 1:
                got -= mag
    if positive_total <= 0:
        return None, "no_positive_rubrics"
    return _clip01(got / positive_total), "ok"


def aggregate(test_results: dict, council_results: dict | None, thresholds: dict) -> dict:
    tests = test_results.get("tests", [])
    funds = float(thresholds["config"]["initial_funds"])

    props = _money_props(tests)
    O = props.get("money_outcome_O")
    horizon_nw = props.get("horizon_pinned_net_worth")
    pinned_nw = props.get("pinned_net_worth")

    excluded, exclude_reason = _integrity_excluded(tests)
    P = _process_fraction(tests)
    R, rubric_status = _rubric_R(council_results)

    out: dict = {
        "task_id": test_results.get("task_id"),
        "integrity": "exclude" if excluded else "ok",
        "O": round(O, 6) if isinstance(O, (int, float)) else None,
        "P": round(P, 6),
        "R": round(R, 6) if R is not None else None,
        "rubric_status": rubric_status,
        "pinned_net_worth": pinned_nw,
        "horizon_pinned_net_worth": horizon_nw,
        "funds": funds,
    }

    money_unreadable = not isinstance(O, (int, float)) or not isinstance(horizon_nw, (int, float))
    if excluded or money_unreadable:
        out.update(
            integrity="exclude",
            exclude_reason=exclude_reason or "money outcome unreadable",
            band="EXCLUDED",
            Q=None,
            score=None,
            mean_contribution=0.0,  # an EXCLUDE never raises a model's mean
        )
        return out

    Q = (0.7 * P + 0.3 * R) if R is not None else P
    out["Q"] = round(Q, 6)
    if horizon_nw > funds:
        out["band"] = "MADE-MONEY"
        out["score"] = round(0.55 + 0.40 * O + 0.05 * Q, 6)
    else:
        out["band"] = "LOST-MONEY"
        out["score"] = round(0.40 * _clip01(horizon_nw / funds) + 0.05 * Q, 6)
    out["mean_contribution"] = out["score"]
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Aggregate RetailBench verifier signals into the reward.")
    ap.add_argument("--test-results", required=True)
    ap.add_argument("--council-results", default=None)
    ap.add_argument("--weights", required=True, help="tests/thresholds.json")
    ap.add_argument("--reward-json", required=True)
    ap.add_argument("--reward-txt", required=True)
    args = ap.parse_args(argv)

    test_results = _load(args.test_results)
    council = None
    if args.council_results and Path(args.council_results).is_file():
        try:
            council = _load(args.council_results)
        except (json.JSONDecodeError, OSError):
            council = None
    thresholds = _load(args.weights)

    result = aggregate(test_results, council, thresholds)

    Path(args.reward_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.reward_json).write_text(json.dumps(result, indent=2))
    Path(args.reward_txt).write_text("" if result.get("score") is None else f"{result['score']}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
