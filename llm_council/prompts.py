"""
LLM council prompt template — single generic builder driven by rubric schema.

The rubric schema (see rubric_eval/rubrics.jsonl) is:
  {number, criterion, is_positive, type, importance, score}

build_criterion_prompt(rubric, evidence_excerpt) returns a str prompt with the
1|0|null verdict contract.

Council spec (locked, do NOT change here):
- 3 judges cross-family: claude-sonnet-4-6, gpt-4o, gemini-2.5-flash
- BAN: claude-opus-* (generated the trajectory being judged)
- Verdict: {"verdict": 1|0|null, "confidence": 0.0-1.0, "evidence": "..."}
- confidence < 0.5 → treated as null by the aggregator.
- Aggregation: majority 1/0; tie → 0 (conservative).
"""

from typing import Any


VERDICT_CONTRACT = """
Return your judgment as a single JSON object on one line, nothing else:
{"verdict": 1 | 0 | null, "confidence": <float 0.0-1.0>, "evidence": "<one-sentence citation of the specific data points that drove your judgment>"}

- verdict = 1 → the criterion IS satisfied by the evidence.
- verdict = 0 → the criterion is NOT satisfied by the evidence.
- verdict = null → the evidence is genuinely insufficient (reserve for missing data, not for personal uncertainty).
- confidence < 0.5 will be treated as null by the aggregator; only vote 1 or 0 when you are confident.
- Do NOT punish coherent alternative strategies. The reference behavior is one valid policy of many.
- Do NOT infer intent beyond what the evidence directly shows.
"""


HEADER = """You are evaluating a single rubric criterion against evidence from a completed retail-agent trajectory. You are one of three cross-family judges; votes are aggregated by majority.

The task governs. Grade only whether the specific behavior described in the criterion is present in the trajectory. Do not require adherence to any reference policy — any coherent policy that satisfies the criterion should receive a positive verdict.
"""


POSITIVE_POLARITY_NOTE = """This is a POSITIVE rubric describing a desired behavior.
- If the described behavior IS present in the evidence → verdict = 1.
- If the described behavior is NOT present → verdict = 0."""


NEGATIVE_POLARITY_NOTE = """This is a NEGATIVE rubric describing an unwanted behavior.
- If the described (unwanted) behavior IS present in the evidence → verdict = 1 (the bad thing happened).
- If the unwanted behavior is NOT present → verdict = 0 (the agent avoided it)."""


def build_criterion_prompt(rubric: dict[str, Any], evidence_excerpt: str) -> str:
    """Build a judge prompt for a single rubric criterion + its evidence excerpt.

    Args:
        rubric: dict with keys {number, criterion, is_positive, type, importance, score}.
        evidence_excerpt: pre-extracted trajectory evidence relevant to this criterion.

    Returns:
        Complete prompt string ready to send to a single judge.
    """
    polarity_note = (
        POSITIVE_POLARITY_NOTE if rubric.get("is_positive", True) else NEGATIVE_POLARITY_NOTE
    )
    parts = [
        HEADER,
        f"Rubric {rubric.get('number', '?')} (type={rubric.get('type', '?')}, "
        f"importance={rubric.get('importance', '?')}):",
        rubric.get("criterion", ""),
        "",
        polarity_note,
        "",
        "Evidence excerpt from the trajectory:",
        "---",
        evidence_excerpt or "(no evidence extracted for this rubric type)",
        "---",
        VERDICT_CONTRACT,
    ]
    return "\n".join(parts)
