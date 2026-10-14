"""
LLM Judge Runner
================
Calls a Bedrock judge agent to evaluate AI advisor responses against
factual assertions and pass criteria from the test set.

Reads benchmark results (with ai_response) and the test set (with
factual_assertions, pass_criteria), calls the judge for each non-deferred
response, and saves scored results.

Usage:
  python eval/run_judge.py --input eval/benchmark_results_<timestamp>.csv
  python eval/run_judge.py --input eval/benchmark_results_<timestamp>.csv --all  # judge deferrals too
  python eval/run_judge.py --input eval/benchmark_results_<timestamp>.csv --dry-run

Environment variables:
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN
  JUDGE_AGENT_ID (default: MXHHFYQDQZ)
  JUDGE_AGENT_ALIAS_ID (default: UQEQ4FFINX)
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
TEST_SET_PATH = SCRIPT_DIR / "ai_advisor_test_set 3.csv"

# Import deferral detection to skip deferrals by default
sys.path.insert(0, str(SCRIPT_DIR))
from student_profiles import STUDENT_PROFILES, STUDENT_CONTEXT_MAP
from qa_benchmark_eval import check_deferral


def load_test_set() -> dict:
    """Load the test set CSV, keyed by test_id."""
    tests = {}
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = row.get("test_id", "").strip()
            if tid:
                tests[tid] = row
    return tests


def load_benchmark_results(path: str) -> list[dict]:
    """Load benchmark results CSV."""
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_judge_input(test: dict, ai_response: str) -> str:
    """Build the input text for the judge agent."""
    query = test.get("query", "")
    ground_truth = test.get("ground_truth_answer", "")
    assertions_raw = test.get("factual_assertions", "")
    pass_criteria = test.get("pass_criteria", "")
    student_key = test.get("student_context", "any_student")

    # Build student context summary
    email = STUDENT_CONTEXT_MAP.get(student_key, student_key)
    profile = STUDENT_PROFILES.get(email, {})
    context_summary = ""
    if profile:
        context_summary = (
            f"Student: {profile.get('name', '')} ({email})\n"
            f"Current courses: {', '.join(profile.get('current_courses', []))}\n"
            f"Completed: {', '.join(profile.get('completed', []))}\n"
            f"Credits: {profile.get('credits_completed', 0)}/128\n"
            f"Notes: {profile.get('notes', '')}"
        )

    # Build assertions list
    assertions = [a.strip() for a in assertions_raw.split(";") if a.strip()]
    assertions_block = "\n".join(f"{i+1}. {a}" for i, a in enumerate(assertions))
    if not assertions_block:
        assertions_block = "(no specific assertions for this question)"

    return f"""STUDENT QUESTION:
"{query}"

EXPECTED ANSWER:
"{ground_truth}"

AI ADVISOR RESPONSE:
"{ai_response}"

STUDENT CONTEXT:
{context_summary}

TASK 1 — Evaluate each factual assertion. Respond PASS, FAIL, or NOT_ADDRESSED.
- PASS: The AI response is consistent with this assertion (even if using different words)
- FAIL: The AI response directly contradicts this assertion
- NOT_ADDRESSED: The AI response does not cover this topic

Assertions:
{assertions_block}

TASK 2 — Evaluate the overall response against this pass criteria:
"{pass_criteria}"

Respond PASS or FAIL with a one-sentence reason.

OUTPUT FORMAT (JSON only):
{{
  "assertions": [
    {{"assertion": "...", "result": "PASS|FAIL|NOT_ADDRESSED", "reason": "one sentence"}},
    ...
  ],
  "pass_criteria_result": "PASS|FAIL",
  "pass_criteria_reason": "one sentence",
  "overall": "PASS|FAIL"
}}

RULES:
- "overall" is PASS only if ALL assertions are PASS and pass_criteria is PASS
- NOT_ADDRESSED counts as a failure for "overall"
- Judge substance not wording — "Calc II" and "MATH 1232" mean the same thing
- Extra true information beyond the assertion is still PASS
- Hedging without stating a wrong fact is NOT_ADDRESSED, not FAIL"""


def invoke_judge(client, agent_id: str, alias_id: str, input_text: str) -> tuple[str, float]:
    """Call the judge agent and return (response_text, elapsed_ms)."""
    session_id = str(uuid.uuid4()).replace("-", "")[:20]
    start = time.time()

    response = client.invoke_agent(
        agentId=agent_id,
        agentAliasId=alias_id,
        sessionId=session_id,
        inputText=input_text,
    )

    response_text = ""
    for event in response.get("completion", []):
        chunk = event.get("chunk", {})
        if "bytes" in chunk:
            response_text += chunk["bytes"].decode("utf-8")

    elapsed_ms = round((time.time() - start) * 1000)
    return response_text, elapsed_ms


def parse_judge_response(raw: str) -> dict:
    """Extract JSON from judge response."""
    json_match = re.search(r'\{[\s\S]*\}', raw)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return {}


def main():
    parser = argparse.ArgumentParser(description="Run LLM judge on benchmark results")
    parser.add_argument("--input", required=True, help="Benchmark results CSV")
    parser.add_argument("--dry-run", action="store_true", help="Print judge inputs without calling")
    parser.add_argument("--all", action="store_true", help="Judge all responses including deferrals")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between calls (default: 2)")
    args = parser.parse_args()

    # Load data
    test_set = load_test_set()
    results = load_benchmark_results(args.input)
    print(f"Loaded {len(results)} results, {len(test_set)} test cases")

    # Agent config
    agent_id = os.environ.get("JUDGE_AGENT_ID", "MXHHFYQDQZ")
    alias_id = os.environ.get("JUDGE_AGENT_ALIAS_ID", "UQEQ4FFINX")

    if not args.dry_run:
        import boto3
        client = boto3.client(
            "bedrock-agent-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
    else:
        client = None

    # Build output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = SCRIPT_DIR / f"judge_results_{timestamp}.csv"

    output_fields = [
        "test_id", "category", "query", "student_context",
        "is_deferral", "judge_overall", "pass_criteria_result", "pass_criteria_reason",
        "assertions_detail", "judge_raw", "response_time_ms",
    ]

    judged = []
    total = len(results)

    print(f"\n{'='*60}")
    if args.dry_run:
        print("  DRY RUN — printing judge inputs only")
    else:
        print(f"  Judging {total} responses")
        print(f"  Judge agent: {agent_id} / {alias_id}")
    print(f"  Output: {output_path.name}")
    print(f"{'='*60}\n")

    for i, row in enumerate(results, 1):
        test_id = row.get("test_id", "")
        ai_response = row.get("ai_response", "")
        category = row.get("category", "")

        # Check if deferred
        is_deferred, _ = check_deferral(ai_response, category)

        if is_deferred and not args.all:
            print(f"[{i}/{total}] {test_id} — skipped (deferral)")
            judged.append({
                "test_id": test_id,
                "category": category,
                "query": row.get("query", "")[:80],
                "student_context": row.get("student_context", ""),
                "is_deferral": True,
                "judge_overall": "SKIPPED",
                "pass_criteria_result": "",
                "pass_criteria_reason": "Deferred — not judged",
                "assertions_detail": "",
                "judge_raw": "",
                "response_time_ms": 0,
            })
            continue

        # Get test case for assertions/criteria
        test = test_set.get(test_id, {})
        if not test:
            print(f"[{i}/{total}] {test_id} — skipped (not in test set)")
            continue

        judge_input = build_judge_input(test, ai_response)

        if args.dry_run:
            print(f"[{i}/{total}] {test_id}")
            print(f"  Input ({len(judge_input)} chars):")
            for line in judge_input.split("\n")[:8]:
                print(f"    {line}")
            print(f"    ...")
            print()
            judged.append({
                "test_id": test_id, "category": category,
                "query": row.get("query", "")[:80],
                "student_context": row.get("student_context", ""),
                "is_deferral": is_deferred,
                "judge_overall": "[DRY RUN]",
                "pass_criteria_result": "", "pass_criteria_reason": "",
                "assertions_detail": "", "judge_raw": "", "response_time_ms": 0,
            })
            continue

        # Call judge
        print(f"[{i}/{total}] {test_id}...", end="", flush=True)
        try:
            raw_response, elapsed = invoke_judge(client, agent_id, alias_id, judge_input)
            parsed = parse_judge_response(raw_response)

            overall = parsed.get("overall", "PARSE_ERROR")
            pc_result = parsed.get("pass_criteria_result", "")
            pc_reason = parsed.get("pass_criteria_reason", "")
            assertions = parsed.get("assertions", [])
            assertions_str = "; ".join(
                f"{a.get('assertion', '')[:40]}: {a.get('result', '?')}"
                for a in assertions
            )

            print(f" {elapsed}ms — {overall}")
        except Exception as e:
            raw_response = f"[ERROR] {e}"
            overall = "ERROR"
            pc_result = ""
            pc_reason = str(e)
            assertions_str = ""
            elapsed = 0
            print(f" ERROR: {e}")

        judged.append({
            "test_id": test_id,
            "category": category,
            "query": row.get("query", "")[:80],
            "student_context": row.get("student_context", ""),
            "is_deferral": is_deferred,
            "judge_overall": overall,
            "pass_criteria_result": pc_result,
            "pass_criteria_reason": pc_reason,
            "assertions_detail": assertions_str,
            "judge_raw": raw_response,
            "response_time_ms": elapsed,
        })

        if i < total and args.delay > 0:
            time.sleep(args.delay)

    # Write results
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(judged)

    print(f"\nResults written to: {output_path}")

    # Summary
    answered = [j for j in judged if j["judge_overall"] not in ("SKIPPED", "[DRY RUN]", "ERROR")]
    if answered:
        passed = sum(1 for j in answered if j["judge_overall"] == "PASS")
        failed = sum(1 for j in answered if j["judge_overall"] == "FAIL")
        skipped = sum(1 for j in judged if j["judge_overall"] == "SKIPPED")
        print(f"\n  Judged: {len(answered)}, Skipped (deferrals): {skipped}")
        print(f"  PASS: {passed}/{len(answered)} ({100*passed/len(answered):.0f}%)")
        print(f"  FAIL: {failed}/{len(answered)} ({100*failed/len(answered):.0f}%)")


if __name__ == "__main__":
    main()
