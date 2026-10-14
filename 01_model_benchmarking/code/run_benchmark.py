"""
Bedrock Agent Benchmark Runner
==============================
Loads the test set CSV, calls the Bedrock Agent for each question with the
appropriate student context, and saves raw results to a timestamped CSV.

Each run produces a separate output file so you can compare performance
across multiple runs.

Usage:
  python eval/run_benchmark.py --dry-run          # Check inputs, no API calls
  python eval/run_benchmark.py                    # Full run (default 2s delay)
  python eval/run_benchmark.py --delay 3          # Custom delay between calls
  python eval/run_benchmark.py --output results   # Custom output prefix

Requirements:
  pip install boto3
  Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
    AWS_SESSION_TOKEN (optional), BEDROCK_AGENT_ID, BEDROCK_AGENT_ALIAS_ID
"""

import argparse
import csv
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
TEST_SET_PATH = SCRIPT_DIR / "ai_advisor_test_set 3.csv"


def load_test_set(path: Path) -> list[dict]:
    """Load the test set CSV into a list of dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("test_id", "").strip():
                rows.append(row)
    print(f"Loaded {len(rows)} test cases from {path.name}")
    return rows


def build_input_text(student_context: str, query: str) -> str:
    """Construct the input text in the same format as the app (route.ts:71-73)."""
    if student_context:
        return f"[Student Academic Profile]\n{student_context}\n\n[Student Question]\n{query}"
    return query


def invoke_bedrock_agent(
    client,
    agent_id: str,
    agent_alias_id: str,
    input_text: str,
    session_id: str,
) -> tuple[str, float]:
    """
    Call the Bedrock Agent and collect the streamed response.

    Returns:
        (response_text, elapsed_ms)
    """
    start = time.time()
    response = client.invoke_agent(
        agentId=agent_id,
        agentAliasId=agent_alias_id,
        sessionId=session_id,
        inputText=input_text,
    )

    # Collect streamed chunks (same pattern as route.ts:86-92)
    response_text = ""
    event_stream = response.get("completion", [])
    for event in event_stream:
        chunk = event.get("chunk", {})
        if "bytes" in chunk:
            response_text += chunk["bytes"].decode("utf-8")

    elapsed_ms = round((time.time() - start) * 1000)
    return response_text, elapsed_ms


def run_benchmark(args):
    """Main benchmark loop."""
    from student_profiles import STUDENT_CONTEXT_MAP, build_student_context

    # Load test set
    test_cases = load_test_set(TEST_SET_PATH)
    if not test_cases:
        print("ERROR: No test cases found.")
        sys.exit(1)

    # Validate env vars (unless dry run)
    agent_id = os.environ.get("BEDROCK_AGENT_ID", "")
    agent_alias_id = os.environ.get("BEDROCK_AGENT_ALIAS_ID", "")

    if not args.dry_run:
        missing = []
        if not agent_id:
            missing.append("BEDROCK_AGENT_ID")
        if not agent_alias_id:
            missing.append("BEDROCK_AGENT_ALIAS_ID")
        if not os.environ.get("AWS_ACCESS_KEY_ID"):
            missing.append("AWS_ACCESS_KEY_ID")
        if not os.environ.get("AWS_SECRET_ACCESS_KEY"):
            missing.append("AWS_SECRET_ACCESS_KEY")
        if missing:
            print(f"ERROR: Missing environment variables: {', '.join(missing)}")
            sys.exit(1)

        import boto3
        client = boto3.client(
            "bedrock-agent-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
    else:
        client = None

    # Build output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_path = SCRIPT_DIR / f"{args.output}_{timestamp}.csv"
    else:
        output_path = SCRIPT_DIR / f"benchmark_results_{timestamp}.csv"

    # Determine output columns: all original CSV columns + runner columns
    original_columns = list(test_cases[0].keys())
    extra_columns = ["ai_response", "response_time_ms", "timestamp", "session_id"]
    all_columns = original_columns + extra_columns

    results = []
    total = len(test_cases)

    print(f"\n{'='*60}")
    if args.dry_run:
        print("  DRY RUN — printing inputs only, no Bedrock calls")
    else:
        print(f"  Running {total} test cases against Bedrock Agent")
        print(f"  Agent: {agent_id} / Alias: {agent_alias_id}")
        print(f"  Delay: {args.delay}s between calls")
    print(f"  Output: {output_path.name}")
    print(f"{'='*60}\n")

    for i, tc in enumerate(test_cases, 1):
        test_id = tc["test_id"]
        student_ctx_key = tc.get("student_context", "any_student")
        query = tc["query"]

        # Build student context string
        context_str = build_student_context(student_ctx_key)
        input_text = build_input_text(context_str, query)

        # New session per question to avoid cross-question context leaking
        session_id = str(uuid.uuid4()).replace("-", "")[:20]

        if args.dry_run:
            print(f"[{i}/{total}] {test_id} ({student_ctx_key})")
            print(f"  Query: {query}")
            print(f"  Input text ({len(input_text)} chars):")
            # Print first few lines of the context for verification
            lines = input_text.split("\n")
            for line in lines[:12]:
                print(f"    {line}")
            if len(lines) > 12:
                print(f"    ... ({len(lines) - 12} more lines)")
            print()

            row = dict(tc)
            row["ai_response"] = "[DRY RUN]"
            row["response_time_ms"] = 0
            row["timestamp"] = datetime.now().isoformat()
            row["session_id"] = session_id
            results.append(row)
            continue

        # Call Bedrock Agent
        print(f"[{i}/{total}] {test_id} ({student_ctx_key}): {query[:60]}...", end="", flush=True)
        try:
            response_text, elapsed_ms = invoke_bedrock_agent(
                client, agent_id, agent_alias_id, input_text, session_id
            )
            print(f" {elapsed_ms}ms, {len(response_text)} chars")
        except Exception as e:
            response_text = f"[ERROR] {e}"
            elapsed_ms = 0
            print(f" ERROR: {e}")

        row = dict(tc)
        row["ai_response"] = response_text
        row["response_time_ms"] = elapsed_ms
        row["timestamp"] = datetime.now().isoformat()
        row["session_id"] = session_id
        results.append(row)

        # Rate-limit delay (skip after last question)
        if i < total and args.delay > 0:
            time.sleep(args.delay)

    # Write results CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to: {output_path}")
    print(f"Total test cases: {total}")

    if not args.dry_run:
        # Print quick summary
        errors = sum(1 for r in results if r["ai_response"].startswith("[ERROR]"))
        avg_time = sum(r["response_time_ms"] for r in results) / total if total else 0
        print(f"Errors: {errors}")
        print(f"Avg response time: {avg_time:.0f}ms")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark test set against Bedrock Agent"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print constructed inputs without calling Bedrock",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between API calls (default: 2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output file prefix (default: benchmark_results)",
    )
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
