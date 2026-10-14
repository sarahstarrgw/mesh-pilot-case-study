"""
AI Advisor QA Benchmark Evaluation Script
==========================================
Scores AI chatbot responses on three measures:
  1. Factual Correctness (Pass/Fail)
  2. Hallucination Count (integer, 0+)
  3. Completeness (Full/Partial/Deflected)

Plus structured criteria from the new test set:
  4. must_contain pass/fail
  5. must_not_contain pass/fail
  6. overall_pass (composite)

Three-pass approach:
  Pass 1: Automated NLP metrics (ROUGE-L, METEOR) — textual overlap with reference
  Pass 2: Rule-based scoring — structured fact-checking against known ground truth
  Pass 3: LLM-as-judge — nuanced evaluation with confidence via logprobs

Supports two input formats:
  - Old format: qa_benchmark_questions.csv (columns: ID, Category, ...)
  - New format: benchmark_results_*.csv from run_benchmark.py (columns: test_id, category, ...)

Usage:
  python qa_benchmark_eval.py                                    # Old format, Passes 1+2
  python qa_benchmark_eval.py --input eval/benchmark_results_*.csv  # New format
  python qa_benchmark_eval.py --input eval/benchmark_results_*.csv --llm  # + LLM judge
  python qa_benchmark_eval.py --llm --provider openai            # Specify LLM provider
  python qa_benchmark_eval.py --output results.csv               # Custom output path

Requirements:
  pip install nltk rouge-score numpy
  For LLM judge: pip install openai (or anthropic, or boto3)
"""

import json
import csv
import re
import os
import sys
import math
import glob
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Import shared data from student_profiles module
try:
    from student_profiles import (
        KNOWN_COURSES,
        PROGRAM_FACTS,
        STUDENT_PROFILES,
        STUDENT_CONTEXT_MAP,
    )
except ImportError:
    # Fallback: add script directory to path and retry
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from student_profiles import (
        KNOWN_COURSES,
        PROGRAM_FACTS,
        STUDENT_PROFILES,
        STUDENT_CONTEXT_MAP,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_QUESTIONS_PATH = os.path.join(SCRIPT_DIR, "qa_benchmark_questions.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "qa_benchmark_results.csv")


# ---------------------------------------------------------------------------
# Student profile resolution (supports both old and new key formats)
# ---------------------------------------------------------------------------

_KEY_TO_EMAIL = {
    "studenta": "studenta@gwsandbox.com",
    "studentb": "studentb@gwsandbox.com",
    "studentc": "studentc@gwsandbox.com",
    "student_a": "studenta@gwsandbox.com",
    "student_b": "studentb@gwsandbox.com",
    "student_c": "studentc@gwsandbox.com",
    "any_student": "studenta@gwsandbox.com",
    "Any": "studenta@gwsandbox.com",
}


def _resolve_profile(student_key: str) -> dict | None:
    """Resolve a student key to the profile dict (with current_courses, completed, etc.)."""
    email = _KEY_TO_EMAIL.get(student_key, student_key)
    return STUDENT_PROFILES.get(email)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkQuestion:
    id: str                   # int for old format, string like "QA-01" for new
    category: str
    subcategory: str
    question: str
    student_context: str
    reference_answer: str
    source: str
    actual_ai_response: str
    # New format fields (empty for old format)
    must_contain: str = ""
    must_not_contain: str = ""
    factual_assertions: str = ""
    pass_criteria: str = ""


@dataclass
class EvalResult:
    question_id: str
    category: str
    subcategory: str
    question: str
    student_context: str
    reference_answer: str
    actual_ai_response: str
    # Pass 1: NLP metrics
    rouge_l: float = 0.0
    meteor: float = 0.0
    # Pass 2+3: Our three measures
    factual_correctness: str = ""       # "Pass" or "Fail"
    hallucination_count: int = 0
    completeness: str = ""              # "Full", "Partial", "Deflected"
    # Deferral detection
    is_deferral: bool = False           # True if agent didn't attempt to answer
    deferral_reason: str = ""           # Why it was classified as a deferral
    # Structured criteria (new format)
    must_contain_pass: bool = True
    must_contain_missing: list = field(default_factory=list)
    must_not_contain_pass: bool = True
    must_not_contain_found: list = field(default_factory=list)
    overall_pass: bool = False
    # Metadata
    scoring_method: str = ""            # "rule-based", "llm-judge", "needs-review"
    factual_errors: list = field(default_factory=list)
    hallucinated_claims: list = field(default_factory=list)
    llm_confidence: float = 0.0         # from logprobs (0-1, higher = more confident)
    notes: str = ""


# ---------------------------------------------------------------------------
# Pass 1: Automated NLP metrics (from Kinga's approach)
# ---------------------------------------------------------------------------

def try_import_nlp():
    """Import NLP libraries, return None if not installed."""
    try:
        from rouge_score import rouge_scorer
        from nltk.tokenize import RegexpTokenizer
        from nltk.translate.meteor_score import meteor_score
        import nltk
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
        return rouge_scorer, RegexpTokenizer, meteor_score
    except ImportError:
        return None


def compute_nlp_metrics(response: str, reference: str) -> dict:
    """Compute ROUGE-L and METEOR scores between response and reference."""
    nlp = try_import_nlp()
    if nlp is None:
        return {"rouge_l": -1.0, "meteor": -1.0}

    rouge_scorer_cls, RegexpTokenizer, meteor_score_fn = nlp

    # ROUGE-L: longest common subsequence overlap
    scorer = rouge_scorer_cls.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, response)
    rouge_l = rouge_scores['rougeL'].fmeasure

    # METEOR: accounts for synonyms and paraphrasing
    tokenizer = RegexpTokenizer(r'\w+')
    ref_tokens = [tokenizer.tokenize(reference)]
    cand_tokens = tokenizer.tokenize(response)
    meteor = meteor_score_fn(ref_tokens, cand_tokens)

    return {"rouge_l": round(rouge_l, 4), "meteor": round(meteor, 4)}


# ---------------------------------------------------------------------------
# Structured criteria checks (new test set format)
# ---------------------------------------------------------------------------

def check_must_contain(response: str, must_contain_str: str) -> tuple[bool, list]:
    """
    Check that all required strings appear in the response (case-insensitive).

    Semicolons separate required terms (AND). Pipes within a term separate
    alternatives (OR). Example: "DegreeMap|Degree Map|degree audit;curriculum"
    means (DegreeMap OR Degree Map OR degree audit) AND curriculum.

    Returns:
        (pass, list_of_missing_strings)
    """
    if not must_contain_str or not must_contain_str.strip():
        return True, []

    required = [s.strip() for s in must_contain_str.split(";") if s.strip()]
    resp_lower = response.lower()
    missing = []
    for term in required:
        alternatives = [a.strip() for a in term.split("|")]
        if not any(a.lower() in resp_lower for a in alternatives):
            missing.append(term)
    return len(missing) == 0, missing


def check_must_not_contain(response: str, must_not_contain_str: str) -> tuple[bool, list]:
    """
    Check that none of the forbidden strings appear in the response (case-insensitive).

    Returns:
        (pass, list_of_found_forbidden_strings)
    """
    if not must_not_contain_str or not must_not_contain_str.strip():
        return True, []

    forbidden = [s.strip() for s in must_not_contain_str.split(";") if s.strip()]
    resp_lower = response.lower()
    found = [s for s in forbidden if s.lower() in resp_lower]
    return len(found) == 0, found


# ---------------------------------------------------------------------------
# Deferral detection
# ---------------------------------------------------------------------------

DEFERRAL_PATTERNS = [
    # "I don't have that information"
    (r"i (don't|do not) have (the |that |this )?(specific )?(prerequisite |course |)information",
     "says it lacks information"),
    (r"i (don't|do not) have .{0,30} in my (current )?(data|database)",
     "says it lacks information"),
    # Asks the student to look it up themselves
    (r"(check|consult|refer to) (the |your )?(course (catalog|bulletin)|student resources|academic advisor)",
     "redirects student to look it up"),
    (r"would you like me to (direct|guide) you (to |on )?where",
     "redirects student to look it up"),
    # Asks a clarifying question without providing any answer
    (r"^(are you |do you have |could you |can you |before i |to provide |to answer )",
     "asks clarifying question instead of answering"),
    # Single short question as the entire response
    (r"^[^.]*\?$",
     "response is just a question"),
]


def check_deferral(response: str, category: str = "") -> tuple[bool, str]:
    """
    Detect whether the agent deferred instead of answering.

    A deferral is when the agent does NOT attempt to answer the question —
    it asks for clarification, says it doesn't have the info, or redirects
    the student elsewhere. This is different from a partial answer.

    For out_of_scope questions, deferral is expected/correct behavior.

    Returns:
        (is_deferral, reason)
    """
    resp_lower = response.lower().strip()

    # Out-of-scope questions: deferral is the correct behavior
    if category == "out_of_scope":
        return False, ""

    # Check for substantive content — if present, it's not a deferral
    has_course_codes = bool(re.search(r'\b[A-Z]{2,4}\s+\d{4}\b', response))
    has_numbered_list = bool(re.search(r'\d+\.\s+\w', response))
    has_credit_info = bool(re.search(r'\d+\s*cr(edit)?', resp_lower))
    has_specific_answer = bool(re.search(r'\b(yes|no|you (can|should|need|have|are))\b', resp_lower))

    # If the response has substantive content (course codes, lists, credit info),
    # it's not a deferral even if it also asks questions
    substantive_signals = sum([has_course_codes, has_numbered_list, has_credit_info])
    if substantive_signals >= 2:
        return False, ""

    # "you can find/check/look" is a redirect, not a real answer
    is_redirect_answer = bool(re.search(
        r'you (can|should|could) (find|check|look|access|visit|review)', resp_lower
    ))

    # Check deferral patterns
    for pattern, reason in DEFERRAL_PATTERNS:
        if re.search(pattern, resp_lower):
            # If it matches a deferral pattern but also has substantive content,
            # only count it if the substantive content is minimal.
            # Don't let redirect phrases ("you can find") override a deferral.
            if substantive_signals >= 1 and has_specific_answer and not is_redirect_answer:
                continue
            return True, reason

    # Short response that's mostly questions
    if len(response) < 200 and response.count("?") >= 1 and not has_specific_answer:
        return True, "short response with no substantive content"

    return False, ""


# ---------------------------------------------------------------------------
# Pass 2: Rule-based scoring
# ---------------------------------------------------------------------------

DEFLECTION_PATTERNS = [
    r"could you (please )?(clarify|confirm|tell me|specify|provide)",
    r"can you (please )?(clarify|confirm|tell me|specify|provide)",
    r"are you (aware of|asking about)",
    r"what (is your|specific|do you mean)",
    r"to (better assist|provide.*accurate|answer.*accurately)",
    r"i (need|would need) (to know|more information|to check)",
    r"i('m| am) not sure what you mean",
]

ADVISOR_REDIRECT_PATTERNS = [
    r"(consult|contact|reach out to|speak with|discuss with).*(advisor|james young)",
    r"recommend.*(consulting|contacting|speaking with|reaching out).*(advisor|james young)",
    r"(advisor|james young).*can (help|provide|assist|guide)",
]


def detect_deflection(response: str) -> str:
    """Classify completeness based on response patterns."""
    resp_lower = response.lower()

    deflection_signals = sum(
        1 for p in DEFLECTION_PATTERNS if re.search(p, resp_lower)
    )

    has_course_mentions = any(code in response for code in KNOWN_COURSES)
    has_numbered_list = bool(re.search(r'\d+\.\s+\w', response))
    has_specific_credits = bool(re.search(r'\d+\s*cr(edit)?', resp_lower))
    substantive_signals = sum([has_course_mentions, has_numbered_list, has_specific_credits])

    question_count = response.count("?")
    if question_count >= 2 and substantive_signals == 0:
        return "Deflected"

    advisor_redirects = sum(
        1 for p in ADVISOR_REDIRECT_PATTERNS if re.search(p, resp_lower)
    )
    if advisor_redirects > 0 and substantive_signals == 0 and deflection_signals > 0:
        return "Deflected"

    if substantive_signals > 0 and (deflection_signals > 0 or advisor_redirects > 0):
        return "Partial"
    if substantive_signals >= 2:
        return "Full"
    if substantive_signals == 1:
        return "Partial"
    if len(response) < 150 and deflection_signals > 0:
        return "Deflected"

    return "Partial"


def check_course_accuracy(response: str) -> list:
    """Check if any course codes mentioned have wrong names."""
    errors = []
    for code, correct_name in KNOWN_COURSES.items():
        if code in response:
            pattern = rf"{re.escape(code)}\s+(.+?)(?:\s*\(|\s*-|\s*\n|\s*$)"
            match = re.search(pattern, response)
            if match:
                mentioned_name = match.group(1).strip().rstrip(",.")
                if (
                    len(mentioned_name) > 5
                    and correct_name.lower() not in mentioned_name.lower()
                    and mentioned_name.lower() not in correct_name.lower()
                ):
                    errors.append(
                        f"Course {code}: AI said '{mentioned_name}', should be '{correct_name}'"
                    )
    return errors


def check_credit_accuracy(response: str) -> list:
    """Check if credit counts mentioned are plausible."""
    errors = []
    # Match "128 total credits" or "128 credits required/needed to graduate"
    # but NOT sub-requirement counts like "98 credits for core Degree Requirements"
    total_match = re.search(
        r"(\d+)\s+total\s+credits?\s*(required|needed|to graduate|for the|for your)",
        response.lower()
    )
    if not total_match:
        total_match = re.search(
            r"(need|require)\w*\s+(\d+)\s+credits?\s*(to graduate|total)",
            response.lower()
        )
        if total_match:
            total_match = type('M', (), {'group': lambda self, n: total_match.group(2)})()
    if total_match:
        stated = int(total_match.group(1))
        if stated != PROGRAM_FACTS["total_credits_required"]:
            errors.append(
                f"Total credits: AI said {stated}, should be {PROGRAM_FACTS['total_credits_required']}"
            )
    return errors


def check_student_specific_errors(response: str, student_key: str) -> list:
    """Check for errors specific to a student's profile."""
    errors = []
    profile = _resolve_profile(student_key)
    if not profile:
        return errors

    for course in profile.get("current_courses", []):
        if course in response:
            rec_pattern = rf"(recommend|suggest|take|enroll|register).*{re.escape(course)}"
            if re.search(rec_pattern, response.lower()):
                taking_pattern = rf"(currently|already|taking|enrolled|in.progress).*{re.escape(course)}"
                if not re.search(taking_pattern, response.lower()):
                    errors.append(f"Recommended {course} but student is already taking it")

    for course in profile.get("completed", []):
        if course in response:
            rec_pattern = rf"(recommend|suggest|take|enroll).*{re.escape(course)}"
            not_taken_pattern = rf"(haven't|not yet|haven't taken).*{re.escape(course)}"
            if re.search(rec_pattern, response.lower()) or re.search(not_taken_pattern, response.lower()):
                errors.append(f"Recommended {course} or said not taken, but student already completed it")

    return errors


def detect_hallucinations(response: str) -> list:
    """Detect potentially hallucinated claims via rule-based checks."""
    hallucinations = []

    mentioned_codes = re.findall(r'\b([A-Z]{2,4})\s+(\d{4})\b', response)
    for dept, num in mentioned_codes:
        code = f"{dept} {num}"
        if code not in KNOWN_COURSES:
            hallucinations.append(f"Unknown course code: {code} (verify if real)")

    policy_claims = re.findall(
        r'(maximum|minimum|limit|cap|overload).{0,30}?(\d+)\s*credit', response.lower()
    )
    for context, num in policy_claims:
        hallucinations.append(
            f"Specific policy claim: '{context}...{num} credit' (verify against university policy)"
        )

    return hallucinations


def rule_based_score(question: BenchmarkQuestion) -> EvalResult:
    """Score a single question using rule-based heuristics."""
    result = EvalResult(
        question_id=question.id,
        category=question.category,
        subcategory=question.subcategory,
        question=question.question,
        student_context=question.student_context,
        reference_answer=question.reference_answer,
        actual_ai_response=question.actual_ai_response,
    )

    response = question.actual_ai_response
    student = question.student_context

    if not response or response.strip() == "" or response == "[DRY RUN]":
        result.scoring_method = "no-response"
        result.notes = "No AI response available"
        return result

    if response.startswith("[ERROR]"):
        result.scoring_method = "error"
        result.notes = f"Agent error: {response}"
        return result

    # NLP metrics (Pass 1)
    if question.reference_answer and "(No FAQ reference" not in question.reference_answer:
        nlp = compute_nlp_metrics(response, question.reference_answer)
        result.rouge_l = nlp["rouge_l"]
        result.meteor = nlp["meteor"]

    # Completeness
    result.completeness = detect_deflection(response)

    # Deferral detection
    is_def, def_reason = check_deferral(response, question.category)
    result.is_deferral = is_def
    result.deferral_reason = def_reason

    # Structured criteria (must_contain / must_not_contain)
    if question.must_contain:
        mc_pass, mc_missing = check_must_contain(response, question.must_contain)
        result.must_contain_pass = mc_pass
        result.must_contain_missing = mc_missing

    if question.must_not_contain:
        mnc_pass, mnc_found = check_must_not_contain(response, question.must_not_contain)
        result.must_not_contain_pass = mnc_pass
        result.must_not_contain_found = mnc_found

    # Factual errors
    errors = []
    errors.extend(check_course_accuracy(response))
    errors.extend(check_credit_accuracy(response))
    if student and student not in ("Any", "any_student"):
        errors.extend(check_student_specific_errors(response, student))
    result.factual_errors = errors

    # Hallucinations
    hallucinations = detect_hallucinations(response)
    result.hallucinated_claims = hallucinations
    result.hallucination_count = len(hallucinations)

    # Factual correctness
    if errors:
        result.factual_correctness = "Fail"
        result.scoring_method = "rule-based"
    elif result.completeness == "Deflected":
        result.factual_correctness = "Pass"
        result.scoring_method = "rule-based"
        result.notes = "Pass by default (no factual claims — deflected)"
    else:
        result.factual_correctness = "Pass"
        result.scoring_method = "rule-based-tentative"
        result.notes = "No rule-based errors found; LLM review recommended"

    if hallucinations:
        result.notes += "; flagged hallucinations need verification"
        result.scoring_method = "needs-review"

    # Overall pass (composite for new format)
    result.overall_pass = (
        result.factual_correctness == "Pass"
        and result.must_contain_pass
        and result.must_not_contain_pass
        and result.completeness != "Deflected"
    )

    return result


# ---------------------------------------------------------------------------
# Pass 3: LLM-as-judge (with logprob confidence from Kinga's approach)
# ---------------------------------------------------------------------------

LLM_SYSTEM_PROMPT = (
    "You are a fair judge assistant who has expertise in advising "
    "university students about the curriculum.\n\n"
)

def build_judge_prompt(question: BenchmarkQuestion, rule_result: EvalResult) -> str:
    """Build the prompt for LLM-as-judge evaluation."""
    student_context = ""
    profile = _resolve_profile(question.student_context)
    if profile:
        student_context = (
            f"\nStudent Profile ({question.student_context}):\n"
            f"- Current courses: {', '.join(profile.get('current_courses', []))}\n"
            f"- Completed: {', '.join(profile.get('completed', []))}\n"
            f"- Credits completed: {profile.get('credits_completed', 0)}\n"
            f"- Standing: {profile.get('standing_simple', 'freshman')}\n"
            f"- Notes: {profile.get('notes', '')}\n"
        )

    structured_info = ""
    if question.must_contain or question.must_not_contain:
        structured_info = f"""
STRUCTURED CRITERIA:
- must_contain check: {'PASS' if rule_result.must_contain_pass else 'FAIL — missing: ' + ', '.join(rule_result.must_contain_missing)}
- must_not_contain check: {'PASS' if rule_result.must_not_contain_pass else 'FAIL — found: ' + ', '.join(rule_result.must_not_contain_found)}
"""

    return f"""You are evaluating an AI academic advisor chatbot's response.

QUESTION ASKED BY STUDENT:
"{question.question}"

REFERENCE ANSWER (from the university's official FAQ):
"{question.reference_answer}"
{student_context}
AI CHATBOT RESPONSE:
"{question.actual_ai_response}"

RULE-BASED PRE-SCREENING:
- Factual errors: {rule_result.factual_errors if rule_result.factual_errors else 'None detected'}
- Potential hallucinations: {rule_result.hallucinated_claims if rule_result.hallucinated_claims else 'None detected'}
- Completeness (rule-based): {rule_result.completeness}
- ROUGE-L score: {rule_result.rouge_l}
- METEOR score: {rule_result.meteor}
{structured_info}
Evaluate and respond with ONLY a JSON object:
{{
  "factual_correctness": "Pass" or "Fail",
  "factual_errors_found": ["list of specific errors, or empty"],
  "hallucination_count": integer,
  "hallucinated_claims": ["list of fabricated claims, or empty"],
  "completeness": "Full" or "Partial" or "Deflected",
  "reasoning": "1-2 sentence explanation"
}}

CRITERIA:
- factual_correctness: "Fail" if ANY stated fact is wrong. "Pass" if all facts correct or no factual claims made.
- hallucination_count: Distinct fabricated claims not in reference, profile, or curriculum. Hedged guesses don't count; confident false claims do.
- completeness: "Full" = directly answers with actionable info. "Partial" = some useful info but incomplete. "Deflected" = only asks clarifying questions or says "ask your advisor."
"""


def call_llm(prompt: str, provider: str = "openai") -> tuple:
    """
    Call LLM provider and return (response_text, confidence).
    Confidence is derived from logprobs where supported.

    Returns:
        (str, float): response text and confidence score (0-1)
    """
    import numpy as np

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            max_retries=10,
        )
        chat_response = client.chat.completions.create(
            model=os.environ.get("LLM_JUDGE_MODEL", "gpt-4o"),
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            logprobs=True,
            seed=123,
        )
        text = chat_response.choices[0].message.content
        # Confidence from logprobs (Kinga's approach)
        confidence = 0.0
        if chat_response.choices[0].logprobs and chat_response.choices[0].logprobs.content:
            mean_logprob = np.mean([
                token.logprob for token in chat_response.choices[0].logprobs.content
            ])
            confidence = float(np.exp(mean_logprob))
        return text, confidence

    elif provider == "anthropic":
        from anthropic import Anthropic
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=os.environ.get("LLM_JUDGE_MODEL", "claude-sonnet-4-5-20250929"),
            max_tokens=1024,
            system=LLM_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        # Anthropic doesn't expose logprobs; confidence = 0 (unavailable)
        return response.content[0].text, 0.0

    elif provider == "bedrock":
        import boto3
        client = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        response = client.invoke_model(
            modelId=os.environ.get(
                "LLM_JUDGE_MODEL",
                "anthropic.claude-3-5-sonnet-20241022-v2:0",
            ),
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "system": LLM_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            }),
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"], 0.0

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai', 'anthropic', or 'bedrock'.")


def parse_llm_response(llm_output: str) -> dict:
    """Parse JSON from LLM judge output."""
    json_match = re.search(r'\{[\s\S]*\}', llm_output)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return {}


def llm_judge_score(
    question: BenchmarkQuestion,
    rule_result: EvalResult,
    provider: str = "openai",
) -> EvalResult:
    """Re-score using LLM-as-judge, with confidence from logprobs."""
    prompt = build_judge_prompt(question, rule_result)

    try:
        llm_output, confidence = call_llm(prompt, provider)
        parsed = parse_llm_response(llm_output)

        if parsed:
            rule_result.factual_correctness = parsed.get("factual_correctness", rule_result.factual_correctness)
            rule_result.hallucination_count = parsed.get("hallucination_count", rule_result.hallucination_count)
            rule_result.completeness = parsed.get("completeness", rule_result.completeness)
            rule_result.factual_errors = parsed.get("factual_errors_found", rule_result.factual_errors)
            rule_result.hallucinated_claims = parsed.get("hallucinated_claims", rule_result.hallucinated_claims)
            rule_result.notes = parsed.get("reasoning", "")
            rule_result.scoring_method = "llm-judge"
            rule_result.llm_confidence = round(confidence, 4)

            # Recompute overall_pass after LLM update
            rule_result.overall_pass = (
                rule_result.factual_correctness == "Pass"
                and rule_result.must_contain_pass
                and rule_result.must_not_contain_pass
                and rule_result.completeness != "Deflected"
            )
        else:
            rule_result.notes += "; LLM returned unparseable response"

    except ImportError as e:
        rule_result.notes += f"; Missing dependency: {e}"
    except Exception as e:
        rule_result.notes += f"; LLM judge error: {e}"

    return rule_result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def detect_format(path: str) -> str:
    """Detect whether a CSV is old format or new format based on headers."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
    if "test_id" in headers:
        return "new"
    if "ID" in headers:
        return "old"
    raise ValueError(f"Unrecognized CSV format. Headers: {headers}")


def load_benchmark_questions(path: str) -> list:
    """Load benchmark questions from old-format CSV."""
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("ID") or row["ID"].strip() == "":
                break
            questions.append(BenchmarkQuestion(
                id=int(row["ID"]),
                category=row["Category"],
                subcategory=row["Subcategory"],
                question=row["Benchmark Question"],
                student_context=row["Student Context"],
                reference_answer=row["Reference Answer"],
                source=row["Source"],
                actual_ai_response=row["Actual AI Response (from logs)"],
            ))
    return questions


def load_new_format(path: str) -> list:
    """Load benchmark questions from new-format CSV (runner output)."""
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_id = row.get("test_id", "").strip()
            if not test_id:
                continue
            questions.append(BenchmarkQuestion(
                id=test_id,
                category=row.get("category", ""),
                subcategory="",  # new format doesn't have subcategory
                question=row.get("query", ""),
                student_context=row.get("student_context", "any_student"),
                reference_answer=row.get("ground_truth_answer", ""),
                source="test_set_v3",
                actual_ai_response=row.get("ai_response", ""),
                must_contain=row.get("must_contain", ""),
                must_not_contain=row.get("must_not_contain", ""),
                factual_assertions=row.get("factual_assertions", ""),
                pass_criteria=row.get("pass_criteria", ""),
            ))
    return questions


# ---------------------------------------------------------------------------
# Results output
# ---------------------------------------------------------------------------

RESULT_FIELDS = [
    "question_id", "category", "subcategory", "question", "student_context",
    "reference_answer", "actual_ai_response",
    "rouge_l", "meteor",
    "factual_correctness", "hallucination_count", "completeness",
    "is_deferral", "deferral_reason",
    "must_contain_pass", "must_contain_missing",
    "must_not_contain_pass", "must_not_contain_found",
    "overall_pass",
    "scoring_method", "llm_confidence",
    "factual_errors", "hallucinated_claims", "notes",
]


def write_results(results: list, path: str):
    """Write scored results to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row["factual_errors"] = "; ".join(row["factual_errors"]) if row["factual_errors"] else ""
            row["hallucinated_claims"] = "; ".join(row["hallucinated_claims"]) if row["hallucinated_claims"] else ""
            row["must_contain_missing"] = "; ".join(row["must_contain_missing"]) if row["must_contain_missing"] else ""
            row["must_not_contain_found"] = "; ".join(row["must_not_contain_found"]) if row["must_not_contain_found"] else ""
            writer.writerow(row)


def print_summary(results: list, is_new_format: bool = False):
    """Print summary statistics."""
    scored = [r for r in results if r.factual_correctness]

    if not scored:
        print("No scored results.")
        return

    def stats(subset, label):
        if not subset:
            print(f"\n  {label}: no data")
            return
        n = len(subset)
        deferrals = sum(1 for r in subset if r.is_deferral)
        non_deferrals = [r for r in subset if not r.is_deferral]
        nd = len(non_deferrals)

        print(f"\n  {label} (n={n}):")
        print(f"    Deferral rate:        {deferrals}/{n} deferred ({100*deferrals/n:.0f}%)")

        if nd > 0:
            pass_ct = sum(1 for r in non_deferrals if r.factual_correctness == "Pass")
            avg_halluc = sum(r.hallucination_count for r in non_deferrals) / nd
            zero_h = sum(1 for r in non_deferrals if r.hallucination_count == 0)
            print(f"    Accuracy (answered):  {pass_ct}/{nd} correct ({100*pass_ct/nd:.0f}%) — excludes deferrals")
            print(f"    Hallucinations:       {avg_halluc:.2f} avg, {zero_h}/{nd} clean ({100*zero_h/nd:.0f}%)")

        full = sum(1 for r in subset if r.completeness == "Full")
        partial = sum(1 for r in subset if r.completeness == "Partial")
        deflected = sum(1 for r in subset if r.completeness == "Deflected")
        print(f"    Completeness:         Full={full} ({100*full/n:.0f}%)  "
              f"Partial={partial} ({100*partial/n:.0f}%)  "
              f"Deflected={deflected} ({100*deflected/n:.0f}%)")

        if is_new_format and nd > 0:
            mc_pass = sum(1 for r in non_deferrals if r.must_contain_pass)
            mnc_pass = sum(1 for r in non_deferrals if r.must_not_contain_pass)
            overall = sum(1 for r in subset if r.overall_pass)
            print(f"    must_contain (answered): {mc_pass}/{nd} pass ({100*mc_pass/nd:.0f}%)")
            print(f"    must_not_contain:      {mnc_pass}/{nd} pass ({100*mnc_pass/nd:.0f}%)")
            print(f"    Overall pass:         {overall}/{n} ({100*overall/n:.0f}%)")

    # NLP metric averages (only where computed)
    has_rouge = [r for r in scored if r.rouge_l > 0]
    has_meteor = [r for r in scored if r.meteor > 0]

    print("\n" + "=" * 65)
    print("  QA BENCHMARK RESULTS")
    print("=" * 65)

    # NLP metrics summary
    if has_rouge:
        avg_rouge = sum(r.rouge_l for r in has_rouge) / len(has_rouge)
        avg_meteor = sum(r.meteor for r in has_meteor) / len(has_meteor) if has_meteor else 0
        print(f"\n  NLP Metrics (vs reference answer, n={len(has_rouge)}):")
        print(f"    Avg ROUGE-L:  {avg_rouge:.3f}")
        print(f"    Avg METEOR:   {avg_meteor:.3f}")

    stats(scored, "Overall")

    if is_new_format:
        # Category breakdown for new format
        categories = sorted(set(r.category for r in scored))
        for cat in categories:
            cat_results = [r for r in scored if r.category == cat]
            stats(cat_results, f"Category: {cat}")
    else:
        # Old A/B breakdown
        cat_a = [r for r in scored if r.category == "A"]
        cat_b = [r for r in scored if r.category == "B"]
        stats(cat_a, "Category A (FAQ-mapped)")
        stats(cat_b, "Category B (Beyond-FAQ)")

    # LLM confidence summary
    llm_scored = [r for r in scored if r.scoring_method == "llm-judge"]
    if llm_scored:
        avg_conf = sum(r.llm_confidence for r in llm_scored) / len(llm_scored)
        print(f"\n  LLM Judge (n={len(llm_scored)}):")
        print(f"    Avg confidence:  {avg_conf:.3f}")

    # Deferrals detail
    deferrals = [r for r in scored if r.is_deferral]
    if deferrals:
        print(f"\n  DEFERRALS ({len(deferrals)}):")
        for r in deferrals:
            print(f"    {r.question_id}: {r.question[:55]}  [{r.deferral_reason}]")

    # Flagged issues (among non-deferrals)
    non_deferrals = [r for r in scored if not r.is_deferral]
    failures = [r for r in non_deferrals if r.factual_correctness == "Fail"]
    high_h = [r for r in non_deferrals if r.hallucination_count >= 2]

    if failures or high_h:
        print(f"\n  ERRORS (in answered questions):")
        if failures:
            print(f"    {len(failures)} factual failures:")
            for r in failures:
                print(f"      {r.question_id}: {r.question[:50]}...")
        if high_h:
            print(f"    {len(high_h)} responses with 2+ hallucinations")

    # must_contain failures detail (new format, non-deferrals only)
    if is_new_format:
        mc_failures = [r for r in non_deferrals if not r.must_contain_pass]
        mnc_failures = [r for r in non_deferrals if not r.must_not_contain_pass]
        if mc_failures or mnc_failures:
            print(f"\n  CONTENT GAPS (answered but missing key info):")
            for r in mc_failures:
                print(f"    {r.question_id} missing: {', '.join(r.must_contain_missing)}")
            for r in mnc_failures:
                print(f"    {r.question_id} should not contain: {', '.join(r.must_not_contain_found)}")

    print("\n" + "=" * 65)


# ---------------------------------------------------------------------------
# Multi-run aggregation
# ---------------------------------------------------------------------------

def aggregate_runs(input_paths: list[str], output_path: str | None = None):
    """
    Aggregate eval results across multiple runs. Loads each results CSV,
    computes per-question pass rates, and prints a summary.
    """
    from collections import defaultdict

    per_question = defaultdict(lambda: {"passes": 0, "total": 0, "mc_passes": 0, "mnc_passes": 0, "overall_passes": 0})
    all_results = []
    run_count = len(input_paths)

    for path in input_paths:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = row.get("question_id", "")
                if not qid:
                    continue
                pq = per_question[qid]
                pq["total"] += 1
                pq["category"] = row.get("category", "")
                pq["question"] = row.get("question", "")
                if row.get("factual_correctness") == "Pass":
                    pq["passes"] += 1
                if row.get("must_contain_pass", "True") == "True":
                    pq["mc_passes"] += 1
                if row.get("must_not_contain_pass", "True") == "True":
                    pq["mnc_passes"] += 1
                if row.get("overall_pass", "False") == "True":
                    pq["overall_passes"] += 1

    print(f"\n{'='*65}")
    print(f"  MULTI-RUN AGGREGATION ({run_count} runs)")
    print(f"{'='*65}")

    total_q = len(per_question)
    always_pass = sum(1 for pq in per_question.values() if pq["overall_passes"] == pq["total"])
    never_pass = sum(1 for pq in per_question.values() if pq["overall_passes"] == 0)

    print(f"\n  Questions: {total_q}")
    print(f"  Always pass (all runs): {always_pass}/{total_q} ({100*always_pass/total_q:.0f}%)")
    print(f"  Never pass (any run):   {never_pass}/{total_q} ({100*never_pass/total_q:.0f}%)")

    # Per-question detail
    print(f"\n  Per-Question Pass Rates:")
    print(f"  {'ID':<12} {'Category':<15} {'Factual':>10} {'MC':>8} {'MNC':>8} {'Overall':>10}")
    print(f"  {'-'*12} {'-'*15} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")
    for qid in sorted(per_question.keys()):
        pq = per_question[qid]
        t = pq["total"]
        print(
            f"  {qid:<12} {pq['category']:<15}"
            f" {pq['passes']}/{t:>5}"
            f" {pq['mc_passes']}/{t:>5}"
            f" {pq['mnc_passes']}/{t:>5}"
            f" {pq['overall_passes']}/{t:>5}"
        )

    # Category aggregation
    categories = sorted(set(pq["category"] for pq in per_question.values()))
    if len(categories) > 1:
        print(f"\n  Per-Category Averages:")
        for cat in categories:
            cat_qs = {qid: pq for qid, pq in per_question.items() if pq["category"] == cat}
            n = len(cat_qs)
            avg_pass = sum(pq["overall_passes"] / pq["total"] for pq in cat_qs.values()) / n
            print(f"    {cat:<15} {n} questions, avg overall pass rate: {100*avg_pass:.0f}%")

    print(f"\n{'='*65}")

    # Write aggregate CSV if requested
    if output_path:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["question_id", "category", "question", "runs",
                             "factual_pass_rate", "mc_pass_rate", "mnc_pass_rate", "overall_pass_rate"])
            for qid in sorted(per_question.keys()):
                pq = per_question[qid]
                t = pq["total"]
                writer.writerow([
                    qid, pq["category"], pq["question"][:80], t,
                    f"{pq['passes']/t:.2f}",
                    f"{pq['mc_passes']/t:.2f}",
                    f"{pq['mnc_passes']/t:.2f}",
                    f"{pq['overall_passes']/t:.2f}",
                ])
        print(f"  Aggregate CSV: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    use_llm = "--llm" in sys.argv
    provider = "openai"
    output_path = OUTPUT_PATH
    input_path = None

    if "--provider" in sys.argv:
        idx = sys.argv.index("--provider")
        if idx + 1 < len(sys.argv):
            provider = sys.argv[idx + 1]

    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_path = sys.argv[idx + 1]

    if "--input" in sys.argv:
        idx = sys.argv.index("--input")
        if idx + 1 < len(sys.argv):
            input_path = sys.argv[idx + 1]

    # Multi-run aggregation mode
    if "--aggregate" in sys.argv:
        idx = sys.argv.index("--aggregate")
        paths = sys.argv[idx + 1:]
        # Filter out other flags
        paths = [p for p in paths if not p.startswith("--")]
        if not paths:
            # Auto-find all eval result files
            paths = sorted(glob.glob(os.path.join(SCRIPT_DIR, "eval_results_*.csv")))
        if not paths:
            print("No result files found for aggregation.")
            sys.exit(1)
        agg_output = output_path if output_path != OUTPUT_PATH else os.path.join(SCRIPT_DIR, "aggregate_results.csv")
        aggregate_runs(paths, agg_output)
        return

    # Determine input file and format
    if input_path:
        fmt = detect_format(input_path)
    else:
        input_path = BENCHMARK_QUESTIONS_PATH
        fmt = "old"

    print(f"Loading: {input_path} (format: {fmt})")

    if fmt == "new":
        questions = load_new_format(input_path)
    else:
        questions = load_benchmark_questions(input_path)

    print(f"Loaded {len(questions)} benchmark questions\n")
    is_new_format = fmt == "new"

    # Check NLP dependencies
    nlp_available = try_import_nlp() is not None
    if nlp_available:
        print("Pass 1: NLP metrics (ROUGE-L, METEOR) — available")
    else:
        print("Pass 1: NLP metrics — SKIPPED (pip install nltk rouge-score numpy)")

    # Pass 2: Rule-based
    print("Pass 2: Rule-based scoring...")
    results = []
    for q in questions:
        results.append(rule_based_score(q))

    needs_review = [r for r in results if r.scoring_method in ("needs-review", "rule-based-tentative")]
    scored = [r for r in results if r.scoring_method not in ("no-response", "error")]
    print(f"  Scored: {len(scored)}, needs LLM review: {len(needs_review)}")

    # Pass 3: LLM-as-judge
    if use_llm:
        print(f"\nPass 3: LLM-as-judge (provider={provider})...")
        review_count = 0
        for i, result in enumerate(results):
            if result.scoring_method in ("needs-review", "rule-based-tentative"):
                q = questions[i]
                results[i] = llm_judge_score(q, result, provider)
                review_count += 1
                print(f"  [{review_count}/{len(needs_review)}] {result.question_id}: {results[i].scoring_method} "
                      f"(confidence={results[i].llm_confidence:.3f})")
    else:
        print("\nPass 3: LLM judge — SKIPPED (run with --llm to enable)")

    # Write results
    eval_output = output_path
    if is_new_format and output_path == OUTPUT_PATH:
        # Auto-name eval output based on input
        input_stem = Path(input_path).stem
        eval_output = os.path.join(SCRIPT_DIR, f"eval_results_{input_stem}.csv")
    write_results(results, eval_output)
    print(f"\nResults: {eval_output}")

    print_summary(results, is_new_format)


if __name__ == "__main__":
    main()
