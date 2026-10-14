"""
Shared Student Profile Data for Benchmark Scripts
==================================================
Mirrors the app's mockDegreeData.ts and getStudentContext() output.

Used by:
  - run_benchmark.py (to build context strings for Bedrock agent calls)
  - qa_benchmark_eval.py (for rule-based scoring lookups)
"""

# ---------------------------------------------------------------------------
# Known courses in the ME curriculum (for validation)
# ---------------------------------------------------------------------------

KNOWN_COURSES = {
    "CHEM 1113": "General Chemistry for Engineers",
    "MATH 1231": "Single-Variable Calculus I",
    "MATH 1232": "Single-Variable Calculus II",
    "MATH 2233": "Multivariable Calculus",
    "MATH 2184": "Linear Algebra I",
    "MAE 1001": "Introduction to Mechanical and Aerospace Engineering",
    "MAE 1004": "Engineering Drawing and Computer Graphics",
    "MAE 1117": "Introduction to Engineering Computations",
    "MAE 2117": "Engineering Computations",
    "MAE 2131": "Thermodynamics",
    "MAE 3119": "Electronics and Devices for Mechanical Engineers",
    "MAE 3120": "Methods of Engineering Experimentation",
    "MAE 3126": "Fluid Mechanics I",
    "MAE 3127": "Fluid Mechanics Lab",
    "MAE 3134": "Linear System Dynamics",
    "MAE 3166W": "Materials Science and Engineering",
    "MAE 3167W": "Mechanics of Materials Lab",
    "MAE 3187": "Heat Transfer",
    "MAE 3191": "Mechanical Design of Machine Elements",
    "MAE 3192": "Manufacturing Processes and Systems",
    "MAE 3193": "Mechanical Systems Design",
    "MAE 4149": "Thermal Systems Design",
    "MAE 4151W": "Capstone Design Project I",
    "MAE 4152W": "Capstone Design Project II",
    "MAE 4182": "Electromechanical Control System Design",
    "PHYS 1021": "University Physics I",
    "PHYS 1022": "University Physics II",
    "APSC 2057": "Analytical Mechanics I",
    "APSC 2058": "Analytical Mechanics II",
    "APSC 2113": "Engineering Analysis I",
    "APSC 3115": "Engineering Analysis III",
    "CE 2220": "Introduction to the Mechanics of Solids",
    "UW 1020": "University Writing",
    "SEAS 1001": "Engineering Orientation",
    "ECON 1011": "Principles of Economics I",
    "ECON 1012": "Principles of Economics II",
    "PSYC 1001": "General Psychology",
    "PSC 1001": "Introduction to Comparative Politics",
}

# ---------------------------------------------------------------------------
# Program-level facts
# ---------------------------------------------------------------------------

PROGRAM_FACTS = {
    "total_credits_required": 128,
    "degree_requirement_credits": 98,
    "non_technical_elective_credits": 18,
    "technical_elective_credits": 12,
    "advisor_name": "James Young",
    "program": "Mechanical Engineering",
    "degree": "Bachelor of Science",
    "college": "School of Engineering & Applied Science",
}

# ---------------------------------------------------------------------------
# Full student profiles (mirrors mockDegreeData.ts)
# ---------------------------------------------------------------------------

# Shared degree requirement courses (all 3 students have the same curriculum)
_DEGREE_REQUIREMENT_COURSES = [
    {"code": "CHEM 1113", "title": "General Chemistry for Engineers", "credits": 4},
    {"code": "MAE 1001", "title": "Introduction to Mechanical and Aerospace Engineering", "credits": 1},
    {"code": "MATH 1231", "title": "Single-Variable Calculus I", "credits": 3},
    {"code": "SEAS 1001", "title": "Engineering Orientation", "credits": 1},
    {"code": "UW 1020", "title": "University Writing", "credits": 4},
    {"code": "MATH 1232", "title": "Single-Variable Calculus II", "credits": 3},
    {"code": "MAE 1004", "title": "Engineering Drawing and Computer Graphics", "credits": 3},
    {"code": "MAE 1117", "title": "Introduction to Engineering Computations", "credits": 3},
    {"code": "MATH 2184", "title": "Linear Algebra I", "credits": 3},
    {"code": "PHYS 1021", "title": "University Physics I", "credits": 4},
    {"code": "APSC 2057", "title": "Analytical Mechanics I", "credits": 3},
    {"code": "APSC 2113", "title": "Engineering Analysis I", "credits": 3},
    {"code": "MAE 2117", "title": "Engineering Computations", "credits": 3},
    {"code": "MAE 3192", "title": "Manufacturing Processes and Systems", "credits": 3},
    {"code": "MATH 2233", "title": "Multivariable Calculus", "credits": 3},
    {"code": "APSC 2058", "title": "Analytical Mechanics II", "credits": 3},
    {"code": "APSC 3115", "title": "Engineering Analysis III", "credits": 3},
    {"code": "CE 2220", "title": "Introduction to the Mechanics of Solids", "credits": 3},
    {"code": "MAE 2131", "title": "Thermodynamics", "credits": 3},
    {"code": "PHYS 1022", "title": "University Physics II", "credits": 4},
    {"code": "MAE 3126", "title": "Fluid Mechanics I", "credits": 3},
    {"code": "MAE 3127", "title": "Fluid Mechanics Lab", "credits": 1},
    {"code": "MAE 3166W", "title": "Materials Science and Engineering", "credits": 3},
    {"code": "MAE 3191", "title": "Mechanical Design of Machine Elements", "credits": 3},
    {"code": "MAE 3119", "title": "Electronics and Devices for Mechanical Engineers", "credits": 3},
    {"code": "MAE 3120", "title": "Methods of Engineering Experimentation", "credits": 3},
    {"code": "MAE 3134", "title": "Linear System Dynamics", "credits": 3},
    {"code": "MAE 3167W", "title": "Mechanics of Materials Lab", "credits": 1},
    {"code": "MAE 3187", "title": "Heat Transfer", "credits": 3},
    {"code": "MAE 3193", "title": "Mechanical Systems Design", "credits": 3},
    {"code": "MAE 4149", "title": "Thermal Systems Design", "credits": 3},
    {"code": "MAE 4151W", "title": "Capstone Design Project I", "credits": 3},
    {"code": "MAE 4182", "title": "Electromechanical Control System Design", "credits": 3},
    {"code": "MAE 4152W", "title": "Capstone Design Project II", "credits": 3},
]

# Non-technical elective placeholder slots (5 empty + 1 filled per student)
_NT_ELECTIVE_PLACEHOLDERS = [
    {"code": "XXXX XXXX", "title": "Non-Technical Elective 2", "credits": 3, "status": "incomplete"},
    {"code": "XXXX XXXX", "title": "Non-Technical Elective 3", "credits": 3, "status": "incomplete"},
    {"code": "XXXX XXXX", "title": "Non-Technical Elective 4", "credits": 3, "status": "incomplete"},
    {"code": "XXXX XXXX", "title": "Non-Technical Elective 5", "credits": 3, "status": "incomplete"},
    {"code": "XXXX XXXX", "title": "Non-Technical Elective 6", "credits": 3, "status": "incomplete"},
]

# Technical elective placeholder slots (same for all students)
_TECH_ELECTIVE_SLOTS = [
    {"code": "XXXX XXXX", "title": "Technical Elective", "credits": 3, "status": "incomplete"},
    {"code": "XXXX XXXX", "title": "Technical Elective", "credits": 3, "status": "incomplete"},
    {"code": "XXXX XXXX", "title": "Technical Elective", "credits": 3, "status": "incomplete"},
    {"code": "XXXX XXXX", "title": "Technical Elective", "credits": 3, "status": "incomplete"},
]


def _build_degree_courses(completed_codes, in_progress_codes, term_overrides=None):
    """Build the degree requirement course list with statuses set per student."""
    term_overrides = term_overrides or {}
    courses = []
    for base in _DEGREE_REQUIREMENT_COURSES:
        c = dict(base)
        code = c["code"]
        if code in completed_codes:
            c["status"] = "complete"
            c["term"] = term_overrides.get(code, "Fall 2025")
            c["grade"] = completed_codes[code]
        elif code in in_progress_codes:
            c["status"] = "in-progress"
            c["term"] = in_progress_codes[code]
        else:
            c["status"] = "incomplete"
        courses.append(c)
    return courses


# --- Student A: Jordan Mitchell ---
_STUDENT_A_COMPLETED = {"MATH 1231": "A"}
_STUDENT_A_IN_PROGRESS = {
    "CHEM 1113": "Fall 2025",
    "MAE 1001": "Fall 2025",
    "SEAS 1001": "Fall 2025",
    "UW 1020": "Transfer",
    "MATH 1232": "Fall 2025",
}

# --- Student B: Alex Donovan ---
_STUDENT_B_COMPLETED = {"MATH 1231": "A"}
_STUDENT_B_IN_PROGRESS = {
    "CHEM 1113": "Fall 2025",
    "MAE 1001": "Fall 2025",
    "SEAS 1001": "Fall 2025",
    "UW 1020": "Fall 2025",
    "MATH 1232": "Fall 2025",
}

# --- Student C: Priya Sharma ---
_STUDENT_C_COMPLETED = {"MATH 1231": "A"}
_STUDENT_C_IN_PROGRESS = {
    "CHEM 1113": "Fall 2025",
    "MAE 1001": "Fall 2025",
    "SEAS 1001": "Fall 2025",
    "UW 1020": "Transfer",
    "MATH 1232": "Fall 2025",
}


STUDENT_PROFILES = {
    "studenta@gwsandbox.com": {
        "name": "Jordan Mitchell",
        "email": "studenta@gwsandbox.com",
        "gwid": "G12345678",
        "degree": "Bachelor of Science",
        "college": "School of Engineering & Applied Science",
        "major": "Mechanical Engineering",
        "class_level": "Freshman",
        "admit_term": "Fall 2025",
        "advisor": "James Young",
        "standing": "Good Standing",
        "overall_gpa": 0,
        "total_credits_required": 128,
        "total_credits_completed": 3,
        "requirement_blocks": [
            {
                "name": "Degree Requirement",
                "credits_required": 98,
                "credits_completed": 3,
                "courses": _build_degree_courses(
                    _STUDENT_A_COMPLETED, _STUDENT_A_IN_PROGRESS
                ),
            },
            {
                "name": "Non-Technical Elective",
                "credits_required": 18,
                "credits_completed": 0,
                "courses": [
                    {"code": "ECON 1011", "title": "Principles of Economics I", "credits": 3, "term": "Fall 2025", "status": "in-progress"},
                ] + _NT_ELECTIVE_PLACEHOLDERS,
            },
            {
                "name": "Technical Elective",
                "credits_required": 12,
                "credits_completed": 0,
                "courses": list(_TECH_ELECTIVE_SLOTS),
            },
        ],
        # Simplified fields for backward compat with eval rule-based checks
        "current_courses": ["MATH 1232", "ECON 1011"],
        "completed": ["MATH 1231"],
        "credits_completed": 3,
        "standing_simple": "freshman",
        "notes": "Ahead on math (taking Calc II already)",
    },
    "studentb@gwsandbox.com": {
        "name": "Alex Donovan",
        "email": "studentb@gwsandbox.com",
        "gwid": "G23456789",
        "degree": "Bachelor of Science",
        "college": "School of Engineering & Applied Science",
        "major": "Mechanical Engineering",
        "class_level": "Freshman",
        "admit_term": "Fall 2025",
        "advisor": "James Young",
        "standing": "Good Standing",
        "overall_gpa": 0,
        "total_credits_required": 128,
        "total_credits_completed": 3,
        "requirement_blocks": [
            {
                "name": "Degree Requirement",
                "credits_required": 98,
                "credits_completed": 3,
                "courses": _build_degree_courses(
                    _STUDENT_B_COMPLETED,
                    _STUDENT_B_IN_PROGRESS,
                    term_overrides={"MATH 1231": "Transfer"},
                ),
            },
            {
                "name": "Non-Technical Elective",
                "credits_required": 18,
                "credits_completed": 0,
                "courses": [
                    {"code": "PSYC 1001", "title": "General Psychology", "credits": 3, "term": "Fall 2025", "status": "in-progress"},
                ] + _NT_ELECTIVE_PLACEHOLDERS,
            },
            {
                "name": "Technical Elective",
                "credits_required": 12,
                "credits_completed": 0,
                "courses": list(_TECH_ELECTIVE_SLOTS),
            },
        ],
        "current_courses": ["MATH 1232", "PSYC 1001"],
        "completed": ["MATH 1231"],
        "credits_completed": 3,
        "standing_simple": "freshman",
        "notes": "Has AP Physics 1 credit pending (not yet in system); possibly AP Linear Algebra",
    },
    "studentc@gwsandbox.com": {
        "name": "Priya Sharma",
        "email": "studentc@gwsandbox.com",
        "gwid": "G34567890",
        "degree": "Bachelor of Science",
        "college": "School of Engineering & Applied Science",
        "major": "Mechanical Engineering",
        "class_level": "Freshman",
        "admit_term": "Fall 2025",
        "advisor": "James Young",
        "standing": "Good Standing",
        "overall_gpa": 0,
        "total_credits_required": 128,
        "total_credits_completed": 3,
        "requirement_blocks": [
            {
                "name": "Degree Requirement",
                "credits_required": 98,
                "credits_completed": 3,
                "courses": _build_degree_courses(
                    _STUDENT_C_COMPLETED, _STUDENT_C_IN_PROGRESS
                ),
            },
            {
                "name": "Non-Technical Elective",
                "credits_required": 18,
                "credits_completed": 0,
                "courses": [
                    {"code": "PSC 1001", "title": "Introduction to Comparative Politics", "credits": 3, "term": "Fall 2025", "status": "in-progress"},
                ] + _NT_ELECTIVE_PLACEHOLDERS,
            },
            {
                "name": "Technical Elective",
                "credits_required": 12,
                "credits_completed": 0,
                "courses": list(_TECH_ELECTIVE_SLOTS),
            },
        ],
        "current_courses": ["MATH 1232", "UW 1020", "CHEM 1113", "MAE 1001", "SEAS 1001", "PSC 1001"],
        "completed": ["MATH 1231"],
        "credits_completed": 3,
        "standing_simple": "freshman",
        "notes": "Considering dropping Calc II; international student",
    },
}

# ---------------------------------------------------------------------------
# Context map: test CSV values → email keys
# ---------------------------------------------------------------------------

STUDENT_CONTEXT_MAP = {
    "student_a": "studenta@gwsandbox.com",
    "student_b": "studentb@gwsandbox.com",
    "student_c": "studentc@gwsandbox.com",
    "any_student": "studenta@gwsandbox.com",  # default to Student A
}


# ---------------------------------------------------------------------------
# build_student_context() — replicates getStudentContext() from route.ts
# ---------------------------------------------------------------------------

def build_student_context(student_key: str) -> str:
    """
    Build the student context string in the same format as the app's
    getStudentContext() (src/app/api/ai-advisor/route.ts:18-62).

    Args:
        student_key: Either an email (studenta@gwsandbox.com) or a short key
                     (student_a, any_student) from the test CSV.

    Returns:
        Multi-line context string matching the app's format.
    """
    # Resolve short keys to emails
    email = STUDENT_CONTEXT_MAP.get(student_key, student_key)
    profile = STUDENT_PROFILES.get(email)
    if not profile:
        return ""

    blocks = []
    for block in profile["requirement_blocks"]:
        course_lines = []
        for c in block["courses"]:
            parts = [f"{c['code']} {c['title']} ({c['credits']}cr)"]
            status = c.get("status", "incomplete")
            # NOTE: The app's route.ts checks status === "completed" but the
            # mock data uses "complete" — so completed courses fall through to
            # "not yet taken" in production. We replicate that behavior here.
            if status == "completed":
                grade = c.get("grade")
                term = c.get("term", "")
                if grade:
                    parts.append(f"grade: {grade}, {term}")
                else:
                    parts.append(f"completed, {term}")
            elif status == "in-progress":
                parts.append(f"in-progress, {c.get('term', '')}")
            else:
                parts.append("not yet taken")
            course_lines.append(f"  - {' — '.join(parts)}")

        block_header = f"{block['name']} ({block['credits_completed']}/{block['credits_required']} credits):"
        blocks.append(f"{block_header}\n" + "\n".join(course_lines))

    lines = [
        f"Student: {profile['name']} ({profile['email']})",
        f"GWID: {profile['gwid']}",
        f"Degree: {profile['degree']}, Major: {profile['major']}",
        f"College: {profile['college']}",
        f"Class Level: {profile['class_level']}, Admit Term: {profile['admit_term']}",
        f"Advisor: {profile['advisor']}",
        f"Standing: {profile['standing']}, GPA: {profile['overall_gpa']}",
        f"Credits: {profile['total_credits_completed']}/{profile['total_credits_required']}",
        "",
        "Degree Requirements:",
    ]
    lines.extend(blocks)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for key in ["student_a", "student_b", "student_c"]:
        print(f"\n{'='*70}")
        print(f"  Context for: {key}")
        print(f"{'='*70}")
        print(build_student_context(key))
        print()
