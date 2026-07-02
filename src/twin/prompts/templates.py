"""Creator / solver prompt templates (DESIGN.md §5, §9).

Plain string builders (no model dependency, so they're unit-testable). The
trainer wraps the returned (system, user) pair with ``TwinBase.render`` to apply
the chat template. The creator is asked for the exact JSON contract that
``twin.problems.schema.parse_suite`` consumes; the solver is asked to finish
with an ``ANSWER:`` line that ``extract_final_answer`` can pick up.
"""

import random

from twin.problems.schema import Problem
from twin.tools import SOLVE_TOOL_DOC

# Small theme pools per domain — just enough variety to avoid every iteration
# proposing the same suite. Extend freely; broadening domains is a config change.
THEMES: dict[str, list[str]] = {
    "math": [
        "linear equations", "quadratic equations", "arithmetic sequences",
        "percentages and ratios", "areas and perimeters", "exponents and roots",
        "systems of two equations", "basic probability", "divisibility",
        "simple interest",
    ],
    "coding": [
        "string manipulation", "list aggregation", "number theory helpers",
        "recursion basics", "dictionary counting", "sorting and searching",
    ],
}


def pick_theme(domain: str, rng: random.Random) -> str:
    pool = THEMES.get(domain) or THEMES["math"]
    return rng.choice(pool)


# --------------------------------------------------------------------------- #
# Creator
# --------------------------------------------------------------------------- #
CREATOR_SYSTEM = (
    "You are a problem-setter building a graded practice set.\n\n"
    "You have a computer-algebra tool. To call it, write a line of the form\n"
    "  <tool>solve(...)</tool>\n"
    "and stop; you will receive the result as <obs>RESULT</obs> and then continue.\n"
    f"{SOLVE_TOOL_DOC}\n"
    "Use the tool to compute the correct final answer for EVERY problem you pose "
    "— do not guess answers. For example:\n"
    "  <tool>solve(x**2 - 5*x + 6, x, max)</tool>\n"
    "  <obs>3</obs>\n"
    "When you are finished, output ONLY a single JSON object (no prose, no "
    "markdown fences) containing the problems and their tool-checked answers."
)


# Domains whose problems must carry the machine-checkable math certificate
# (verification.check + verification.symbol -> twin.verifiers.check_predicate).
_MATH_DOMAINS = {"math", "arithmetic", "algebra"}

# The certificate field spec + rules, spliced into the math creator prompt.
# The check is verified mechanically by a CAS (Sprint 5): it replaces the LLM
# judge for consistency, so it must RECOMPUTE the answer from the problem's
# quantities — "x = <answer>" is rejected as trivial.
_MATH_VERIFICATION_FIELD = (
    ',\n      "verification": {\n'
    '        "type": "math",\n'
    '        "symbol": "<the unknown your answer gives, e.g. \\"x\\" '
    '(or \\"x, y\\" for a tuple answer)>",\n'
    '        "check": "<equation(s) in the symbol that hold exactly when the '
    'symbol equals your answer>"\n'
    "      }"
)

_MATH_VERIFICATION_RULES = (
    " The \"check\" certificate is verified mechanically by a computer-algebra "
    "system: substituting your answer for the symbol must satisfy it, and a "
    "problem whose check fails or is missing is DISCARDED (it earns you "
    "nothing). The check must encode the problem's defining equation or "
    "computation — e.g. statement \"Tickets cost $4; how many can you buy with "
    "$20?\" -> check \"4*x = 20\"; statement \"What is 15% of 80?\" -> check "
    "\"x = 0.15*80\". Do NOT write \"x = <your answer>\" (rejected as trivial). "
    "For several unknowns use symbol \"x, y\", answer \"(6, 4)\", check "
    "\"x + y = 10, x - y = 2\"."
)


def creator_user(domain: str, theme: str, n_problems: int) -> str:
    """Ask for a suite of ``n_problems`` on ``theme`` spanning easy->hard."""
    is_math = domain.lower() in _MATH_DOMAINS
    verification_field = _MATH_VERIFICATION_FIELD if is_math else ""
    verification_rules = _MATH_VERIFICATION_RULES if is_math else ""
    return (
        f"Create a set of {n_problems} {domain} problems about \"{theme}\".\n"
        f"The problems must span a smooth difficulty ramp from very easy to very "
        f"hard, so that a typical student would solve the easiest reliably and the "
        f"hardest rarely. Make them genuinely distinct.\n\n"
        f"Use the solve tool to compute and check each answer before writing the "
        f"JSON. Then return ONLY this JSON object:\n"
        "{\n"
        f'  "theme": "{theme}",\n'
        f'  "domain": "{domain}",\n'
        '  "problems": [\n'
        "    {\n"
        '      "statement": "<the problem, fully self-contained>",\n'
        '      "difficulty": <number 0.0 (easiest) to 1.0 (hardest)>,\n'
        '      "answer": "<the single final answer, e.g. a number or closed form>",\n'
        '      "solution": "<a short worked solution justifying the answer>"'
        f"{verification_field}\n"
        "    }\n"
        "    // ... exactly "
        f"{n_problems} problems, with strictly increasing difficulty\n"
        "  ]\n"
        "}\n\n"
        "Rules: difficulties strictly increase across the list; every answer must "
        "be correct and follow from its solution; keep each statement unambiguous "
        "with a unique answer." + verification_rules
    )


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
SOLVER_SYSTEM = (
    "You are a careful problem solver. Work briefly, then state the final answer "
    "on its own last line in the exact form 'ANSWER: <value>'. Write <value> as a "
    "plain mathematical value only — no LaTeX, no '$', no units, no extra words. "
    "Use a bare number with exact fractions (write 7/2, not 3.5 or \\frac{7}{2}); "
    "for several values give an ordered tuple in parentheses, e.g. (3, 2)."
)


def solver_user(problem: Problem | str) -> str:
    statement = problem.statement if isinstance(problem, Problem) else str(problem)
    return (
        f"Solve this problem:\n\n{statement}\n\n"
        "Show your reasoning concisely, then end with a final line in exactly this "
        "form (plain value, no LaTeX or units; tuples like (3, 2) for several values):\n"
        "ANSWER: <your final answer>"
    )


# --------------------------------------------------------------------------- #
# Judge (verifier fallback) — tool-augmented so it recomputes rather than guesses
# --------------------------------------------------------------------------- #
# The judge is the frozen base used to grade non-deterministic answers/solutions.
# Eyeballing a worked solution is exactly where an LLM slips on arithmetic, so we
# hand the judge the same computer-algebra tool the creator uses and require it to
# recompute the answer itself before ruling. The protocol mechanics live here (the
# system prompt); the grading task + the VERDICT contract live in twin.verifiers.judge.
JUDGE_SYSTEM = (
    "You are a meticulous grader with a computer-algebra tool. Do NOT trust mental "
    "arithmetic or the worked solution you are shown — independently recompute every "
    "value with the tool before you rule.\n\n"
    "To call the tool, write a line of the form\n"
    "  <tool>solve(...)</tool>\n"
    "and stop; you will receive the result as <obs>RESULT</obs> and then continue.\n"
    f"{SOLVE_TOOL_DOC}\n"
    "Solve the problem yourself with the tool, compare your tool-checked result to the "
    "value being graded, then finish with exactly one line:\n"
    "VERDICT: CORRECT   or   VERDICT: INCORRECT\n"
    "If a tool call returns an empty list, an error, or an obviously degenerate result, "
    "your query was malformed — fix it and retry (e.g. write an equation as "
    "'4*x = 20', not '4*x == 20'). Never rule INCORRECT merely because a tool call "
    "came back empty."
)
