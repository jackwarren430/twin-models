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
    "logic": [
        "direct accusations", "mutual references", "chained implications",
        "self-referential statements", "contradicting pairs",
        "statements about statements", "exclusive-or claims",
        "silent characters others talk about",
    ],
}


def pick_theme(
    domain: str,
    rng: random.Random,
    weights: dict[str, list] | None = None,
) -> str:
    """Pick this iteration's theme. Uniform over the static ``THEMES`` pool by
    default. ``weights`` (Sprint 9 — SPICE-style target grounding,
    RELATED_WORK.md §3) maps a domain to ``[[theme, weight], ...]`` — usually
    written by ``scripts/build_grounded_themes.py`` from a held-out-benchmark
    result, so the curriculum leans toward measured failure topics instead of
    a fixed hand-picked list. Domains absent from ``weights`` (or with empty/
    non-positive entries) fall back to the static pool."""
    if weights and weights.get(domain):
        entries = [(str(t), float(w)) for t, w in weights[domain] if float(w) > 0]
        if entries:
            themes, ws = zip(*entries)
            return rng.choices(themes, weights=ws, k=1)[0]
    pool = THEMES.get(domain) or THEMES["math"]
    return rng.choice(pool)


# --------------------------------------------------------------------------- #
# Personas (Sprint 7)
# --------------------------------------------------------------------------- #
# Names are glued to ADAPTERS, not roles: A is always "alpha", B always
# "omega", whatever role each is playing this iteration. The judge and the
# held-out benchmark never see personas (they must stay neutral graders).
PERSONAS: dict[str, str] = {"A": "alpha", "B": "omega"}


def persona_of(model: str) -> str:
    return PERSONAS.get(model, model)


def opponent_of(model: str) -> str:
    others = [v for k, v in PERSONAS.items() if k != model]
    return others[0] if len(others) == 1 else "your opponent"


def creator_persona(model: str) -> str:
    """Competition framing for the creator. Deliberately worded as a
    *calibration* game ("predict exactly what the opponent can and cannot
    solve"), not raw stumping — an all-impossible suite loses on the gradient
    reward, and mini-03a showed aspirational difficulty language alone sends a
    thinking model into deliberation spirals."""
    me, opp = persona_of(model), opponent_of(model)
    return (
        f"You are \"{me}\", in a fair competition against \"{opp}\". Right now "
        f"you set the problems and {opp} must solve them. You win by predicting "
        f"exactly what {opp} can and cannot solve: each problem comes with a "
        f"target solve rate, and you score highest when {opp}'s actual success "
        f"rate lands on that target. Be creative — for the hardest targets, "
        f"invent problems you are confident {opp} cannot crack; an unexpected "
        f"structural twist beats bigger numbers every time."
    )


def solver_persona(model: str) -> str:
    me, opp = persona_of(model), opponent_of(model)
    return (
        f"You are \"{me}\", in a fair competition against \"{opp}\". {opp} "
        f"designed this problem to probe the edge of what you can solve — "
        f"solving a problem {opp} bet you would miss is how you win. Be sharp "
        f"and resourceful."
    )


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
    "markdown fences) containing the problems and their tool-checked answers.\n"
    "Your output budget is limited: keep any hidden reasoning brief (a short "
    "plan plus tool checks). If you spend the budget deliberating, the output "
    "is truncated before the JSON and the whole set is discarded."
)


def creator_system(*, native_tools: bool = False, persona: str | None = None) -> str:
    """Creator system prompt. ``native_tools=True`` drops the legacy
    ``<tool>...</tool>`` protocol markup — the chat template declares the tools
    and Qwen3 emits native ``<tool_call>`` blocks it was actually trained on
    (the legacy protocol produced ZERO real calls in mini-03b; the model
    simulated the tool inside <think> instead). ``persona`` (from
    :func:`creator_persona`) is prepended when the run has personas on.

    The native body is the mini-04(a) rewrite (2026-07-04): two iterations
    measured ZERO real creator tool calls under the previous wording. The
    transcript showed why — the model concluded mid-<think> that "I can't
    actually run the function here, I need to simulate", and the instruction
    to output "ONLY a single JSON object" read as forbidding a tool call
    before the JSON. The rewrite names a two-phase protocol (VERIFY with a
    real call, then DELIVER the JSON), states explicitly that tools cannot
    run inside private reasoning, and keeps the anti-deliberation warning
    that fixed mini-03a's truncation spirals."""
    if native_tools:
        body = (
            "You are a problem-setter building a graded practice set.\n\n"
            "You have a computer-algebra tool (`solve`) available as a REAL "
            "function call: when you emit a tool call, the runtime executes "
            "it and sends you the result. The tool can NOT run inside your "
            "private reasoning — deliberating about what it would return is "
            "worthless. Work in two phases, every time:\n"
            "  Phase 1 — VERIFY: as soon as you have a candidate problem, stop "
            "reasoning and emit the tool call that computes its answer. Wait "
            "for the response.\n"
            "  Phase 2 — DELIVER: after the tool response arrives, output the "
            "single JSON object (no prose around it, no markdown fences).\n"
            "Keep your private reasoning to a few sentences: pick the first "
            "workable design and go verify it with the tool. If you spend the "
            "budget deliberating, the output is truncated and the problem is "
            "discarded."
        )
    else:
        body = CREATOR_SYSTEM
    return f"{persona}\n\n{body}" if persona else body


def solver_system(
    *, persona: str | None = None, code: bool = False, logic: bool = False
) -> str:
    """Solver system prompt, optionally with the competition persona
    prepended. ``code=True`` selects the programming variant (Sprint 8 coding
    pipeline) — the math ANSWER-line contract makes no sense for code, which
    is graded by executing the solver's function against the creator's tests.
    ``logic=True`` selects the knights-and-knaves variant (assignment-format
    ANSWER line). The held-out benchmark builds its own prompts and never
    passes a persona, so it stays neutral by construction."""
    body = SOLVER_CODE_SYSTEM if code else (SOLVER_LOGIC_SYSTEM if logic else SOLVER_SYSTEM)
    return f"{persona}\n\n{body}" if persona else body


# Domains whose problems must carry the machine-checkable math certificate
# (verification.check + verification.symbol -> twin.verifiers.check_predicate).
_MATH_DOMAINS = {"math", "arithmetic", "algebra"}
# Domains verified by executing code against tests (twin.verifiers.verify_code).
# Mirrors twin.verifiers.dispatch._CODE_DOMAINS.
_CODE_DOMAINS = {"coding", "code", "python"}
# Domains verified by knight/knave enumeration (twin.verifiers.verify_logic).
# Mirrors twin.verifiers.dispatch._LOGIC_DOMAINS.
_LOGIC_DOMAINS = {"logic", "puzzle", "puzzles", "knights-and-knaves"}

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
    "\"x + y = 10, x - y = 2\". The example equations above are format "
    "illustrations ONLY — never use them, or trivial variants of them, as "
    "your problem."
    # mini-04b measured both adapters submitting the "x + y = 10" example
    # verbatim as a problem (EXPERIMENTS.md checkpoint-15) — hence the ban.
)

# The coding contract (Sprint 8 pipeline): the creator must ship an executable
# reference solution AND the tests that define correctness — the consistency
# check literally runs solution_code + tests in the sandbox, and the solver is
# graded by running ITS code against the same tests. mini-02's coding domain
# scored 0/141 consistent because the prompt never asked for any of this and
# verify_code fails closed without tests.
_CODE_VERIFICATION_FIELD = (
    ',\n      "verification": {\n'
    '        "type": "code",\n'
    '        "entry_point": "<the exact function name the statement asks for>",\n'
    '        "solution_code": "<complete working Python defining that '
    'function>",\n'
    '        "tests": "<assert statements calling the function, one per line>"\n'
    "      }"
)

_CODE_VERIFICATION_RULES = (
    " The problem is verified mechanically: \"solution_code\" and \"tests\" "
    "are executed together in a sandbox, and a problem whose own solution "
    "fails its own tests (or that omits either field) is DISCARDED — it earns "
    "you nothing. The statement must name the required function and its "
    "signature exactly (e.g. \"Write a function count_vowels(s) that ...\"); "
    "\"tests\" must be 3-6 plain assert lines calling that function, covering "
    "a normal case and an edge case (e.g. assert count_vowels(\"abc\") == 1); "
    "deterministic pure Python only — no input(), files, network, randomness, "
    "or imports beyond the standard library. Set \"answer\" to the entry-point "
    "name. Use the run_python tool to RUN solution_code plus the tests before "
    "writing the JSON — an untested solution is usually a discarded one."
)


# The logic contract (domain expansion 2026-07-04): Knights & Knaves, the
# Logic-RL / Reasoning Gym verifiable-logic staple. The certificate is one
# formal claims string; consistency enumerates all 2^n assignments and
# requires EXACTLY ONE (well-posedness), and the solver is graded against
# that enumerated solution — never against the creator's answer string.
_LOGIC_VERIFICATION_FIELD = (
    ',\n      "verification": {\n'
    '        "type": "logic",\n'
    '        "claims": "<one \'Speaker: statement\' per character who speaks, '
    "';'-separated, formalizing the statements in your story>\"\n"
    "      }"
)

_LOGIC_VERIFICATION_RULES = (
    " Every character is a knight (always tells the truth) or a knave "
    "(always lies); the solver must identify everyone. The \"claims\" "
    "certificate is verified mechanically by enumerating all knight/knave "
    "assignments: a puzzle with zero or several solutions is DISCARDED (it "
    "earns you nothing), so is one whose answer disagrees with the unique "
    "solution. Claim syntax: a name means 'that person is a knight'; "
    "operators ~ (not), & (and), | (or), ^ (exactly one), -> (implies), "
    "parentheses — so 'Ava: Ben & ~Cal' formalizes Ava saying 'Ben is a "
    "knight and Cal is a knave'. The statement must tell the story in "
    "natural language and quote every speaker faithfully; the claims must "
    "formalize exactly those quotes. Difficulty comes from more characters "
    "(2-8) and deeper nesting or implication chains, never from vagueness. "
    "Set \"answer\" to the full assignment like \"Ava=knight, Ben=knave, "
    "Cal=knight\". The syntax examples above are format illustrations ONLY "
    "— invent your own characters and claims."
)


def _verification_spec(domain: str) -> tuple[str, str, str, str]:
    """Per-domain contract pieces: (field, rules, tool sentence, answer desc)."""
    d = domain.lower()
    if d in _LOGIC_DOMAINS:
        return (
            _LOGIC_VERIFICATION_FIELD,
            _LOGIC_VERIFICATION_RULES,
            "Before writing any JSON, CALL the logic_solve tool on your "
            "claims — it reports whether exactly one solution exists and "
            "what it is; read your \"answer\" directly off the tool "
            "response, never solve the puzzle in your head.",
            "<the full assignment, e.g. \"Ava=knight, Ben=knave\">",
        )
    if d in _MATH_DOMAINS:
        return (
            _MATH_VERIFICATION_FIELD,
            _MATH_VERIFICATION_RULES,
            "Before writing any JSON, CALL the solve tool to compute and "
            "confirm each answer — a real tool call with a real response, "
            "never an imagined one.",
            "<the single final answer your solution yields, e.g. a number or "
            "closed form>",
        )
    if d in _CODE_DOMAINS:
        return (
            _CODE_VERIFICATION_FIELD,
            _CODE_VERIFICATION_RULES,
            "Use the run_python tool to execute each solution_code together "
            "with its tests before writing the JSON.",
            "<the entry-point function name>",
        )
    return (
        "", "",
        "Use the solve tool to check any computation before writing the JSON.",
        "<the single final answer your solution yields>",
    )


def creator_user(domain: str, theme: str, n_problems: int) -> str:
    """Ask for a suite of ``n_problems`` on ``theme`` spanning easy->hard."""
    verification_field, verification_rules, tool_sentence, answer_desc = (
        _verification_spec(domain))
    return (
        f"Create a set of {n_problems} {domain} problems about \"{theme}\".\n"
        f"The problems must span a smooth difficulty ramp from easy to genuinely "
        f"hard, judged against a strong solver: the easiest solved reliably, the "
        f"hardest solved only rarely. Hard means structurally hard — several "
        f"dependent steps or combined concepts — never merely bigger numbers or "
        f"more tedious arithmetic. Make the problems genuinely distinct.\n"
        f"Design decisively: commit to the first workable idea for each problem, "
        f"verify its answer with the tool, and write the JSON. Do not deliberate "
        f"over candidate designs — a long deliberation gets your output cut off "
        f"before the JSON, which scores nothing.\n\n"
        f"{tool_sentence} Then return ONLY this JSON object:\n"
        "{\n"
        f'  "theme": "{theme}",\n'
        f'  "domain": "{domain}",\n'
        '  "problems": [\n'
        "    {\n"
        '      "statement": "<the problem, fully self-contained>",\n'
        '      "difficulty": <number 0.0 (easiest) to 1.0 (hardest)>,\n'
        '      "solution": "<a short worked solution deriving the answer>",\n'
        f'      "answer": "{answer_desc}"'
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
# Per-problem creator prompt (Sprint 7, game.creator_mode = "per_problem")
# --------------------------------------------------------------------------- #
def _difficulty_brief(target_rate: float, opponent: str) -> str:
    """Rank-specific design instruction, keyed on the target solve rate. Keeps
    the decisive-design phrasing that fixed mini-03a's think-spiral."""
    pct = f"{round(target_rate * 100)}%"
    if target_rate <= 0.2:
        return (
            f"This is the top of the ramp: {opponent} should solve it only about "
            f"{pct} of the time. Make the hardest problem you can — one you are "
            f"confident {opponent} will almost never crack. Hard means "
            f"structurally hard (several dependent steps or combined concepts), "
            f"never merely bigger numbers or more tedious arithmetic. Design it "
            f"in ONE pass: pick a structure, verify it with the tool, commit — "
            f"searching for a better design will truncate your output and score "
            f"nothing (a merely-good hard problem beats a discarded perfect one)."
        )
    if target_rate >= 0.8:
        return (
            f"This is the easy end of the ramp: {opponent} should solve it about "
            f"{pct} of the time — a clean warm-up {opponent} will essentially "
            f"never miss."
        )
    return (
        f"{opponent} should solve this one about {pct} of the time: genuinely "
        f"challenging, but within reach on a good attempt."
    )


def creator_problem_user(
    domain: str,
    theme: str,
    *,
    rank: int,
    n_problems: int,
    difficulty: float,
    target_rate: float,
    previous: list[str] | None = None,
    opponent: str | None = None,
) -> str:
    """Ask for ONE problem: rank ``rank`` (0-based) of ``n_problems``, with a
    dictated ``difficulty`` value and a target solve rate for the opposing
    solver. ``previous`` carries the JSONs of the already-written problems
    (never their thinking — that's the memory point of per-problem mode);
    pass None when ``game.condition_on_previous`` is off."""
    opp = opponent or "the solver"
    verification_field, verification_rules, tool_sentence, answer_desc = (
        _verification_spec(domain))

    if previous:
        prev_block = (
            "You already wrote these problems for this set (as JSON, "
            "easiest first):\n"
            + "\n".join(previous)
            + "\n\nThis problem must be strictly harder than all of them and "
            "genuinely distinct — do not reuse their structure or dress the "
            "same computation in a new story.\n\n"
        )
    else:
        prev_block = ""

    return (
        f"You are writing problem {rank + 1} of {n_problems} in a graded "
        f"{domain} set about \"{theme}\". The set forms a difficulty gradient "
        f"from 0.0 (easiest) to 1.0 (hardest); this problem's difficulty is "
        f"{difficulty:.2f}.\n"
        f"{_difficulty_brief(target_rate, opp)}\n\n"
        f"{prev_block}"
        f"Design decisively: commit to the first workable idea, verify it "
        f"with the tool, and write the JSON. Do not deliberate "
        f"over candidate designs — a long deliberation gets your output cut "
        f"off before the JSON, which scores nothing.\n\n"
        f"{tool_sentence} "
        f"Work out the solution first, then state the answer it yields. Return "
        f"ONLY this JSON object (one problem, no wrapper list):\n"
        "{\n"
        '  "statement": "<the problem, fully self-contained>",\n'
        f'  "difficulty": {difficulty:.2f},\n'
        '  "solution": "<a short worked solution deriving the answer>",\n'
        f'  "answer": "{answer_desc}"'
        f"{_indent_verification(verification_field)}\n"
        "}\n\n"
        "Rules: the answer must be tool-checked and follow from the solution; "
        "keep the statement unambiguous with a unique answer."
        + verification_rules
    )


def _indent_verification(field_spec: str) -> str:
    """The suite contract's verification block is indented for its nesting
    depth (6 spaces); the single-problem contract sits two levels shallower."""
    return field_spec.replace("\n      ", "\n  ").replace("\n        ", "\n    ")


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

# Coding variant (Sprint 8 pipeline): the answer IS the code — it is graded by
# running the solver's function against the creator's hidden tests.
SOLVER_CODE_SYSTEM = (
    "You are a careful programmer. Think briefly, then reply with exactly one "
    "fenced Python code block (```python ... ```) containing your complete, "
    "self-contained solution. Define exactly the function the problem names, "
    "with the stated signature; standard library only; no prints, no example "
    "usage, no prose outside the code block."
)

# Logic variant (domain expansion 2026-07-04): knights-and-knaves answers are
# full assignments, graded against the certificate's enumerated solution.
SOLVER_LOGIC_SYSTEM = (
    "You are a careful logic-puzzle solver. Every character is a knight (always "
    "tells the truth) or a knave (always lies). Work through the statements "
    "case by case, then state the final answer on its own last line in the "
    "exact form 'ANSWER: <assignment>', where <assignment> names EVERY "
    "character once, comma-separated, e.g. 'ANSWER: Ava=knight, Ben=knave'. "
    "No other words on that line."
)


def is_code_problem(problem: "Problem | str") -> bool:
    """Whether this problem is graded by executing code (mirrors
    twin.verifiers.dispatch method selection: explicit verification type wins,
    else the domain decides)."""
    if not isinstance(problem, Problem):
        return False
    vtype = str(problem.verification.get("type", "")).strip().lower()
    if vtype:
        return vtype.startswith(("code", "exec", "python"))
    return problem.domain.lower() in _CODE_DOMAINS


def is_logic_problem(problem: "Problem | str") -> bool:
    """Whether this problem is graded by knight/knave enumeration (mirrors
    twin.verifiers.dispatch method selection)."""
    if not isinstance(problem, Problem):
        return False
    vtype = str(problem.verification.get("type", "")).strip().lower()
    if vtype:
        return vtype.startswith("logic")
    return problem.domain.lower() in _LOGIC_DOMAINS


def solver_user(problem: Problem | str) -> str:
    statement = problem.statement if isinstance(problem, Problem) else str(problem)
    if is_code_problem(problem):
        entry = str(problem.verification.get("entry_point", "")).strip()
        entry_line = (
            f"Name the function exactly `{entry}`.\n" if entry else ""
        )
        return (
            f"Solve this programming problem:\n\n{statement}\n\n"
            f"{entry_line}"
            "Reply with one complete ```python code block containing your "
            "solution — it will be run against tests."
        )
    if is_logic_problem(problem):
        return (
            f"Solve this logic puzzle:\n\n{statement}\n\n"
            "Reason through the statements concisely, then end with a final "
            "line naming every character once, in exactly this form:\n"
            "ANSWER: <Name>=knight, <Name>=knave, ..."
        )
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
