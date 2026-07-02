"""Run the benchmark: prompt the model per item, grade, collect per-item results.

The core, :func:`run_items`, is model-agnostic — it takes a ``solve_fn(item) ->
str`` and a grader. That seam keeps the loop unit-testable with a fake solver
(no weights) while the real run injects :func:`make_model_solver`, which wires in
``TwinBase`` + ``Adapters`` exactly like the trainer's solver path.

Prompting is keyed on ``item.vtype`` (not the category) so each grading method
gets an answer in the form its extractor expects:

* ``math_numeric`` -> the trainer's own ``SOLVER_SYSTEM`` / ``solver_user``
* ``exact``        -> a short-answer system asking for an ``ANSWER:`` line
* ``mcq``          -> choices rendered ``A) ...``; answer with the letter
* ``code``         -> "return a single ```python code block"
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from twin.bench.dataset import BenchItem
from twin.bench.grade import grade as default_grade
from twin.prompts import SOLVER_SYSTEM, solver_user
from twin.verifiers.result import VerificationResult

_SHORT_ANSWER_SYSTEM = (
    "You are answering a general-knowledge or reasoning question. Think briefly if "
    "needed, then give the answer on its own last line in exactly the form "
    "'ANSWER: <value>'. Keep <value> as short as possible — a single word, name, "
    "number, or 'yes'/'no' — with no extra words, units, or punctuation."
)

_MCQ_SYSTEM = (
    "You are answering a multiple-choice question. Consider the options, then give "
    "your choice on its own last line in exactly the form 'ANSWER: <letter>', where "
    "<letter> is one of the option letters (A, B, C, ...). Output only the letter, "
    "not the option text."
)

_CODE_SYSTEM = (
    "You are a careful Python programmer. Write a correct, self-contained solution. "
    "Return ONLY a single Python code block fenced with ```python ... ``` that "
    "defines the requested function. Do not include tests, example calls, prose, or "
    "input() — only the function definition (plus any imports it needs)."
)


def _render_choices(choices: list[str]) -> str:
    return "\n".join(f"{chr(ord('A') + i)}) {c}" for i, c in enumerate(choices))


def build_prompt(item: BenchItem) -> tuple[str, str]:
    """Return the ``(system, user)`` pair for ``item``, keyed on its grading type."""
    vtype = item.vtype
    if vtype == "math_numeric":
        return SOLVER_SYSTEM, solver_user(item.prompt)
    if vtype == "code":
        return _CODE_SYSTEM, (
            f"{item.prompt}\n\n"
            "Return only a single ```python code block defining the function."
        )
    if vtype == "mcq":
        user = (
            f"{item.prompt}\n\n{_render_choices(item.choices)}\n\n"
            "Answer with the letter of the best option, ending with a final line "
            "'ANSWER: <letter>'."
        )
        return _MCQ_SYSTEM, user
    # exact (and any short-answer fallback)
    return _SHORT_ANSWER_SYSTEM, (
        f"{item.prompt}\n\nEnd with a final line in exactly this form:\nANSWER: <value>"
    )


@dataclass
class ItemResult:
    id: str
    category: str
    subcategory: str
    vtype: str
    correct: bool
    score: float
    method: str
    detail: str
    response: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def run_items(
    items: list[BenchItem],
    solve_fn,
    *,
    grade_fn=default_grade,
    on_result=None,
    keep_response: bool = True,
) -> list[ItemResult]:
    """Solve and grade every item, returning one :class:`ItemResult` each.

    Neither a generation error nor a grader error aborts the run — both are
    captured and the item is scored incorrect, so one bad item can't sink a whole
    benchmark pass. ``on_result`` (if given) is called after each item for live
    progress; ``keep_response`` stores the raw model text for later inspection."""
    results: list[ItemResult] = []
    for item in items:
        response, error = "", ""
        try:
            response = solve_fn(item) or ""
        except Exception as e:  # noqa: BLE001 - never crash the suite on one item
            error = f"generation error: {type(e).__name__}: {e}"[:300]
        if error:
            vr = VerificationResult.fail("error", error)
        else:
            try:
                vr = grade_fn(item, response)
            except Exception as e:  # noqa: BLE001 - verifier boundary
                vr = VerificationResult.fail("error", f"grading error: {e}"[:300])
                error = vr.detail
        res = ItemResult(
            id=item.id,
            category=item.category,
            subcategory=item.subcategory,
            vtype=item.vtype,
            correct=bool(vr.correct),
            score=float(vr.score),
            method=vr.method,
            detail=vr.detail,
            response=response if keep_response else "",
            error=error,
        )
        results.append(res)
        if on_result is not None:
            on_result(res)
    return results


def make_model_solver(
    base,
    adapters,
    adapter_name: str,
    *,
    max_tokens: int = 512,
    temp: float = 0.0,
    top_p: float = 0.95,
    enable_thinking: bool = False,
    seed: int | None = None,
):
    """Build a ``solve_fn`` that runs items through the real model under
    ``adapter_name`` ('base' | 'A' | 'B'). Greedy (``temp=0``) by default for a
    reproducible benchmark. Mirrors the trainer's solver call path."""

    def solve(item: BenchItem) -> str:
        system, user = build_prompt(item)
        adapters.activate(adapter_name)
        prompt = base.render(user, system=system, enable_thinking=enable_thinking)
        return base.generate(
            prompt, max_tokens=max_tokens, temp=temp, top_p=top_p, seed=seed
        ).text

    return solve
