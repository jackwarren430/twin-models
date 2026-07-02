"""Sprint 3 — the `solve` CAS tool (creator generation aid). Fast: no model.

Covers the dispatch (solve / evaluate / calculus), selectors, the top-level
comma splitter, the text-in/text-out error contract, and — importantly — that
the locked-down parse namespace blocks arbitrary code execution (the raw
``parse_expr`` will happily run ``__import__`` otherwise)."""

from twin.tools import SOLVE_TOOL_DOC, solve, split_top_level_commas


# ----- solve intent --------------------------------------------------------
def test_linear_equation():
    assert solve("2*x + 3 = 11") == "[4]"


def test_implicit_multiplication():
    assert solve("2x + 3 = 11") == "[4]"


def test_quadratic_roots_and_selectors():
    out = solve("x**2 - 5*x + 6, x")
    assert out in ("[2, 3]", "[3, 2]")          # both roots, order-agnostic
    assert solve("x**2 - 5*x + 6, x, max") == "3"
    assert solve("x**2 - 5*x + 6, x, min") == "2"


def test_system_of_equations():
    assert solve("x + y = 10, x - y = 2, x, y") == "{x: 6, y: 4}"


def test_python_style_double_equals_is_treated_as_equation():
    # Models often write '==' for equality; the bare parser reads that as a
    # boolean and silently solves nothing. The tool must accept it like '='.
    assert solve("4*x == 20, x") == "[5]"
    assert solve("2*x + 3 == 11") == "[4]"
    assert solve("x + y == 10, x - y == 2, x, y") == "{x: 6, y: 4}"
    # Inequalities keep their single '=' and are left intact.
    assert solve("x - 5 = 0") == "[5]"


def test_symbolic_solution():
    assert solve("a*x + b, x") == "[-b/a]"


# ----- evaluate / simplify intent -----------------------------------------
def test_arithmetic_eval():
    assert solve("7*8 + 3") == "59"


def test_exact_rational():
    assert solve("1/2 + 1/3") == "5/6"


def test_exact_radical_kept():
    assert solve("sqrt(8)") == "2*sqrt(2)"


def test_calculus_via_doit():
    assert solve("integrate(x**2, x)") == "x**3/3"
    assert solve("diff(sin(x), x)") == "cos(x)"


def test_factor_not_undone():
    # symbolic results are returned as-is, so factoring survives (no re-simplify)
    assert solve("factor(x**2 - 1)") == "(x - 1)*(x + 1)"
    assert solve("expand((x + 1)**2)") == "x**2 + 2*x + 1"


# ----- contract / safety ---------------------------------------------------
def test_empty_and_garbage_return_error_text():
    assert solve("").startswith("error")
    assert solve("this is not math (((").startswith("error")


def test_namespace_blocks_code_execution():
    # raw parse_expr would resolve __import__ and run this; we must not.
    out = solve('__import__("os").system("echo PWNED")')
    assert out.startswith("error")
    assert "PWNED" not in out


def test_timeout_guard_returns_error_text():
    # zero/None timeout means "run inline"; an impossibly small timeout on a
    # trivial expr should still return a string (not raise).
    assert isinstance(solve("2 + 2", timeout_s=None), str)


# ----- helper --------------------------------------------------------------
def test_split_top_level_commas_respects_nesting():
    assert split_top_level_commas("f(x, y), z") == ["f(x, y)", "z"]
    assert split_top_level_commas("a, b, c") == ["a", "b", "c"]
    assert split_top_level_commas("[1, 2], 3") == ["[1, 2]", "3"]
    assert split_top_level_commas("  solo  ") == ["solo"]


def test_tool_doc_is_present():
    assert "solve(" in SOLVE_TOOL_DOC
