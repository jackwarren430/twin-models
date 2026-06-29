"""Code verifier: run candidate code against tests in the sandbox (DESIGN.md §7).

Ground truth = the tests pass under isolated execution. Used for solver scoring
(run the solver's code against the creator's hidden tests) and for creator
consistency (run the creator's reference solution against its own tests).
"""

from twin.tools.sandbox import run_python
from twin.verifiers.result import VerificationResult

_METHOD = "exec"
_MAX_DETAIL = 600


def verify_code(
    code: str,
    tests: str,
    *,
    setup: str = "",
    timeout_s: float = 5.0,
    cpu_s: int = 5,
    mem_mb: int = 512,
) -> VerificationResult:
    """True iff ``code`` + ``tests`` runs to a clean exit in the sandbox.

    ``tests`` is expected to be ``assert``-style checks referencing names the
    candidate defines. A non-zero exit (assertion/exception) or a timeout is a
    failure, with stderr captured in ``detail``."""
    if not (code or "").strip():
        return VerificationResult.fail(_METHOD, "empty candidate code")
    if not (tests or "").strip():
        return VerificationResult.fail(_METHOD, "no tests supplied")

    program = "\n".join(s for s in (setup, code, tests) if s.strip())
    res = run_python(program, timeout_s=timeout_s, cpu_s=cpu_s, mem_mb=mem_mb)

    if res.ok:
        return VerificationResult.ok(_METHOD, "all tests passed")
    if res.timed_out:
        return VerificationResult.fail(_METHOD, f"timeout after {timeout_s}s")
    return VerificationResult.fail(_METHOD, _trim(res.stderr or res.stdout))


def _trim(text: str) -> str:
    text = (text or "").strip()
    return text if len(text) <= _MAX_DETAIL else text[-_MAX_DETAIL:]
