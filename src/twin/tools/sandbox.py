"""Subprocess sandbox for executing model-generated Python.

This is the single most security-sensitive component in the project (DESIGN.md
§7): it runs code written by the policy. Isolation, from day one:

  * **Separate process** — a fresh `python -I` (isolated mode: ignores env vars,
    user site-packages, and ``PYTHON*`` settings).
  * **New session / process group** (``start_new_session=True``) so a timeout
    kills the whole group, not just the direct child (a forked grandchild can't
    outlive the call).
  * **Wall-clock timeout** + POSIX ``resource`` limits (CPU seconds, file size,
    open files, no core dumps; address space on Linux only — see ``_ENFORCE_AS``)
    applied in the child before ``exec`` via ``preexec_fn``.
  * **Restricted cwd** — a throwaway temp directory, removed afterwards.
  * **Minimal env** — only ``PATH``; no proxy vars, no secrets.

Network is *not* hard-blocked on macOS in v1 (that needs ``sandbox-exec`` /
seccomp, which is brittle and deprecated). The threat model here is our own
offline self-play model, and nothing in the env hands it credentials or a
proxy. Hardening the socket layer is a tracked follow-up, not a v1 blocker.
"""

import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass

# macOS counts *reserved* virtual address space against RLIMIT_AS, and the
# interpreter reserves gigabytes at startup, so an AS cap there kills python
# before it runs a line. Only enforce the address-space limit on Linux; on
# Darwin we bound memory indirectly via CPU + wall-clock + fsize limits.
_ENFORCE_AS = sys.platform.startswith("linux")

# Conservative defaults; the verifier/tool callers can tighten per-call.
DEFAULT_TIMEOUT_S = 5.0
DEFAULT_CPU_S = 5
DEFAULT_MEM_MB = 512
DEFAULT_FSIZE_MB = 8
DEFAULT_NOFILE = 64


@dataclass
class SandboxResult:
    ok: bool                # returncode == 0 and not timed_out
    returncode: int | None  # None if killed by timeout
    stdout: str
    stderr: str
    timed_out: bool

    @property
    def output(self) -> str:
        """stdout, or stderr when the run failed/produced nothing on stdout."""
        if self.stdout.strip():
            return self.stdout
        return self.stderr


def _limit_resources(cpu_s: int, mem_mb: int, fsize_mb: int, nofile: int):
    """Returned closure runs in the child between fork and exec (POSIX only)."""

    def _apply():
        # NB: the parent passes start_new_session=True, which already calls
        # setsid() in the child — do NOT call it again here (EPERM).
        mem = mem_mb * 1024 * 1024
        fsize = fsize_mb * 1024 * 1024
        _setlimit(resource.RLIMIT_CPU, cpu_s)
        if _ENFORCE_AS:
            _setlimit(resource.RLIMIT_AS, mem)
            _setlimit(resource.RLIMIT_DATA, mem)
        _setlimit(resource.RLIMIT_FSIZE, fsize)
        _setlimit(resource.RLIMIT_NOFILE, nofile)
        _setlimit(resource.RLIMIT_CORE, 0)

    return _apply


def _setlimit(which: int, value: int) -> None:
    """Best-effort soft/hard rlimit; ignore platforms that reject one."""
    try:
        soft, hard = resource.getrlimit(which)
        new_hard = value if hard == resource.RLIM_INFINITY else min(value, hard)
        resource.setrlimit(which, (min(value, new_hard), new_hard))
    except (ValueError, OSError):
        pass


def run_python(
    code: str,
    *,
    stdin: str = "",
    timeout_s: float = DEFAULT_TIMEOUT_S,
    cpu_s: int = DEFAULT_CPU_S,
    mem_mb: int = DEFAULT_MEM_MB,
    fsize_mb: int = DEFAULT_FSIZE_MB,
    nofile: int = DEFAULT_NOFILE,
) -> SandboxResult:
    """Execute ``code`` in an isolated subprocess and capture its output.

    Never raises on user-code failure: a crash, non-zero exit, or timeout is
    reported in the returned :class:`SandboxResult` (``ok=False``). Only truly
    exceptional host conditions (e.g. the interpreter is missing) propagate.
    """
    workdir = tempfile.mkdtemp(prefix="twin-sandbox-")
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=workdir,
        env=env,
        text=True,
        start_new_session=True,
        preexec_fn=_limit_resources(cpu_s, mem_mb, fsize_mb, nofile),
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(input=stdin, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(proc.pid)
        stdout, stderr = proc.communicate()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    rc = None if timed_out else proc.returncode
    return SandboxResult(
        ok=(not timed_out and rc == 0),
        returncode=rc,
        stdout=stdout or "",
        stderr=stderr or "",
        timed_out=timed_out,
    )


def _kill_group(pid: int) -> None:
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
