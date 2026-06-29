"""Sandbox execution tests. Spawn real subprocesses but stay fast (<~2s total).

These are the security-critical paths, so we assert isolation behaviour
(timeouts, crash capture, env minimization) explicitly."""

from twin.tools.sandbox import run_python


def test_basic_stdout():
    res = run_python("print('hello')")
    assert res.ok
    assert res.returncode == 0
    assert res.stdout.strip() == "hello"
    assert not res.timed_out


def test_nonzero_exit_on_exception():
    res = run_python("raise ValueError('boom')")
    assert not res.ok
    assert res.returncode != 0
    assert "ValueError" in res.stderr


def test_assertion_failure_is_failure():
    res = run_python("assert 1 == 2")
    assert not res.ok


def test_timeout_is_caught_not_raised():
    res = run_python("while True:\n    pass", timeout_s=1.0)
    assert res.timed_out
    assert not res.ok
    assert res.returncode is None


def test_stdin_is_piped():
    res = run_python("import sys; print(sys.stdin.read().upper())", stdin="abc")
    assert res.ok
    assert res.stdout.strip() == "ABC"


def test_env_is_minimized():
    # Only PATH (+ encoding) is forwarded; arbitrary host vars are absent.
    res = run_python("import os; print(os.environ.get('HOME', 'MISSING'))")
    assert res.ok
    assert res.stdout.strip() == "MISSING"
