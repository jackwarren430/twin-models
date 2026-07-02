"""Turn a training run's JSONL log into the Sprint-4 study curves (DESIGN.md §11).

Everything here is pure: it consumes the iteration records the trainer logs
(:class:`twin.log.JsonlLogger`) and produces plain Python data structures —
no model, no mlx. That keeps the curve maths unit-testable on synthetic records.

The headline curves:

* **solve-rate vs difficulty** — ``solve_rate_curve`` averages each suite's
  realized ``solve_rates_by_rank`` by difficulty position; ``linearity`` fits a
  line through it (slope / Pearson r / R²) and measures its MSE to the ideal
  easy→hard ramp. A healthy creator yields a clean descending line.
* **oracle / tool usage over time** — ``creator_oracle_calls`` etc. as series.
* **KL over time** — ``creator_kl`` / ``solver_kl`` (drift away from the frozen
  base should stay bounded).
* **adapter drift** — ``adapter_drift_A`` / ``_B`` = L2 distance of each LoRA
  tree from its init snapshot, so we can see how far the policies have moved.

Plotting (``render_plots``) is optional: it uses matplotlib if installed and
otherwise no-ops, so the analysis never hard-depends on a plotting stack. The
terminal report (``report_text``) renders ASCII sparklines with no dependencies.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from twin.log import read_jsonl
from twin.problems.schema import ProblemSuite

# name -> path into an iteration record (nested keys allowed)
SERIES_SPEC: dict[str, tuple[str, ...]] = {
    "iter": ("iter",),
    "creator_reward": ("creator_reward_mean",),
    "solver_reward": ("solver_reward_mean",),
    "solve_rate": ("solve_rate_mean",),
    "parse_ok_rate": ("parse_ok_rate",),
    "r_gradient": ("r_gradient_mean",),
    "creator_kl": ("creator_update", "kl"),
    "solver_kl": ("solver_update", "kl"),
    "creator_grad_norm": ("creator_update", "grad_norm"),
    "solver_grad_norm": ("solver_update", "grad_norm"),
    "creator_oracle_calls": ("creator_oracle_calls",),
    "solver_oracle_calls": ("solver_oracle_calls",),
    "creator_tool_calls": ("creator_tool_calls",),
    "adapter_norm_A": ("adapter_norm", "A"),
    "adapter_norm_B": ("adapter_norm", "B"),
    "adapter_drift_A": ("adapter_drift", "A"),
    "adapter_drift_B": ("adapter_drift", "B"),
}

_BLOCKS = "▁▂▃▄▅▆▇█"


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_iterations(path: str | Path) -> list[dict]:
    """Read a run JSONL and return its ``iteration`` records, sorted by ``iter``
    (the ``meta`` header line and any other record types are skipped)."""
    recs = [r for r in read_jsonl(path) if r.get("type") == "iteration"]
    recs.sort(key=lambda r: r.get("iter", 0))
    return recs


def _get(rec: dict, *path: str, default: Any = None) -> Any:
    cur: Any = rec
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# --------------------------------------------------------------------------- #
# time series
# --------------------------------------------------------------------------- #
def time_series(records: list[dict]) -> dict[str, list]:
    """Pull every metric in ``SERIES_SPEC`` into a parallel list keyed by name.
    Missing keys (older logs, parse-gated iters) become ``None`` placeholders so
    every series stays aligned with ``iter``."""
    out: dict[str, list] = {name: [] for name in SERIES_SPEC}
    for r in records:
        for name, path in SERIES_SPEC.items():
            out[name].append(_get(r, *path))
    return out


def moving_average(xs: list, window: int = 10) -> list[float]:
    """Trailing moving average (window clamps at the series start). ``None``
    entries are skipped from the window; positions with no data so far yield the
    running mean of what's been seen, or ``None`` if nothing yet."""
    out: list[float | None] = []
    acc: list[float] = []
    for x in xs:
        if x is not None:
            acc.append(float(x))
            if len(acc) > window:
                acc.pop(0)
        out.append(sum(acc) / len(acc) if acc else None)
    return out


# --------------------------------------------------------------------------- #
# solve-rate-vs-difficulty curve + linearity
# --------------------------------------------------------------------------- #
def solve_rate_curve(records: list[dict], *, last: int | None = None) -> list[float]:
    """Mean realized solve rate at each difficulty rank, averaged over every
    parsed suite in (optionally the last ``last``) iterations. Index 0 = easiest.
    Variable-length suites are handled rank-by-rank."""
    recs = records[-last:] if last else records
    sums: list[float] = []
    counts: list[int] = []
    for r in recs:
        for suite in r.get("suites", []):
            for i, v in enumerate(suite.get("solve_rates_by_rank") or []):
                if i >= len(sums):
                    sums.append(0.0)
                    counts.append(0)
                sums[i] += float(v)
                counts[i] += 1
    return [sums[i] / counts[i] if counts[i] else 0.0 for i in range(len(sums))]


def linearity(curve: list[float]) -> dict:
    """Least-squares fit of ``curve`` against difficulty rank, plus its fit to
    the ideal easy→hard ramp. Returns ``slope`` (expect < 0: harder ⇒ lower
    solve rate), ``intercept``, ``pearson_r`` (rank↔rate; ~ -1 for a clean
    descending line), ``r2``, and ``ramp_mse`` (MSE to the target curve the
    creator reward uses, so it matches ``RewardEngine``'s gradient term)."""
    n = len(curve)
    if n == 0:
        return {"n": 0, "slope": 0.0, "intercept": 0.0,
                "pearson_r": 0.0, "r2": 0.0, "ramp_mse": 0.0}
    ys = [float(v) for v in curve]
    ramp = [float(t) for t in ProblemSuite.target_curve(n)]
    ramp_mse = sum((y - t) ** 2 for y, t in zip(ys, ramp)) / n
    if n < 2:
        return {"n": n, "slope": 0.0, "intercept": ys[0],
                "pearson_r": 0.0, "r2": 0.0, "ramp_mse": ramp_mse}
    xs = list(range(n))
    mx_ = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx_) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx_) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    intercept = my - slope * mx_
    r = sxy / ((sxx * syy) ** 0.5) if sxx > 0 and syy > 0 else 0.0
    return {"n": n, "slope": slope, "intercept": intercept,
            "pearson_r": r, "r2": r * r, "ramp_mse": ramp_mse}


# --------------------------------------------------------------------------- #
# whole-run summary
# --------------------------------------------------------------------------- #
def _clean(xs: list) -> list[float]:
    return [float(x) for x in xs if x is not None]


def summarize_run(records: list[dict]) -> dict:
    """A compact, JSON-serializable digest: the solve-rate curve + its linearity,
    plus final and mean values of the headline series."""
    ts = time_series(records)

    def last(name: str):
        vals = _clean(ts[name])
        return vals[-1] if vals else None

    def mean(name: str):
        vals = _clean(ts[name])
        return sum(vals) / len(vals) if vals else None

    curve = solve_rate_curve(records)
    return {
        "n_iterations": len(records),
        "first_iter": records[0].get("iter") if records else None,
        "last_iter": records[-1].get("iter") if records else None,
        "solve_rate_curve": curve,
        "linearity": linearity(curve),
        "final": {k: last(k) for k in (
            "creator_reward", "solver_reward", "solve_rate", "parse_ok_rate",
            "creator_kl", "solver_kl", "adapter_drift_A", "adapter_drift_B")},
        "mean": {k: mean(k) for k in (
            "creator_reward", "solver_reward", "solve_rate", "r_gradient",
            "creator_kl", "solver_kl", "creator_oracle_calls",
            "solver_oracle_calls", "creator_tool_calls")},
    }


# --------------------------------------------------------------------------- #
# rendering — CSV / JSON / terminal / (optional) plots
# --------------------------------------------------------------------------- #
def sparkline(xs: list) -> str:
    """Unicode-block sparkline; ``None`` entries render as a space."""
    vals = _clean(xs)
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return _BLOCKS[0] * len(xs)
    out = []
    for x in xs:
        if x is None:
            out.append(" ")
            continue
        idx = int((float(x) - lo) / (hi - lo) * (len(_BLOCKS) - 1) + 0.5)
        out.append(_BLOCKS[idx])
    return "".join(out)


def write_csv(series: dict[str, list], path: str | Path) -> Path:
    """Write the time-series dict as columns to ``path`` (one row per iteration)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(series.keys())
    n = max((len(v) for v in series.values()), default=0)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names)
        for i in range(n):
            w.writerow([series[name][i] if i < len(series[name]) else ""
                        for name in names])
    return path


def write_summary_json(summary: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return path


def _fmt(x: Any) -> str:
    return f"{x:.4g}" if isinstance(x, (int, float)) else str(x)


def report_text(records: list[dict]) -> str:
    """A dependency-free terminal report: sparklines for the headline series and
    the solve-rate curve + its linearity stats."""
    if not records:
        return "(no iterations)"
    ts = time_series(records)
    summary = summarize_run(records)
    lines: list[str] = []
    lines.append(f"run: {len(records)} iterations "
                 f"({summary['first_iter']}..{summary['last_iter']})")
    spark_rows = [
        ("creator_reward", "Rc       "),
        ("solver_reward", "Rs       "),
        ("solve_rate", "solve%   "),
        ("parse_ok_rate", "parseOK  "),
        ("creator_kl", "KL(c)    "),
        ("solver_kl", "KL(s)    "),
        ("creator_oracle_calls", "oracle(c)"),
        ("solver_oracle_calls", "oracle(s)"),
        ("creator_tool_calls", "tools(c) "),
        ("adapter_drift_A", "driftA   "),
        ("adapter_drift_B", "driftB   "),
    ]
    lines.append("")
    for name, label in spark_rows:
        vals = ts[name]
        clean = _clean(vals)
        rng = f"{_fmt(clean[0])}->{_fmt(clean[-1])}" if clean else "-"
        lines.append(f"  {label} {sparkline(vals)}  {rng}")

    curve = summary["solve_rate_curve"]
    lin = summary["linearity"]
    lines.append("")
    lines.append("  solve-rate vs difficulty (easy->hard):")
    lines.append(f"    curve   {sparkline(curve)}  "
                 f"[{', '.join(_fmt(v) for v in curve)}]")
    lines.append(f"    linearity: slope={_fmt(lin['slope'])} "
                 f"pearson_r={_fmt(lin['pearson_r'])} r2={_fmt(lin['r2'])} "
                 f"ramp_mse={_fmt(lin['ramp_mse'])}")
    return "\n".join(lines)


def _pct(num: int, den: int) -> str:
    return f"{(100.0 * num / den):.0f}%" if den else "n/a"


def format_run_summary(records: list[dict]) -> str:
    """Plain-text end-of-run digest for the e2e transcript: the parse-error
    breakdown, the creator's problem-creation accuracy (parse + consistency),
    and the solver's solve accuracy. Pure — consumes iteration records only, so
    it is unit-testable on synthetic records with no model."""
    if not records:
        return "(no iterations)"

    suites_total = parsed = problems_authored = consistent = 0
    parse_errors: dict[str, int] = {}
    s_problems = s_attempts = s_attempts_solved = s_problems_solved = 0
    per_iter: list[str] = []

    for rec in records:
        suites = rec.get("suites", []) or []
        it_parsed = sum(1 for s in suites if s.get("parsed"))
        it_authored = sum(s.get("n_problems", 0) for s in suites if s.get("parsed"))
        it_consistent = sum(s.get("n_consistent", 0) for s in suites if s.get("parsed"))
        suites_total += len(suites)
        parsed += it_parsed
        problems_authored += it_authored
        consistent += it_consistent
        for s in suites:
            if not s.get("parsed"):
                msg = str(s.get("error", "parse failed"))
                parse_errors[msg] = parse_errors.get(msg, 0) + 1

        st = rec.get("solver_stats", {}) or {}
        s_problems += st.get("problems", 0)
        s_attempts += st.get("attempts", 0)
        s_attempts_solved += st.get("attempts_solved", 0)
        s_problems_solved += st.get("problems_solved", 0)

        per_iter.append(
            f"  iter {rec.get('iter')}: {it_parsed}/{len(suites)} suites parsed, "
            f"{it_consistent}/{it_authored} consistent, "
            f"{st.get('problems_solved', 0)}/{st.get('problems', 0)} solved"
        )

    parse_fail = suites_total - parsed
    lines = [
        f"iterations: {len(records)}",
        "",
        "problem creation (creator authors the problems)",
        f"  suites generated      : {suites_total}",
        f"  parsed OK             : {parsed}/{suites_total}  ({_pct(parsed, suites_total)})",
        f"  parse failures        : {parse_fail}/{suites_total}  ({_pct(parse_fail, suites_total)})",
        f"  problems authored     : {problems_authored}",
        f"  internally consistent : {consistent}/{problems_authored}  ({_pct(consistent, problems_authored)})",
    ]
    if parse_errors:
        lines.append("  parse errors:")
        for msg, n in sorted(parse_errors.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {n}x  {msg}")
    lines += [
        "",
        "solver performance (consistent problems only)",
        f"  problems posed        : {s_problems}",
        f"  solve attempts        : {s_attempts}",
        f"  attempts solved       : {s_attempts_solved}/{s_attempts}  ({_pct(s_attempts_solved, s_attempts)})",
        f"  problems solved (>=1) : {s_problems_solved}/{s_problems}  ({_pct(s_problems_solved, s_problems)})",
        "",
        "per iteration",
        *per_iter,
    ]
    return "\n".join(lines)


def render_plots(records: list[dict], outdir: str | Path) -> list[Path]:
    """Write PNG plots to ``outdir`` if matplotlib is importable; otherwise a
    no-op returning ``[]``. Kept import-local so analysis never requires a
    plotting stack."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = time_series(records)
    iters = [r.get("iter") for r in records]
    paths: list[Path] = []

    def _line_plot(fname: str, title: str, ylabel: str, series: list[tuple[str, str]]):
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, label in series:
            ys = ts[name]
            xs = [x for x, y in zip(iters, ys) if y is not None]
            yy = [y for y in ys if y is not None]
            if yy:
                ax.plot(xs, yy, label=label)
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        p = outdir / fname
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)

    _line_plot("rewards.png", "Rewards over time", "reward",
               [("creator_reward", "creator"), ("solver_reward", "solver"),
                ("solve_rate", "solve rate")])
    _line_plot("kl.png", "KL to frozen base", "KL",
               [("creator_kl", "creator"), ("solver_kl", "solver")])
    _line_plot("oracle_tools.png", "Oracle / tool usage", "calls / iter",
               [("creator_oracle_calls", "oracle(c)"),
                ("solver_oracle_calls", "oracle(s)"),
                ("creator_tool_calls", "tools(c)")])
    _line_plot("adapter_drift.png", "Adapter drift from init", "L2 distance",
               [("adapter_drift_A", "A"), ("adapter_drift_B", "B")])

    # solve-rate-vs-difficulty curve (its own axes)
    curve = solve_rate_curve(records)
    if curve:
        fig, ax = plt.subplots(figsize=(7, 4))
        ranks = list(range(len(curve)))
        ax.plot(ranks, curve, marker="o", label="realized")
        ax.plot(ranks, ProblemSuite.target_curve(len(curve)), "--", label="target ramp")
        ax.set_title("Solve rate vs difficulty rank")
        ax.set_xlabel("difficulty rank (easy -> hard)")
        ax.set_ylabel("solve rate")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        fig.tight_layout()
        p = outdir / "solve_rate_curve.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)
    return paths
