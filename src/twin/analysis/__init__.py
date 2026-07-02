"""Run analysis (Sprint 4): curves and summaries from a training run's JSONL."""

from twin.analysis.curves import (
    format_run_summary,
    linearity,
    load_iterations,
    moving_average,
    render_plots,
    report_text,
    solve_rate_curve,
    sparkline,
    summarize_run,
    time_series,
    write_csv,
    write_summary_json,
)

__all__ = [
    "load_iterations",
    "time_series",
    "moving_average",
    "solve_rate_curve",
    "linearity",
    "summarize_run",
    "format_run_summary",
    "sparkline",
    "report_text",
    "write_csv",
    "write_summary_json",
    "render_plots",
]
