#!/usr/bin/env python3
"""Plot per-generation TwentyQ v3 solver win rates as a standalone SVG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTROL = ROOT / "tq-runs/q-fullv3-ctrl.jsonl"
DEFAULT_ROTATION = ROOT / "tq-runs/q-fullv3-rot.jsonl"
DEFAULT_OUTPUT = ROOT / "tq-runs/q-fullv3-win-rate-comparison.svg"


def load_rates(path: Path) -> list[float]:
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") == "iteration":
            records.append(record)
    records.sort(key=lambda record: record["iter"])
    return [
        100.0 * record["episodes"]["guessed"] / record["episodes"]["total"]
        for record in records
    ]


def trailing_mean(values: list[float], window: int = 5) -> list[float]:
    return [sum(values[max(0, i - window + 1): i + 1]) / min(window, i + 1)
            for i in range(len(values))]


def points(values: list[float], x, y) -> str:
    return " ".join(f"{x(i):.2f},{y(value):.2f}" for i, value in enumerate(values))


def build_svg(control: list[float], rotation: list[float]) -> str:
    if len(control) != 30 or len(rotation) != 30:
        raise ValueError(
            f"expected 30 iterations per arm, got control={len(control)}, "
            f"rotation={len(rotation)}"
        )

    width, height = 1200, 700
    left, right, top, bottom = 105, 55, 115, 100
    plot_w = width - left - right
    plot_h = height - top - bottom
    ymax = 30.0
    x = lambda i: left + i * plot_w / 29
    y = lambda value: top + plot_h * (1 - value / ymax)
    colors = {"control": "#2563EB", "rotation": "#EA580C"}

    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="title description">',
        '<title id="title">TwentyQ v3 solver win rate by generation</title>',
        '<desc id="description">Control and rotation-on solver win rates across '
        '30 generations, with observed values and five-generation trailing averages.</desc>',
        '<rect width="1200" height="700" fill="#FFFFFF"/>',
        '<style>text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}'
        '.tick{fill:#475569;font-size:14px}.label{fill:#334155;font-size:16px}'
        '.legend{fill:#0F172A;font-size:16px}.title{fill:#0F172A;font-size:28px;font-weight:700}'
        '.subtitle{fill:#64748B;font-size:15px}</style>',
        '<text x="105" y="48" class="title">TwentyQ v3: Solver Win Rate by Generation</text>',
        '<text x="105" y="76" class="subtitle">Markers show observed rates; thick lines show the 5-generation trailing average</text>',
    ]

    for value in range(0, 31, 5):
        yy = y(value)
        out.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{left + plot_w}" y2="{yy:.2f}" '
                   'stroke="#E2E8F0" stroke-width="1"/>')
        out.append(f'<text x="{left - 16}" y="{yy + 5:.2f}" text-anchor="end" '
                   f'class="tick">{value}%</text>')

    for generation in [1, 5, 10, 15, 20, 25, 30]:
        xx = x(generation - 1)
        out.append(f'<line x1="{xx:.2f}" y1="{top + plot_h}" x2="{xx:.2f}" '
                   f'y2="{top + plot_h + 7}" stroke="#94A3B8"/>')
        out.append(f'<text x="{xx:.2f}" y="{top + plot_h + 29}" text-anchor="middle" '
                   f'class="tick">{generation}</text>')

    out.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#94A3B8"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
        f'y2="{top + plot_h}" stroke="#94A3B8"/>',
        f'<text x="{left + plot_w / 2:.2f}" y="{height - 35}" text-anchor="middle" '
        'class="label">Generation</text>',
        f'<text x="30" y="{top + plot_h / 2:.2f}" text-anchor="middle" class="label" '
        'transform="rotate(-90 30 357.5)">Solver win rate</text>',
    ])

    for name, values in (("Control", control), ("Rotation on", rotation)):
        color = colors["control" if name == "Control" else "rotation"]
        out.append(f'<polyline points="{points(values, x, y)}" fill="none" stroke="{color}" '
                   'stroke-width="2" stroke-opacity="0.30" stroke-linejoin="round"/>')
        for i, value in enumerate(values):
            out.append(f'<circle cx="{x(i):.2f}" cy="{y(value):.2f}" r="4" fill="#FFFFFF" '
                       f'stroke="{color}" stroke-width="2" opacity="0.75">'
                       f'<title>{name}, generation {i + 1}: {value:.2f}%</title></circle>')
        smooth = trailing_mean(values)
        out.append(f'<polyline points="{points(smooth, x, y)}" fill="none" stroke="{color}" '
                   'stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>')

    legend_x, legend_y = 790, 48
    for offset, (label, color) in enumerate((("Control", colors["control"]),
                                             ("Rotation on", colors["rotation"]))):
        xx = legend_x + offset * 175
        out.append(f'<line x1="{xx}" y1="{legend_y}" x2="{xx + 36}" y2="{legend_y}" '
                   f'stroke="{color}" stroke-width="5" stroke-linecap="round"/>')
        out.append(f'<text x="{xx + 48}" y="{legend_y + 6}" class="legend">{label}</text>')

    control_total = sum(control) / len(control)
    rotation_total = sum(rotation) / len(rotation)
    out.append(
        f'<text x="{left + plot_w - 10}" y="{top + plot_h - 18}" text-anchor="end" '
        f'class="subtitle">30-generation pooled rate: control {control_total:.2f}% · '
        f'rotation on {rotation_total:.2f}%</text>'
    )
    out.append('</svg>')
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--rotation", type=Path, default=DEFAULT_ROTATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.write_text(build_svg(load_rates(args.control), load_rates(args.rotation)))
    print(args.output)


if __name__ == "__main__":
    main()
