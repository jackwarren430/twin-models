"""Build a grounded theme-weights file from a held-out benchmark result
(Sprint 9 — SPICE-style target grounding, RELATED_WORK.md §3).

Reads a ``runs/bench-*.json`` produced by scripts/benchmark.py, computes the
per-subcategory failure profile of one adapter, and writes
``{domain: [[theme, weight], ...]}`` for ``game.theme_weights``. Weight =
``floor + failures`` per subcategory, so every observed topic stays sampleable
(floor) while topics the model actually fails lean the curriculum toward the
target distribution — the literature's fix for self-play curricula drifting
toward what the creator finds easy to pose rather than what the solver needs.

Bench categories map to training domains (math -> math, coding -> coding);
knowledge/reasoning are skipped until those domains are trainable.

    python scripts/build_grounded_themes.py \
        --bench runs/bench-hard-base-nothink.json --adapter base \
        --out data/themes-grounded.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# bench category -> training domain (game.domains vocabulary)
_CATEGORY_DOMAIN = {"math": "math", "coding": "coding"}


def build_weights(
    result: dict, adapter: str, *, floor: float = 1.0
) -> dict[str, list[list]]:
    block = result["adapters"][adapter]
    fails: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    seen: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in block["items"]:
        domain = _CATEGORY_DOMAIN.get(item.get("category", ""))
        sub = (item.get("subcategory") or "").strip()
        if not domain or not sub:
            continue
        theme = sub.replace("&", "and").lower()
        seen[domain][theme] += 1
        if not item.get("correct"):
            fails[domain][theme] += 1
    return {
        domain: sorted(
            ([theme, floor + fails[domain][theme]] for theme in themes),
            key=lambda tw: -tw[1],
        )
        for domain, themes in seen.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="runs/bench-*.json result file")
    ap.add_argument("--adapter", default="base", help="adapter block to profile")
    ap.add_argument("--out", default=str(ROOT / "data" / "themes-grounded.json"))
    ap.add_argument("--floor", type=float, default=1.0,
                    help="baseline weight so no observed topic drops to zero")
    args = ap.parse_args()

    result = json.loads(Path(args.bench).read_text())
    weights = build_weights(result, args.adapter, floor=args.floor)
    Path(args.out).write_text(json.dumps(weights, indent=2) + "\n")
    print(f"Wrote {args.out}")
    for domain, entries in weights.items():
        top = ", ".join(f"{t} ({w:g})" for t, w in entries[:5])
        print(f"  {domain}: {top}")


if __name__ == "__main__":
    main()
