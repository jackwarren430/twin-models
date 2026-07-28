#!/usr/bin/env python3
"""Compose validation-v5: every base-calibrated non-pinned secret not in bank-v2.

Why compose rather than generate. v9's instrument has to resolve an effect the
size v8 actually produced (+0.069), and at the paired sd measured under common
random numbers that needs roughly 70-120 secrets. Generation cannot supply
them: validation-v4 drew 2400 raw rollouts and, with 259 names already reserved,
yielded 170 distinct novel candidates of which **88% are 0/8 for the base
model**. The winnable-secret vocabulary in these three categories is exhausted,
and no amount of further sampling changes that.

But the secrets already exist, spread across four files built at different
times. Every one of them satisfies the same criterion — measured by PLAYING the
base policy, kept only if not pinned at 0 or K — so the union is a single
coherent instrument, not a grab bag:

    validation-v3   36   base-calibrated K=16, band [0.0625, 0.9375]
    validation-v4   19   base-calibrated K=8,  band [0.125, 0.875]
    bank-v1 \\ bank-v2   base-calibrated K=8,  band [0.125, 0.875]

bank-v1 is usable as EVALUATION data for v9 specifically, and this is the one
step that needs care. v9's adapters are zero-init, so v9 starts from the base
policy and never trains on bank-v1 — only on bank-v2. The carryover (secrets in
both banks) must be removed, because those ARE in bank-v2. What remains is the
44% of bank-v1 that drifted off the frontier during v8, which is clean for v9.

The resulting set is therefore valid for v9 and NOT for v8, which trained on
bank-v1. That asymmetry is deliberate and must be stated wherever it is used:
v8's numbers stay on validation-v2 (pre-registered) and validation-v3.

    python scripts/compose_validation_v5.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twin.games.twentyq.schema import normalize_guess, repeat_matches  # noqa: E402

SOURCES = [
    ("data/twentyq-validation-v3.json", "v3"),
    ("data/twentyq-validation-v4.json", "v4"),
    ("data/twentyq-bank-v1.json", "b1"),
]
EXCLUDE = "data/twentyq-bank-v2.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data/twentyq-validation-v5.json")
    ap.add_argument("--min-size", type=int, default=70,
                    help="refuse to write a set too small to resolve +0.069")
    args = ap.parse_args()

    excluded = json.loads((ROOT / EXCLUDE).read_text())["secrets"]
    excl_names = [s["secret"] for s in excluded]
    print(f"excluding {len(excl_names)} secrets in {EXCLUDE} (v9 trains on them)")

    rows: list[dict] = []
    seen: set[str] = set()
    by_source: dict[str, int] = defaultdict(int)
    dropped_carryover = 0

    for path, tag in SOURCES:
        payload = json.loads((ROOT / path).read_text())
        for s in payload["secrets"]:
            key = normalize_guess(s["secret"])
            if not key or key in seen:
                continue
            # Edit-distance-1, same matcher the repeat gate uses: a near-miss
            # against a training secret is contamination just as surely as an
            # exact one.
            if any(repeat_matches(s["secret"], e) for e in excl_names):
                dropped_carryover += 1
                continue
            seen.add(key)
            by_source[tag] += 1
            rows.append({
                "secret": s["secret"],
                "category": s["category"],
                "difficulty": s["difficulty"],
                "measured_guess_rate": s.get("measured_guess_rate"),
                "notes": s.get("notes", ""),
                "source": tag,
            })

    # Ids are assigned after merging so they are unique and per-category, which
    # is what the validation loader requires.
    per_cat: dict[str, int] = defaultdict(int)
    for r in rows:
        cat = r["category"]
        r["secret_id"] = (f"twentyq-validation-v5-{cat.split()[0]}-"
                          f"{per_cat[cat]:03d}")
        per_cat[cat] += 1

    print(f"  dropped {dropped_carryover} as present in bank-v2")
    for tag, n in by_source.items():
        print(f"  from {tag}: {n}")
    print(f"  total {len(rows)} secrets")
    for cat, n in sorted(per_cat.items()):
        print(f"    {cat:<20} {n}")

    if len(rows) < args.min_size:
        print(f"\nTOO SMALL: {len(rows)} < {args.min_size}", file=sys.stderr)
        return 1

    sd = 0.2746  # paired per-secret sd measured under CRN on validation-v3
    mde = 2.8016 * sd / len(rows) ** 0.5
    print(f"  MDE at the CRN-measured sd {sd}: {mde:+.4f}  "
          f"(v8's transfer was +0.069)")

    (ROOT / args.out).write_text(json.dumps({
        "name": args.out.stem,
        "version": 5,
        "description": (
            "Composed evaluation set: the union of every base-calibrated, "
            "non-pinned secret in validation-v3, validation-v4 and bank-v1, "
            "minus everything present in bank-v2. All entries were selected by "
            "PLAYING the base policy and keeping only secrets it wins "
            "sometimes and loses sometimes, so the union is one criterion "
            "applied three times rather than a mixture. Built because "
            "generation is exhausted: validation-v4 drew 2400 rollouts and 88% "
            "of the distinct novel candidates were 0/8 for the base model. "
            "VALID FOR v9 ONLY — v9 starts from the base policy and trains "
            "solely on bank-v2, whereas v8 trained on bank-v1, so this set is "
            "contaminated for v8 and v8's numbers stay on validation-v2 and "
            "validation-v3."),
        "source_model": json.loads(
            (ROOT / "data/twentyq-validation-v3.json").read_text())
            .get("source_model"),
        "solver_adapter": None,
        "composed_from": {tag: n for tag, n in by_source.items()},
        "excludes": EXCLUDE,
        "secrets": rows,
    }, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
