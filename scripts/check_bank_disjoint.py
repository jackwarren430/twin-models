#!/usr/bin/env python3
"""Verify a secret bank shares nothing with the held-out validation set.

This guards the single assumption the headline v8 result rests on. The bank
builder already holds the validation set out, but that is one implementation
protecting the claim that "the solver improved" means something: if a
validation secret reached the bank, improvement and memorisation of the test
would be indistinguishable after the fact, and nothing downstream would flag
it. Cheap to check, unrecoverable to get wrong.

Matching uses ``repeat_matches`` — normalized equality plus edit-distance-1 on
5+ character strings, over both raw and ASCII-folded forms — so near-miss
collisions ("giraff" for "giraffe") are caught, not just exact ones.

Exit 0 if disjoint, 1 if contaminated or unreadable.

    python scripts/check_bank_disjoint.py BANK.json [--holdout PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twin.games.twentyq.schema import repeat_matches  # noqa: E402
from twin.games.twentyq.trainer import load_validation_secret_set  # noqa: E402


def collisions(bank_secrets: list[str], reserved: list[str]) -> list[tuple[str, str]]:
    """Every (bank secret, colliding held-out secret) pair."""
    out: list[tuple[str, str]] = []
    for name in bank_secrets:
        for r in reserved:
            if repeat_matches(name, r):
                out.append((name, r))
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bank", type=Path)
    ap.add_argument("--holdout", nargs="+", type=Path,
                    default=[ROOT / "data/twentyq-validation-v2.json"],
                    help="one or more sets the bank must not overlap")
    args = ap.parse_args()

    try:
        bank = json.loads(args.bank.read_text())["secrets"]
    except (OSError, json.JSONDecodeError, KeyError) as e:
        print(f"UNREADABLE: {args.bank}: {e}", file=sys.stderr)
        return 1

    reserved: list[str] = []
    names: list[str] = []
    for path in args.holdout:
        meta, holdout = load_validation_secret_set(path)
        reserved.extend(s.secret for s in holdout)
        names.append(str(meta.get("name", path.stem)))
    label = " + ".join(names)
    hits = collisions([b["secret"] for b in bank], reserved)

    if hits:
        print(f"CONTAMINATED: {len(hits)} of {len(bank)} bank secrets collide "
              f"with {label}", file=sys.stderr)
        for name, r in hits[:10]:
            print(f"  {name!r} matches held-out {r!r}", file=sys.stderr)
        return 1

    print(f"disjointness OK: {len(bank)} bank secrets, none matching "
          f"{len(reserved)} held out in {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
