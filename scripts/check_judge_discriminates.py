#!/usr/bin/env python3
"""Confirm the validity judge actually rejects things, not just accepts them.

The v8 bank build vetted 118/118 candidates VALID with zero rejections and zero
terse retries. That is plausible — the fixed easy-target creator prompt emits
Lion/Elephant/Dog-class entities, which genuinely are real, unambiguous,
singular members of their category, and v7's rejections clustered on corrupted
non-words that the fixed parser no longer produces. But a sound judge and a
rubber-stamping judge produce an identical 100% pass rate, so "it passed" is
not evidence until the negatives are checked.

This drives the REAL judge (same model, same prompt, same terse-retry fallback
and fail_closed mode as the builder) over cases that must be rejected and cases
that must be accepted, and reports both error rates.

    python scripts/check_judge_discriminates.py [--judge-model Qwen/Qwen3-8B]

Exit 0 if the judge separates them, 1 if it rubber-stamps or over-rejects.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twin.backends import get_backend  # noqa: E402
from twin.config import Config, ModelConfig  # noqa: E402
from twin.games.twentyq.judge import judge_secret_validity  # noqa: E402
from twin.games.twentyq.prompts import JUDGE_SYSTEM  # noqa: E402
from twin.games.twentyq.schema import Secret  # noqa: E402

# (secret, category, should_be_valid, why)
CASES: list[tuple[str, str, bool, str]] = [
    # --- must be REJECTED ---
    ("Okoupee", "animal", False, "non-word; the exact class v7 passed through fail_open"),
    ("Flimberwocky", "animal", False, "invented non-entity"),
    ("water", "food", False, "category mismatch — the borderline case the creator prompt names"),
    ("happiness", "food", False, "vague abstraction, not an instance"),
    ("mammals", "animal", False, "multi-entity class, not a singular instance"),
    ("a dog or a cat", "animal", False, "multi-entity answer"),
    ("Pikachu", "animal", False, "fictional-only"),
    ("Xenoturbella churro", "animal", False, "too obscure to be guessable"),
    # --- must be ACCEPTED ---
    ("Lion", "animal", True, "the bank's own top-scoring entry"),
    ("Elephant", "animal", True, "familiar, unambiguous"),
    ("Broccoli", "food", True, "familiar, unambiguous"),
    ("Toilet", "household object", True, "familiar, unambiguous"),
    ("Penguin", "animal", True, "familiar, unambiguous"),
    ("Bread", "food", True, "familiar, unambiguous"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=ROOT / "configs/twentyq-v8-bank-solver.yaml")
    ap.add_argument("--judge-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--mode", default="fail_closed")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    backend = get_backend(cfg.compute.backend)
    print(f"loading judge: {args.judge_model}", flush=True)
    judge = backend.load_base(
        ModelConfig(path=args.judge_model, enable_thinking=False), cfg.compute)

    n_terse = 0

    def judge_once(question: str, *, terse: bool) -> str:
        if terse:
            question += ("\n\nReply with ONLY the verdict line and nothing "
                         "else. Do not explain.")
        prompt = judge.render(question, system=JUDGE_SYSTEM,
                              enable_thinking=False)
        return judge.generate(prompt, max_tokens=384, temp=0.0, top_p=1.0,
                              suppress_thinking=True).text

    def oracle(question: str) -> str:
        nonlocal n_terse
        reply = judge_once(question, terse=False)
        if "VERDICT" in (reply or "").upper():
            return reply
        n_terse += 1
        return judge_once(question, terse=True)

    false_accepts: list[str] = []
    false_rejects: list[str] = []
    print(f"\n{'secret':<24} {'category':<18} {'want':>6} {'got':>6}   note")
    for name, category, want_valid, why in CASES:
        got = judge_secret_validity(
            Secret(secret=name, category=category, difficulty=0.5),
            oracle, mode=args.mode).correct
        flag = "" if got == want_valid else "   <-- MISMATCH"
        print(f"{name:<24} {category:<18} "
              f"{'VALID' if want_valid else 'INVALID':>6} "
              f"{'VALID' if got else 'INVALID':>6}{flag}   {why}")
        if got and not want_valid:
            false_accepts.append(name)
        if want_valid and not got:
            false_rejects.append(name)

    n_bad = sum(1 for c in CASES if not c[2])
    n_good = len(CASES) - n_bad
    print(f"\nfalse accepts {len(false_accepts)}/{n_bad}  "
          f"false rejects {len(false_rejects)}/{n_good}  "
          f"terse retries {n_terse}")

    if len(false_accepts) == n_bad:
        print("\nRUBBER-STAMPING: the judge accepted every invalid secret. The "
              "118/118 vet rate carries no information; do not describe the "
              "bank as vetted.", file=sys.stderr)
        return 1
    if false_accepts or false_rejects:
        print(f"\nPARTIAL: accepts={false_accepts} rejects={false_rejects}. The "
              f"judge discriminates but is not clean; the band selection "
              f"(0/8 secrets fall out) remains the effective backstop.",
              file=sys.stderr)
        return 1
    print("\nOK: the judge separates valid from invalid on every case.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
