#!/usr/bin/env python
"""Isolated probe for the 21-questions ensemble log-prob reward.

NOT wired into training — this is a scratch harness to sanity-check that the
reward behaves as expected: as the Q/A history pins down the secret, the
ensemble log-prob of the correct answer should RISE (toward 0), and a wrong /
under-determined answer should score lower.

Edit the CASES block below and re-run.  Nothing else needs changing.

    .venv/bin/python scripts/twentyq_ensemble_probe.py                # all 5 models
    .venv/bin/python scripts/twentyq_ensemble_probe.py qwen3-4b llama-3.2-3b   # subset
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from twin.games.twentyq.ensemble_reward import (  # noqa: E402
    DEFAULT_INSTRUCTION, ENSEMBLE_MODELS, EnsembleReward,
)

# ---------------------------------------------------------------------------
# EDIT ME.  A "user" string is the Q/A history + guess instruction the frozen
# models condition on; "answer" is the secret string we score the log-prob of.
# Group related cases so you can watch the score move as history sharpens.
# ---------------------------------------------------------------------------
SYSTEM = "You are playing the game 21 questions. Deduce the secret entity."

# Reusable histories at increasing levels of information about "elephant".
H_EMPTY = "No questions have been asked yet."
H_WEAK = (
    "Q: Is it a living thing? A: YES\n"
    "Q: Is it an animal? A: YES"
)
H_STRONG = (
    "Q: Is it a living thing? A: YES\n"
    "Q: Is it an animal? A: YES\n"
    "Q: Is it a mammal? A: YES\n"
    "Q: Is it larger than a car? A: YES"
)

H_VERY_STRONG = (
    "Q: Is it a living thing? A: YES\n"
    "Q: Is it an animal? A: YES\n"
    "Q: Is it a mammal? A: YES\n"
    "Q: Is it larger than a car? A: YES\n"
    "Q: Does it live in Africa or Asia? A: YES\n"
    "Q: Does it have a trunk? A: YES"
)


def hist(h: str) -> str:
    return f"{DEFAULT_INSTRUCTION}\n\nHistory:\n{h}"


# (label, user_string, answer)
CASES: list[tuple[str, str, str]] = [
    # Fixed answer, sharpening history — score should climb.
    ("empty  -> elephant", hist(H_EMPTY), "elephant"),
    ("weak   -> elephant", hist(H_WEAK), "elephant"),
    ("strong -> elephant", hist(H_STRONG), "elephant"),
    # Fixed (strong) history, different answers — correct should win.
    ("strong -> mouse", hist(H_STRONG), "mouse"),
    ("strong -> car", hist(H_STRONG), "car"),
    ("strong -> banana", hist(H_STRONG), "banana"),
]
# ---------------------------------------------------------------------------


def main() -> None:
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or None
    if names:
        unknown = [n for n in names if n not in ENSEMBLE_MODELS]
        if unknown:
            sys.exit(f"unknown model(s) {unknown}; known: {list(ENSEMBLE_MODELS)}")

    ensemble = EnsembleReward.load(names)

    results = [(label, answer, user, ensemble.score(user, answer, system=SYSTEM))
               for label, user, answer in CASES]

    log_path = Path(__file__).resolve().parent.parent / "runs" / "ensemble_probe.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        for label, answer, user, res in results:
            _write_case(f, label, answer, SYSTEM, user, res)
        f.write("\n(Higher / closer to 0 log-prob = the ensemble finds the "
                "answer more likely.)\n")

    # brief stdout confirmation + one-line-per-case recap
    print(f"\nwrote {log_path}\n")
    print(f"{'case':<22}{'ENSEMBLE logp':>14}{'exp':>10}")
    print("-" * 46)
    for label, answer, user, res in results:
        print(f"{label:<22}{res.score:>+14.4f}{math.exp(res.score):>10.4f}")


def _write_case(f, label, answer, system, user, res) -> None:
    """Emit one case block: every model's full rendered string + per-token
    probabilities + that model's combined normalized probability, then the
    final combined score across models."""
    f.write("\n" + "#" * 100 + "\n")
    f.write(f"CASE: {label}\n")
    f.write(f"answer: {answer!r}\n")
    f.write(f"system: {system}\n")
    f.write(f"user:\n{user}\n")
    f.write("#" * 100 + "\n")

    for m in res.members:
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"MODEL: {m.name}\n")
        f.write("=" * 100 + "\n")
        f.write("FULL RENDERED STRING (special tokens included):\n")
        f.write("-" * 100 + "\n")
        f.write(m.full_text + "\n")
        f.write("-" * 100 + "\n")
        f.write(f"PER-TOKEN PROBABILITIES  (answer span = {m.n_tokens} token(s), "
                f"each scored from the previous position's logits):\n")
        f.write(f"  {'idx':>3}  {'token':<24}{'log P':>12}{'P':>12}\n")
        for i, (tok, lp) in enumerate(zip(m.answer_tokens, m.token_logprobs)):
            f.write(f"  {i:>3}  {tok!r:<24}{lp:>12.4f}{math.exp(lp):>12.6f}\n")
        f.write("COMBINED NORMALIZED PROBABILITY (this model):\n")
        f.write(f"  mean log P = {m.logprob:+.4f}   |   "
                f"exp(mean log P) = {m.prob:.6f}   |   "
                f"perplexity = {m.perplexity:.3f}\n")

    f.write("\n" + "*" * 100 + "\n")
    f.write(f"FINAL COMBINED SCORE  (mean over {len(res.members)} models):\n")
    f.write(f"  score (mean of model mean-log-P) = {res.score:+.4f}   |   "
            f"exp = {math.exp(res.score):.6f}\n")
    f.write("*" * 100 + "\n")


if __name__ == "__main__":
    main()
