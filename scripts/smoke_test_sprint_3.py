"""Sprint 3 end-to-end smoke test (loads the real 6-bit Qwen3-8B base).

Runs the full self-play loop for a couple of iterations on a shrunk tiny config
and checks the RL machinery holds together (DESIGN.md §11 Sprint-3 deliverable):

  1. an iteration completes: creator -> consistency -> solver -> rewards -> two
     GRPO updates, with a JSONL record produced
  2. all reward / loss / grad-norm / KL metrics are finite
  3. KL to the frozen base stays bounded (not exploding) after the updates
  4. the base ('base') adapter still reproduces base logits exactly (we only
     mutated A/B) — the Sprint-1 invariant survives a training step

Run:  conda run -n twin-models python scripts/smoke_test_sprint_3.py
Exits non-zero on any failed invariant. Expect a few minutes (it generates).
"""

import math
import sys
import tempfile
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twin.config import Config              # noqa: E402
from twin.log import JsonlLogger, read_jsonl  # noqa: E402
from twin.models import Adapters, TwinBase  # noqa: E402
from twin.train import SelfPlayTrainer      # noqa: E402

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
failures = 0


def check(name: str, ok: bool, detail: str = ""):
    global failures
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures += 1


def finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def main():
    cfg = Config.from_yaml(ROOT / "configs" / "tiny.yaml")
    # Shrink further so the smoke is quick.
    cfg.game.n_problems = 2
    cfg.game.creator_group = 2
    cfg.game.solver_attempts = 2
    cfg.gen.creator_max_tokens = 320
    cfg.gen.solver_max_tokens = 160
    cfg.gen.oracle_max_tokens = 128

    print(f"Loading base: {cfg.model.path}")
    base = TwinBase(cfg.model.path)
    adapters = Adapters.from_config(base.model, cfg.lora)
    print(f"  loaded; {adapters.num_params('A'):,} params/adapter.\n")

    # Baseline base-adapter logits, to confirm training doesn't touch the base.
    ids = base.tokenizer.encode(base.render("What is 2+2? Answer with a number."))
    adapters.activate("base")
    base_lp0 = base.token_logprobs(ids)
    mx.eval(base_lp0)

    with tempfile.TemporaryDirectory() as td:
        logger = JsonlLogger(Path(td) / "smoke.jsonl", meta={"smoke": True})
        trainer = SelfPlayTrainer(base, adapters, cfg, logger=logger)

        print("== iteration 0 ==")
        rec0 = trainer.run_iteration(0)
        print("== iteration 1 ==")
        rec1 = trainer.run_iteration(1)
        logger.close()
        records = read_jsonl(Path(td) / "smoke.jsonl")

    print("\n== checks ==")
    check("two iteration records logged", len([r for r in records if r.get("type") == "iteration"]) == 2)

    for tag, rec in (("iter0", rec0), ("iter1", rec1)):
        cu, su = rec["creator_update"], rec["solver_update"]
        check(f"{tag}: creator reward finite", finite(rec["creator_reward_mean"]),
              detail=f"{rec['creator_reward_mean']}")
        check(f"{tag}: solver reward finite", finite(rec["solver_reward_mean"]),
              detail=f"{rec['solver_reward_mean']}")
        for who, m in (("creator", cu), ("solver", su)):
            check(f"{tag}: {who} loss/grad/kl finite",
                  finite(m["loss"]) and finite(m["grad_norm"]) and finite(m["kl"]),
                  detail=f"loss={m['loss']:.3f} gnorm={m['grad_norm']:.3f} kl={m['kl']:.4f}")
            check(f"{tag}: {who} KL >= 0 and bounded", m["kl"] >= -1e-6 and m["kl"] < 50.0,
                  detail=f"kl={m['kl']:.4f}")

    # Base invariant: zeroed adapter still == base after training A/B.
    adapters.activate("base")
    base_lp1 = base.token_logprobs(ids)
    mx.eval(base_lp1)
    md = float(mx.max(mx.abs(base_lp1 - base_lp0)))
    check("base adapter unchanged by training", md < 1e-4, detail=f"maxdiff={md:.2e}")

    print()
    if failures:
        print(f"{FAIL}: {failures} invariant(s) failed.")
        sys.exit(1)
    print(f"{PASS}: Sprint-3 loop runs end-to-end; rewards finite, KL bounded.")


if __name__ == "__main__":
    main()
