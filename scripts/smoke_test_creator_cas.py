"""Creator inline-CAS smoke test (loads the real 6-bit Qwen3-8B base).

Proves the new creator generation aid end-to-end (DESIGN.md §9):

  PART A — inline ReAct splice + mask, controlled:
    1. the model emits a <tool>solve(...)</tool> call; generation pauses
    2. the CAS runs, <obs>RESULT</obs> is spliced in and generation continues
    3. completion_tokens/loss_mask align; injected obs tokens carry mask 0 and
       the policy tokens carry mask 1 (no double-counting of the observation)

  PART B — one real creator-driven iteration through SelfPlayTrainer:
    4. the iteration completes; creator/solver GRPO metrics are finite & KL bounded
    5. the frozen base ('base') adapter is unchanged by the masked update

Run:  conda run -n twin-models python scripts/smoke_test_creator_cas.py
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
from twin.log import JsonlLogger            # noqa: E402
from twin.models import Adapters, TwinBase  # noqa: E402
from twin.tools import ToolHarness, solve   # noqa: E402
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
    cfg.game.n_problems = 2
    cfg.game.creator_group = 2
    cfg.game.solver_attempts = 2
    cfg.gen.creator_max_tokens = 320
    cfg.gen.solver_max_tokens = 160

    print(f"Loading base: {cfg.model.path}")
    base = TwinBase(cfg.model.path)
    adapters = Adapters.from_config(base.model, cfg.lora)
    print(f"  loaded; {adapters.num_params('A'):,} params/adapter.\n")

    # ---- PART A: controlled inline ReAct splice + mask --------------------
    print("== part A: inline tool splice + mask ==")
    adapters.activate("A")

    def runner(segment_text: str):
        h = ToolHarness({"solve": solve}, max_tool_calls=2)
        results = h.run(segment_text)
        if not results:
            return None
        return "\n" + ToolHarness.format_observations(results) + "\n"

    prompt = base.render(
        "Reply with EXACTLY this single line and nothing after it:\n"
        "<tool>solve(6*7)</tool>",
        system="You output exactly what the user asks for, verbatim.",
    )
    res = base.generate_react(
        prompt, tool_runner=runner, max_tokens=64, temp=0.0, top_p=1.0, max_rounds=3
    )
    print(f"  text: {res.text.strip()[:160]!r}")
    check("a tool call fired", res.n_tool_calls >= 1, detail=f"n_tool_calls={res.n_tool_calls}")
    check("loss_mask aligns with completion", len(res.loss_mask) == len(res.completion_tokens))
    check("mask has injected (0) positions", 0 in res.loss_mask)
    check("mask has policy (1) positions", 1 in res.loss_mask)
    check("CAS result spliced into text", "42" in res.text, detail="solve(6*7)=42")

    obs_ids = [t for t, m in zip(res.completion_tokens, res.loss_mask) if m == 0]
    pol_ids = [t for t, m in zip(res.completion_tokens, res.loss_mask) if m == 1]
    obs_text = base.tokenizer.decode(obs_ids) if obs_ids else ""
    pol_text = base.tokenizer.decode(pol_ids) if pol_ids else ""
    check("masked-out tokens decode to the observation", "42" in obs_text,
          detail=obs_text.strip()[:60].replace("\n", " "))
    check("observation not double-counted as policy", "<obs>" not in pol_text)

    # ---- PART B: one real creator-driven iteration -----------------------
    print("\n== part B: a creator iteration through the trainer ==")
    ids = base.tokenizer.encode(base.render("What is 2+2? Answer with a number."))
    adapters.activate("base")
    base_lp0 = base.token_logprobs(ids)
    mx.eval(base_lp0)

    with tempfile.TemporaryDirectory() as td:
        logger = JsonlLogger(Path(td) / "smoke.jsonl", meta={"smoke": "creator_cas"})
        trainer = SelfPlayTrainer(base, adapters, cfg, logger=logger)
        rec = trainer.run_iteration(0)
        logger.close()

    cu, su = rec["creator_update"], rec["solver_update"]
    check("creator update finite", finite(cu["loss"]) and finite(cu["grad_norm"]) and finite(cu["kl"]),
          detail=f"loss={cu['loss']:.3f} gnorm={cu['grad_norm']:.3f} kl={cu['kl']:.4f}")
    check("creator KL >= 0 and bounded", cu["kl"] >= -1e-6 and cu["kl"] < 50.0, detail=f"kl={cu['kl']:.4f}")
    check("solver update finite", finite(su["loss"]) and finite(su["kl"]))
    n_tool = sum(s.get("n_tool_calls", 0) for s in rec.get("suites", []))
    print(f"  creator tool calls this iteration: {n_tool}")

    adapters.activate("base")
    base_lp1 = base.token_logprobs(ids)
    mx.eval(base_lp1)
    md = float(mx.max(mx.abs(base_lp1 - base_lp0)))
    check("base adapter unchanged by masked update", md < 1e-4, detail=f"maxdiff={md:.2e}")

    print()
    if failures:
        print(f"{FAIL}: {failures} invariant(s) failed.")
        sys.exit(1)
    print(f"{PASS}: creator inline-CAS splices, masks, and trains without touching the base.")


if __name__ == "__main__":
    main()
