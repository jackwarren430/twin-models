"""Probe creator NATIVE tool adoption under the current prompts (2026-07-04).

mini-04(a) measured ZERO real creator tool calls across 40 rollouts — the
model simulated the CAS inside <think> ("I can't actually run the function
here"). The prompts were rewritten to a two-phase VERIFY/DELIVER protocol;
this probe answers, with ~15 minutes of compute and no training: do rollouts
under the rewritten prompt actually emit <tool_call>?

Reports per rollout: rank, n_tool_calls, tool_ok, parsed, answer_in_obs,
think_share, completion tokens. DECISION RULE (pre-registered): adoption > 0
on a meaningful fraction (>= 2/6) -> relaunch mini-04b with
game.require_tool_use: true (the gate now has something to select on);
adoption still 0 -> iterate the prompt again before any long run.

DO NOT run while a training run is live. Usage:
    conda run --no-capture-output -n twin-models python -u \
        scripts/probe_tool_adoption.py --config configs/mini4b.yaml
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from twin.config import Config  # noqa: E402
from twin.models import Adapters, TwinBase  # noqa: E402
from twin.problems.schema import ProblemSuite, SuiteParseError, parse_problem  # noqa: E402
from twin.prompts import (  # noqa: E402
    creator_persona,
    creator_problem_user,
    creator_system,
    opponent_of,
)
from twin.think import think_share  # noqa: E402
from twin.tools import tool_schemas  # noqa: E402
from twin.train.loop import SelfPlayTrainer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "mini4b.yaml"))
    ap.add_argument("--ranks", default="0,2,4,4,2,0",
                    help="ranks to probe (rank 4 = hardest of N=5)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    print(f"Loading base: {cfg.model.path}")
    base = TwinBase(cfg.model.path)
    adapters = Adapters.from_config(base.model, cfg.lora)

    # Reuse the REAL trainer generation path (native harness, budgets, seeds)
    # without any training machinery.
    t = SelfPlayTrainer.__new__(SelfPlayTrainer)
    t.cfg = cfg
    t.base = base
    t.adapters = adapters
    t.transcript = None

    n = cfg.game.n_problems
    targets = ProblemSuite.target_curve(n, cfg.rewards.target_hi, cfg.rewards.target_lo)
    sys_prompt = creator_system(
        native_tools=(cfg.tools.protocol == "native"),
        persona=creator_persona("A") if cfg.game.personas else None,
    )
    opp = opponent_of("A") if cfg.game.personas else None

    ranks = [int(x) for x in args.ranks.split(",")]
    header = f"{'rank':>4} {'calls':>5} {'ok':>3} {'parsed':>6} {'in_obs':>6} {'think%':>6} {'tokens':>6}"
    print(header)
    n_with_calls = 0
    for i, rank in enumerate(ranks):
        difficulty = round(rank / (n - 1), 2) if n > 1 else 0.5
        user = creator_problem_user(
            "math", "systems of two equations", rank=rank, n_problems=n,
            difficulty=difficulty, target_rate=targets[rank], opponent=opp,
        )
        adapters.activate("A")
        cgen, harness = t._generate_creator("A", sys_prompt, user)
        parsed = True
        answer = ""
        try:
            problem = parse_problem(cgen.text, default_domain="math")
            answer = (problem.answer or "").strip()
        except SuiteParseError:
            parsed = False
        n_ok = sum(1 for res in harness.calls if res.ok)
        in_obs = bool(answer) and any(
            res.ok and answer in res.output for res in harness.calls)
        if cgen.n_tool_calls > 0:
            n_with_calls += 1
        print(f"{rank:>4} {cgen.n_tool_calls:>5} {n_ok:>3} {str(parsed):>6} "
              f"{str(in_obs):>6} {think_share(cgen.text):>6.2f} "
              f"{len(cgen.completion_tokens):>6}")

    print(f"\nAdoption: {n_with_calls}/{len(ranks)} rollouts made >=1 real tool call")
    print("Decision rule: >=2 -> relaunch mini-04b with require_tool_use: true; "
          "0 -> iterate the prompt again.")


if __name__ == "__main__":
    main()
