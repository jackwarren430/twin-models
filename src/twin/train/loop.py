"""SelfPlayTrainer: one self-play RLVR iteration end-to-end (DESIGN.md §5).

Wires together everything from Sprints 1-2 plus the Sprint-3 GRPO core:

    creator rollout (G_c suites)
      -> parse / parse-gate
      -> consistency check on the creator's own solutions  (verifier-first)
      -> solver rollout (K attempts) on each *consistent* problem
      -> realized solve-rate curve
      -> creator & solver rewards (RewardEngine)
      -> group-relative advantages
      -> two sequential GRPO updates (solver adapter, then creator adapter),
         each with a KL penalty toward the frozen base (zeroed adapter)
      -> JSONL log; periodic checkpoint.

Everything is sequential (generate -> score -> update A -> update B), which keeps
peak memory low (only one adapter trains at a time). The combinatorial cost knobs
are G_c (``creator_group``), N (``n_problems``), K (``solver_attempts``) — keep
them tiny until the loop is proven (configs/tiny.yaml).

v1 rollouts are single-turn: we score the exact completion the policy emitted, so
there are no injected tool-observation tokens to mask. The oracle tax still bites
via :func:`count_oracle_calls` (counts the oracle calls the policy *wrote*).
Inline ReAct tool execution with <obs> masking is a documented later extension.
"""

import random
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim

from twin.config import Config
from twin.log import JsonlLogger, TranscriptLogger
from twin.models import Adapters, TwinBase
from twin.problems.schema import SuiteParseError, parse_suite
from twin.prompts import (
    CREATOR_SYSTEM,
    JUDGE_SYSTEM,
    SOLVER_SYSTEM,
    creator_user,
    pick_theme,
    solver_user,
)
from twin.rewards import RewardEngine, solve_rate
from twin.rl import Trajectory, all_zero_advantages, group_advantages, grpo_update
from twin.roles import RoleManager
from twin.tools import ToolHarness, calc, solve
from twin.train.extract import count_oracle_calls, extract_final_answer
from twin.verifiers import check_consistency, verify_answer


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


class SelfPlayTrainer:
    def __init__(
        self,
        base: TwinBase,
        adapters: Adapters,
        cfg: Config,
        *,
        logger: JsonlLogger | None = None,
        transcript: TranscriptLogger | None = None,
        seed: int | None = None,
    ):
        self.base = base
        self.adapters = adapters
        self.cfg = cfg
        self.engine = RewardEngine(cfg.rewards)
        self.roles = RoleManager(
            swap_interval=cfg.roles.swap_interval,
            warmup_iterations=cfg.roles.warmup_iterations,
            injection_rate=cfg.roles.injection_rate,
            names=adapters.NAMES,
        )
        # One AdamW per adapter so the moment estimates don't bleed across A/B.
        # weight_decay=0: decaying LoRA toward zero would fight the RL signal.
        self.optimizers = {
            name: optim.AdamW(learning_rate=cfg.train.learning_rate, weight_decay=0.0)
            for name in adapters.NAMES
        }
        self.logger = logger
        # Optional raw-text transcript (everything the models write, in order).
        # No-op when unset, so a normal run carries no overhead.
        self.transcript = transcript
        seed_val = cfg.train.seed if seed is None else seed
        self.rng = random.Random(seed_val)
        # Seed MLX's global RNG too: sampling (make_sampler) draws from it, so
        # without this the domain/theme schedule is reproducible but every
        # generation differs run-to-run. (A --resume-step run reseeds from the
        # start, so its sample stream matches a fresh run's *schedule*, not the
        # interrupted run's mid-stream state — acceptable for our purposes.)
        mx.random.seed(seed_val)
        # Init snapshot of each adapter tree, so every iteration can report how
        # far that adapter has drifted from its starting point (adapter-drift
        # curve, DESIGN.md §11). Taken before any training touches the trees.
        self._init_trees = {n: adapters.snapshot(n) for n in adapters.NAMES}
        mx.eval([t for t in self._init_trees.values()])

    # GRPO scoring primitive; ignores the passed `model` (it IS self.base.model).
    def _score(self, model, prompt_ids, completion_ids):
        return self.base.completion_logprobs(prompt_ids, completion_ids)

    # ----- raw-text transcript helpers (no-op when no sink) ----------------
    def _tr_section(self, title: str) -> None:
        if self.transcript is not None:
            self.transcript.section(title)

    def _tr(self, label: str, text: str = "", **meta) -> None:
        if self.transcript is not None:
            self.transcript.entry(label, text, **meta)

    # Verifier judge fallback = the frozen base (DESIGN §12 Q3). Distinct from the
    # taxed OracleTool: judging must not consume the policy's oracle budget.
    #
    # The judge runs a ReAct loop with the CAS `solve` tool so it recomputes each
    # answer instead of eyeballing the worked solution (where LLMs slip on
    # arithmetic). Tool use here is untaxed and never scored — it's verification,
    # not a trained rollout. JUDGE_SYSTEM carries the tool protocol; the grading
    # task + VERDICT contract come in `question` from twin.verifiers.judge.
    def _judge(self, question: str) -> str:
        tcfg = self.cfg.tools
        with self.adapters.using("base"):
            prompt = self.base.render(
                question, system=JUDGE_SYSTEM,
                enable_thinking=self.cfg.model.enable_thinking,
            )
            result = self.base.generate_react(
                prompt,
                tool_runner=self._build_tool_runner(
                    tcfg.judge_tools, tcfg.judge_max_tool_calls
                ),
                max_tokens=self.cfg.gen.oracle_max_tokens,
                temp=self.cfg.gen.oracle_temp,
                top_p=self.cfg.gen.top_p,
                max_rounds=tcfg.judge_tool_rounds,
            )
        self._tr("judge(base+cas)", f"Q: {question}\n\nA: {result.text}",
                 n_tool_calls=result.n_tool_calls)
        return result.text

    def _generate(self, adapter, system, user, *, max_tokens, temp):
        self.adapters.activate(adapter)
        prompt = self.base.render(
            user, system=system, enable_thinking=self.cfg.model.enable_thinking
        )
        return self.base.generate(
            prompt, max_tokens=max_tokens, temp=temp, top_p=self.cfg.gen.top_p
        )

    # ----- inline-tool (ReAct) generation ---------------------------------
    def _build_tool_runner(self, tool_names, max_tool_calls):
        """Fresh ToolHarness + runner for ONE ReAct rollout (creator or judge).
        The runner takes the segment text the model just emitted (ending in
        ``</tool>``), executes the call, and returns the ``<obs>...</obs>`` text
        to splice in (or None to finish). Reuses the existing
        ToolHarness/parse_tool_calls protocol."""
        tcfg = self.cfg.tools
        tools = {}
        for name in tool_names:
            if name == "solve":
                tools["solve"] = lambda arg, _ts=tcfg.cas_timeout_s: solve(arg, timeout_s=_ts)
            elif name == "calc":
                tools["calc"] = calc
        harness = ToolHarness(tools, max_tool_calls=max_tool_calls)

        def runner(segment_text: str):
            results = harness.run(segment_text)
            if not results:
                return None
            return "\n" + ToolHarness.format_observations(results) + "\n"

        return runner

    def _creator_tool_runner(self):
        """Tool runner for a creator rollout — untaxed generation aids; the solver
        path never sees them."""
        tcfg = self.cfg.tools
        return self._build_tool_runner(tcfg.creator_tools, tcfg.creator_max_tool_calls)

    def _generate_creator(self, adapter, system, user):
        self.adapters.activate(adapter)
        prompt = self.base.render(
            user, system=system, enable_thinking=self.cfg.model.enable_thinking
        )
        return self.base.generate_react(
            prompt,
            tool_runner=self._creator_tool_runner(),
            max_tokens=self.cfg.gen.creator_max_tokens,
            temp=self.cfg.gen.creator_temp,
            top_p=self.cfg.gen.top_p,
            max_rounds=self.cfg.tools.creator_tool_rounds,
        )

    # ----- GRPO update with a base-reference KL pass -----------------------
    def _grpo(self, adapter: str, trajs: list[Trajectory]) -> dict:
        trajs = [t for t in trajs if t.n_tokens > 0]
        if not trajs:
            return {"n_traj": 0, "n_tokens": 0, "loss": 0.0, "pg": 0.0,
                    "kl": 0.0, "grad_norm": 0.0}
        # Fully tied batch (every advantage zero): the PG term vanishes exactly
        # and the pure-KL gradient is ≈0, so skip the reference pass + policy
        # pass + backward outright — a saturated 4096-token math iteration
        # otherwise pays several minutes for a no-op step (DESIGN §11 S6).
        if all_zero_advantages(trajs):
            return {"n_traj": len(trajs), "n_tokens": 0, "loss": 0.0, "pg": 0.0,
                    "kl": 0.0, "grad_norm": 0.0, "skipped_zero_adv": True}
        # Reference log-probs from the frozen base (zeroed adapter), no grad.
        # Realize each immediately (mx.eval) rather than batching the eval: a
        # single completion's [T,V] logits is ~2.5GB at a 4096 budget, so
        # deferring would hold every trajectory's forward graph live at once and
        # OOM. (Lazy eval would also reference whatever adapter is active later.)
        self.adapters.activate("base")
        for t in trajs:
            t.ref_logprobs = self.base.completion_logprobs(t.prompt_ids, t.completion_ids)
            mx.eval(t.ref_logprobs)
        # Policy update on this adapter.
        self.adapters.activate(adapter)
        metrics = grpo_update(
            self.base.model,
            self.optimizers[adapter],
            trajs,
            self._score,
            kl_beta=self.cfg.train.kl_beta,
            grad_clip=self.cfg.train.grad_clip,
            microbatch_size=self.cfg.train.grpo_microbatch,
        )
        self.adapters.capture(adapter)
        return metrics

    # ----- one iteration ---------------------------------------------------
    def run_iteration(self, iteration: int) -> dict:
        cfg = self.cfg
        assign = self.roles.assignment(iteration)
        if self.roles.is_swap_iteration(iteration):
            self.roles.maybe_blend(self.adapters, assign)

        domain = self.rng.choice(cfg.game.domains)
        theme = pick_theme(domain, self.rng)
        n = cfg.game.n_problems

        self._tr_section(
            f"iter {iteration} | domain={domain} theme={theme} | "
            f"creator={assign.creator} solver={assign.solver}"
        )

        creator_trajs: list[Trajectory] = []
        solver_trajs: list[Trajectory] = []
        suite_summaries: list[dict] = []
        solver_oracle_total = 0  # oracle calls the solver wrote this iteration
        # Solver accuracy tallies (consistent problems only get a solver rollout).
        solver_problems = 0        # problems posed to the solver
        solver_attempts = 0        # total attempts across those problems
        solver_attempts_solved = 0 # attempts that verified correct
        solver_problems_solved = 0 # problems solved on >=1 attempt

        for g in range(cfg.game.creator_group):
            cgen = self._generate_creator(
                assign.creator, CREATOR_SYSTEM, creator_user(domain, theme, n),
            )
            n_oracle_c = count_oracle_calls(cgen.text)
            self._tr(f"creator[{g}] adapter={assign.creator}", cgen.text,
                     n_tool_calls=cgen.n_tool_calls, n_oracle=n_oracle_c)

            try:
                suite = parse_suite(cgen.text)
            except SuiteParseError as e:
                r = self.engine.creator_parse_gate()
                self._tr(f"creator[{g}] PARSE-FAIL", str(e)[:200],
                         reward=round(r.total, 4))
                creator_trajs.append(Trajectory(
                    cgen.prompt_tokens, cgen.completion_tokens,
                    reward=r.total, loss_mask=cgen.loss_mask, meta={"parsed": False}))
                suite_summaries.append({"parsed": False, "error": str(e)[:120],
                                        "n_tool_calls": cgen.n_tool_calls,
                                        "reward": r.total})
                continue

            # Consistency of the creator's own solutions (verifier-first).
            flags: list[bool] = []
            for p in suite.problems:
                try:
                    flags.append(bool(check_consistency(p, oracle=self._judge).correct))
                except Exception:  # noqa: BLE001 - verifier boundary, never crash a run
                    flags.append(False)

            # Solver rollout on consistent problems only. Void problems are
            # excluded from solver reward (scored_mask=False) but still drag the
            # creator's R_consistency (DESIGN §6.3).
            solve_rates = [0.0] * len(suite.problems)
            scored_mask = [False] * len(suite.problems)
            for i, p in enumerate(suite.problems):
                self._tr(f"problem[{g}.{i}]", p.statement,
                         difficulty=round(p.difficulty, 3),
                         creator_answer=p.answer, consistent=flags[i])
                if not flags[i]:
                    continue
                group: list[Trajectory] = []
                attempt_flags: list[bool] = []
                for k in range(cfg.game.solver_attempts):
                    sgen = self._generate(
                        assign.solver, SOLVER_SYSTEM, solver_user(p),
                        max_tokens=cfg.gen.solver_max_tokens, temp=cfg.gen.solver_temp,
                    )
                    ans = extract_final_answer(sgen.text)
                    try:
                        solved = bool(verify_answer(p, ans, oracle=self._judge).correct)
                    except Exception:  # noqa: BLE001
                        solved = False
                    n_oracle_s = count_oracle_calls(sgen.text)
                    solver_oracle_total += n_oracle_s
                    sr = self.engine.solver_reward(
                        solved, n_oracle_calls=n_oracle_s,
                        n_tokens=len(sgen.completion_tokens),
                        max_tokens=cfg.gen.solver_max_tokens)
                    self._tr(f"solver[{g}.{i}] attempt={k} adapter={assign.solver}",
                             sgen.text, answer=ans, solved=solved,
                             n_oracle=n_oracle_s, n_tokens=len(sgen.completion_tokens),
                             r_brevity=round(sr.r_brevity, 4),
                             reward=round(sr.total, 4))
                    group.append(Trajectory(
                        sgen.prompt_tokens, sgen.completion_tokens, reward=sr.total,
                        meta={"problem_id": p.problem_id, "solved": solved}))
                    attempt_flags.append(solved)
                # Solver GRPO group = the K attempts at THIS problem.
                for t, a in zip(group, group_advantages(
                        [t.reward for t in group], mode=cfg.train.adv_mode)):
                    t.advantage = a
                solver_trajs.extend(group)
                solve_rates[i] = solve_rate(attempt_flags)
                scored_mask[i] = True
                solver_problems += 1
                solver_attempts += len(attempt_flags)
                solver_attempts_solved += sum(attempt_flags)
                solver_problems_solved += 1 if any(attempt_flags) else 0

            creward = self.engine.creator_reward(
                suite, solve_rates, flags, n_oracle_c, scored_mask=scored_mask)
            self._tr(f"creator[{g}] reward", reward=round(creward.total, 4),
                     r_gradient=round(creward.r_gradient, 4),
                     r_consistency=round(creward.r_consistency, 4),
                     n_consistent=int(sum(flags)))
            creator_trajs.append(Trajectory(
                cgen.prompt_tokens, cgen.completion_tokens,
                reward=creward.total, loss_mask=cgen.loss_mask,
                meta={"parsed": True, "suite_id": suite.suite_id}))
            suite_summaries.append({
                "parsed": True, "n_problems": len(suite.problems),
                # Certificate coverage (Sprint 5): problems carrying the
                # machine-checkable check+symbol cert. Watch this rise to
                # n_problems on math iters; a stall/drop means the creator is
                # evading certification for the softer judge fallback.
                "n_cert": sum(
                    1 for p in suite.problems
                    if p.verification.get("check") and p.verification.get("symbol")
                ),
                "n_consistent": int(sum(flags)),
                "solve_rates_by_rank": [round(x, 3) for x in creward.solve_rates_by_rank],
                "r_gradient": round(creward.r_gradient, 4),
                "r_consistency": round(creward.r_consistency, 4),
                "n_oracle": n_oracle_c, "n_tool_calls": cgen.n_tool_calls,
                "reward": round(creward.total, 4),
            })

        # Creator GRPO group = the G_c candidate suites for this prompt.
        for t, a in zip(creator_trajs, group_advantages(
                [t.reward for t in creator_trajs], mode=cfg.train.adv_mode)):
            t.advantage = a

        # Two sequential updates (different trees; order irrelevant).
        solver_metrics = self._grpo(assign.solver, solver_trajs)
        creator_metrics = self._grpo(assign.creator, creator_trajs)
        self._tr_section(
            f"iter {iteration} updates | "
            f"solver loss={solver_metrics['loss']:.4f} kl={solver_metrics['kl']:.4f} "
            f"gnorm={solver_metrics['grad_norm']:.4f} | "
            f"creator loss={creator_metrics['loss']:.4f} kl={creator_metrics['kl']:.4f} "
            f"gnorm={creator_metrics['grad_norm']:.4f}"
        )

        n_parsed = sum(1 for s in suite_summaries if s.get("parsed"))
        # Per-iteration curve instrumentation (DESIGN.md §11): oracle/tool usage,
        # curve-fit quality, and adapter magnitude/drift. Cheap scalars consumed
        # by twin.analysis.curves to plot trends over a run.
        creator_oracle_total = sum(s.get("n_oracle", 0) for s in suite_summaries)
        creator_tool_total = sum(s.get("n_tool_calls", 0) for s in suite_summaries)
        r_gradient_vals = [s["r_gradient"] for s in suite_summaries if "r_gradient" in s]
        adapter_norm = {n: round(self.adapters.global_norm(n), 6)
                        for n in self.adapters.NAMES}
        adapter_drift = {n: round(self.adapters.drift_from(n, self._init_trees[n]), 6)
                         for n in self.adapters.NAMES}
        record = {
            "iter": iteration,
            "domain": domain,
            "theme": theme,
            "creator": assign.creator,
            "solver": assign.solver,
            "n_swaps": assign.n_swaps,
            "creator_reward_mean": round(_mean([t.reward for t in creator_trajs]), 4),
            "solver_reward_mean": round(_mean([t.reward for t in solver_trajs]), 4),
            "solve_rate_mean": round(_mean(
                [s for sm in suite_summaries
                 for s in sm.get("solve_rates_by_rank", [])]), 4),
            "parse_ok_rate": round(n_parsed / max(1, len(suite_summaries)), 3),
            "r_gradient_mean": round(_mean(r_gradient_vals), 4),
            "creator_oracle_calls": creator_oracle_total,
            "solver_oracle_calls": solver_oracle_total,
            "creator_tool_calls": creator_tool_total,
            "adapter_norm": adapter_norm,
            "adapter_drift": adapter_drift,
            "n_solver_trajs": len(solver_trajs),
            "solver_stats": {
                "problems": solver_problems,
                "attempts": solver_attempts,
                "attempts_solved": solver_attempts_solved,
                "problems_solved": solver_problems_solved,
            },
            "solver_update": solver_metrics,
            "creator_update": creator_metrics,
            "suites": suite_summaries,
        }
        if self.logger is not None:
            self.logger.log(record)
        return record

    # ----- driver ----------------------------------------------------------
    def train(self, iters: int | None = None) -> list[dict]:
        iters = iters if iters is not None else self.cfg.train.iters
        history: list[dict] = []
        for it in range(iters):
            history.append(self.run_iteration(it))
            ce = self.cfg.train.checkpoint_every
            if ce and (it + 1) % ce == 0:
                self.save_checkpoints(it + 1)
        return history

    def save_checkpoints(self, step: int) -> None:
        out = Path(self.cfg.paths.checkpoints)
        out.mkdir(parents=True, exist_ok=True)
        for name in self.adapters.NAMES:
            self.adapters.save(name, str(out / f"adapter_{name}_step{step}.safetensors"))
