"""BaseTrainer: the game-agnostic trainer plumbing shared by the single-step
self-play loop (:class:`twin.train.loop.SelfPlayTrainer`) and the multi-turn
21-questions loop (twentyq/DESIGN.md, Sprint Q1).

Extracted verbatim from ``SelfPlayTrainer`` — everything here is about wiring
(backend, adapters, optimizers, roles, RNG, logging) or about mechanics every
game needs (plain generation, the tool-runner factory, the frozen-base judge,
the GRPO update with its base-reference KL pass, checkpointing, the train
driver). What a "rollout", a "problem"/"secret", or a reward IS stays in the
concrete trainer's :meth:`run_iteration`.
"""

import random
from pathlib import Path

from twin.backends import get_backend
from twin.config import Config
from twin.log import JsonlLogger, TranscriptLogger
from twin.prompts import JUDGE_SYSTEM
from twin.rewards import RewardEngine
from twin.rl import Trajectory, all_zero_advantages
from twin.roles import RoleManager
from twin.tools import (
    OracleTool,
    ToolHarness,
    calc,
    format_tool_responses,
    parse_native_tool_calls,
    solve,
)
from twin.tools.logic import logic_solve
from twin.tools.sandbox import format_sandbox_result, run_python


class BaseTrainer:
    """Owns everything below the game: model + adapters + backend, one AdamW
    per adapter, role rotation, seeding, JSONL/transcript sinks, the judge,
    and the GRPO step. Subclasses implement :meth:`run_iteration`."""

    def __init__(
        self,
        base,
        adapters,
        cfg: Config,
        *,
        logger: JsonlLogger | None = None,
        transcript: TranscriptLogger | None = None,
        seed: int | None = None,
        backend=None,
    ):
        self.base = base
        self.adapters = adapters
        self.cfg = cfg
        # Compute backend (mlx | torch). Resolved from cfg by default so scripted
        # test trainers built without one still work; the framework-specific ops
        # (optimizer, RNG seed, array realize, GRPO step) all route through it.
        self.backend = backend if backend is not None else get_backend(cfg.compute.backend)
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
            name: self.backend.make_optimizer(adapters, name, cfg.train.learning_rate)
            for name in adapters.NAMES
        }
        self.logger = logger
        # Optional raw-text transcript (everything the models write, in order).
        # No-op when unset, so a normal run carries no overhead.
        self.transcript = transcript
        seed_val = cfg.train.seed if seed is None else seed
        self.rng = random.Random(seed_val)
        # Seed the backend's global sampling RNG too: sampling draws from it, so
        # without this the domain/theme schedule is reproducible but every
        # generation differs run-to-run. (A --resume-step run reseeds from the
        # start, so its sample stream matches a fresh run's *schedule*, not the
        # interrupted run's mid-stream state — acceptable for our purposes.)
        self.backend.seed(seed_val)
        # Init snapshot of each adapter tree, so every iteration can report how
        # far that adapter has drifted from its starting point (adapter-drift
        # curve, DESIGN.md §11). Taken before any training touches the trees.
        self._init_trees = {n: adapters.snapshot(n) for n in adapters.NAMES}
        self.backend.realize(list(self._init_trees.values()))

    # GRPO scoring primitive; ignores the passed `model` (it IS self.base.model).
    def _score(self, model, prompt_ids, completion_ids):
        return self.base.completion_logprobs(
            prompt_ids, completion_ids,
            logit_chunk=self.cfg.train.logit_chunk or None,
        )

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
    def _judge(self, question: str, *, enable_thinking=None) -> str:
        tcfg = self.cfg.tools
        # The judge stays on the legacy ReAct protocol regardless of
        # tools.protocol (native migration queued for Sprint 8) and never gets
        # a persona — it must remain a neutral grader. ``enable_thinking``
        # defaults to the global model setting (SelfPlayTrainer's math judge
        # keeps its CAS-backed reasoning); the twentyq grader passes False —
        # its tasks are pure NL judgment (VALID/AUDIT/CLOSENESS), and a
        # thinking budget truncated the closeness trace before the verdict
        # line (q-shakeout-03: phi None on every episode).
        think = (self.cfg.model.enable_thinking if enable_thinking is None
                 else enable_thinking)
        judge_runner, _ = self._build_tool_runner(
            tcfg.judge_tools, tcfg.judge_max_tool_calls
        )
        with self.adapters.using("base"):
            prompt = self.base.render(
                question, system=JUDGE_SYSTEM, enable_thinking=think,
            )
            result = self.base.generate_react(
                prompt,
                tool_runner=judge_runner,
                max_tokens=self.cfg.gen.oracle_max_tokens,
                temp=self.cfg.gen.oracle_temp,
                top_p=self.cfg.gen.top_p,
                max_rounds=tcfg.judge_tool_rounds,
            )
        self._tr("judge(base+cas)", f"Q: {question}\n\nA: {result.text}",
                 n_tool_calls=result.n_tool_calls)
        return result.text

    def _generate(self, adapter, system, user, *, max_tokens, temp,
                  enable_thinking=None):
        """Plain single-turn generation on ``adapter``. ``enable_thinking``
        defaults to the global ``model.enable_thinking`` (so SelfPlayTrainer is
        unchanged); the twentyq loop passes a per-role override, because a
        tight thinking budget truncates mid-trace into a zero-output turn
        (q-shakeout-01: guesser think-share 1.0, every turn a format fail)."""
        self.adapters.activate(adapter)
        think = (self.cfg.model.enable_thinking if enable_thinking is None
                 else enable_thinking)
        prompt = self.base.render(user, system=system, enable_thinking=think)
        return self.base.generate(
            prompt, max_tokens=max_tokens, temp=temp, top_p=self.cfg.gen.top_p
        )

    # ----- inline-tool generation (legacy ReAct or Qwen3 native) ----------
    def _build_tool_runner(self, tool_names, max_tool_calls, *, native: bool = False):
        """Fresh ToolHarness + runner for ONE tool-augmented rollout (creator
        or judge). Returns ``(runner, harness)`` — the harness so the caller
        can inspect per-rollout tool usage afterwards (strict tool gate,
        answer-in-obs diagnostics, Sprint 7).

        Legacy: the runner takes the segment ending in ``</tool>`` and returns
        ``<obs>...</obs>`` text. Native: the segment ends in ``</tool_call>``
        (Qwen3 function calling) and the runner returns the chat-template
        ``<tool_response>`` turn glue. Either way the splice is masked out of
        the GRPO loss by generate_react."""
        tcfg = self.cfg.tools
        tools = {}
        for name in tool_names:
            if name == "solve":
                tools["solve"] = lambda arg, _ts=tcfg.cas_timeout_s: solve(arg, timeout_s=_ts)
            elif name == "calc":
                tools["calc"] = calc
            elif name == "run_python":
                # Sandboxed exec (Sprint 8 coding pipeline): the creator's
                # analog of the CAS — test the reference solution + its own
                # asserts before committing to the problem.
                tools["run_python"] = (
                    lambda code, _ts=tcfg.run_python_timeout_s:
                    format_sandbox_result(run_python(code, timeout_s=_ts))
                )
            elif name == "logic_solve":
                # K&K enumerator (logic domain): tells the creator whether
                # its claims pin a unique solution and what it is — the
                # answer becomes correct by construction.
                tools["logic_solve"] = logic_solve
            elif name == "oracle":
                # The TAXED base-model reference (DESIGN §9), finally wired
                # (§14 gap): each executed call is counted from the harness
                # (see _oracle_count) and taxed by the reward engine. The
                # zeroed adapter is activated around each query and restored
                # after, so a mid-rollout call never leaks policy weights.
                tools["oracle"] = OracleTool.from_model(
                    self.base, self.adapters,
                    max_tokens=self.cfg.gen.oracle_max_tokens,
                    temp=self.cfg.gen.oracle_temp,
                    max_calls=self.cfg.oracle.max_calls_per_turn,
                )
        harness = ToolHarness(
            tools,
            max_tool_calls=max_tool_calls,
            parser=parse_native_tool_calls if native else None,
        )

        if native:
            def runner(segment_text: str):
                results = harness.run(segment_text)
                if not results:
                    return None
                return format_tool_responses(results)
        else:
            def runner(segment_text: str):
                results = harness.run(segment_text)
                if not results:
                    return None
                return "\n" + ToolHarness.format_observations(results) + "\n"

        return runner, harness

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
        # Realize each immediately (backend.realize == mx.eval on MLX) rather
        # than batching: a single completion's [T,V] logits is ~2.5GB at a 4096
        # budget, so deferring would hold every trajectory's forward graph live
        # at once and OOM. backend.no_grad() keeps the torch reference pass off
        # the autograd tape (a null context on MLX, where grad comes only from
        # value_and_grad tracing).
        self.adapters.activate("base")
        with self.backend.no_grad():
            for t in trajs:
                t.ref_logprobs = self.base.completion_logprobs(
                    t.prompt_ids, t.completion_ids,
                    logit_chunk=self.cfg.train.logit_chunk or None,
                )
                self.backend.realize(t.ref_logprobs)
        # Policy update on this adapter.
        self.adapters.activate(adapter)
        metrics = self.backend.grpo_update(
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

    # ----- one iteration (game-specific) -----------------------------------
    def run_iteration(self, iteration: int) -> dict:
        raise NotImplementedError

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
