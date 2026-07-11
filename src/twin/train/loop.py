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

import json

from twin.analysis.diversity import cross_suite_penalties, mean_pairwise_similarity
from twin.config import Config
from twin.log import JsonlLogger, TranscriptLogger
from twin.problems.schema import (
    ProblemSuite,
    SuiteParseError,
    parse_problem,
    parse_suite,
)
from twin.prompts import (
    creator_persona,
    creator_problem_user,
    creator_system,
    creator_user,
    is_code_problem,
    is_logic_problem,
    opponent_of,
    pick_theme,
    solver_persona,
    solver_system,
    solver_user,
)
from twin.rewards import solve_rate
from twin.rl import Trajectory, group_advantages
from twin.think import think_share
from twin.tools import NATIVE_STOP, tool_schemas
from twin.train.base import BaseTrainer
from twin.train.extract import (
    count_oracle_calls,
    extract_code_block,
    extract_final_answer,
)
from twin.verifiers import check_consistency, verify_answer


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


class SelfPlayTrainer(BaseTrainer):
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
        super().__init__(
            base, adapters, cfg,
            logger=logger, transcript=transcript, seed=seed, backend=backend,
        )
        # Sprint 9 — grounded theme weights (game.theme_weights): loaded once;
        # None keeps the uniform static pool.
        self._theme_weights = None
        if cfg.game.theme_weights:
            with open(cfg.game.theme_weights) as f:
                self._theme_weights = json.load(f)

    def _oracle_count(self, text: str, harness) -> int:
        """Taxable oracle calls for one creator rollout. When the oracle is a
        REAL harness tool (audit watch item — now wired), count executed
        calls from the harness; the legacy text count would double-count
        every executed ``<tool>oracle(...)</tool>`` under the react protocol.
        Without the tool, keep v1 semantics (count what the policy wrote)."""
        if "oracle" in self.cfg.tools.creator_tools:
            return sum(1 for res in harness.calls if res.call.name == "oracle")
        return count_oracle_calls(text)

    def _generate_solver_group(self, adapter, system, user):
        """The K solver attempts at ONE problem. ``gen.solver_batch == 1`` is
        the v1 sequential path (one :meth:`_generate` per attempt); >1 samples
        the shared prompt in continuous-batching chunks (Sprint 8 — the K
        attempts are the ideal batch: identical prompt, independent draws).
        Chunking keeps peak KV-cache bounded at solver_batch sequences."""
        cfg = self.cfg
        k = cfg.game.solver_attempts
        if cfg.gen.solver_batch <= 1:
            return [
                self._generate(
                    adapter, system, user,
                    max_tokens=cfg.gen.solver_max_tokens, temp=cfg.gen.solver_temp,
                )
                for _ in range(k)
            ]
        self.adapters.activate(adapter)
        prompt = self.base.render(
            user, system=system, enable_thinking=cfg.model.enable_thinking
        )
        results = []
        for start in range(0, k, cfg.gen.solver_batch):
            b = min(cfg.gen.solver_batch, k - start)
            results.extend(self.base.generate_batch(
                [prompt] * b,
                max_tokens=cfg.gen.solver_max_tokens,
                temp=cfg.gen.solver_temp,
                top_p=cfg.gen.top_p,
                completion_batch_size=b,
            ))
        return results

    def _generate_creator(self, adapter, system, user):
        """One creator rollout under the configured tool protocol. Returns
        ``(ReactResult, ToolHarness)`` — the harness carries what the tools
        actually did (ok/error per call), which the loop logs per problem and
        the strict tool gate consumes."""
        tcfg = self.cfg.tools
        native = tcfg.protocol == "native"
        self.adapters.activate(adapter)
        creator_thinking = (
            self.cfg.model.enable_thinking
            if self.cfg.model.creator_enable_thinking is None
            else self.cfg.model.creator_enable_thinking
        )
        prompt = self.base.render(
            user,
            system=system,
            enable_thinking=creator_thinking,
            tools=tool_schemas(tcfg.creator_tools) if native else None,
        )
        runner, harness = self._build_tool_runner(
            tcfg.creator_tools, tcfg.creator_max_tool_calls, native=native
        )
        result = self.base.generate_react(
            prompt,
            tool_runner=runner,
            max_tokens=self.cfg.gen.creator_max_tokens,
            temp=self.cfg.gen.creator_temp,
            top_p=self.cfg.gen.top_p,
            max_rounds=tcfg.creator_tool_rounds,
            stop=NATIVE_STOP if native else "</tool>",
        )
        return result, harness

    # ----- per-problem creator generation (Sprint 7) -----------------------
    def _create_suite_per_problem(self, assign, domain, theme, n, g, creator_sys):
        """Build ONE candidate suite as ``n`` separate creator rollouts, one
        problem each (``game.creator_mode = "per_problem"``). Each rollout gets
        the full creator token budget for its own thinking, and each backward
        sees a much shorter trajectory than a whole-suite rollout.

        Rank ``i`` is prompted with its dictated difficulty value and the
        target solve rate from the reward's own ramp, plus (when
        ``game.condition_on_previous``) the JSONs of the problems already
        written — never their thinking. Returns ``(suite, rollouts)``:
        the suite holds the parsed problems in rank order (difficulty is
        overwritten with the dictated value so a scrambled claim can't re-sort
        ranks); ``rollouts`` has one bookkeeping dict per rank, parsed or not.
        A rank that fails to parse is simply absent from the suite — the
        reward engine re-stretches the target over the remaining ranks, and
        the failed rank's trajectory is parse-gated at broadcast time."""
        cfg = self.cfg
        targets = ProblemSuite.target_curve(
            n, cfg.rewards.target_hi, cfg.rewards.target_lo)
        opp = opponent_of(assign.creator) if cfg.game.personas else None
        problems = []
        prev_jsons: list[str] = []
        rollouts: list[dict] = []
        for i in range(n):
            difficulty = round(i / (n - 1), 2) if n > 1 else 0.5
            user = creator_problem_user(
                domain, theme, rank=i, n_problems=n,
                difficulty=difficulty, target_rate=targets[i],
                previous=prev_jsons if (cfg.game.condition_on_previous and prev_jsons) else None,
                opponent=opp,
            )
            if cfg.train.log_prompts:
                self._tr(f"prompt creator[{g}.{i}]", user)
            cgen, harness = self._generate_creator(assign.creator, creator_sys, user)
            roll = {
                "rank": i,
                "prompt_tokens": cgen.prompt_tokens,
                "completion_tokens": cgen.completion_tokens,
                "loss_mask": cgen.loss_mask,
                "n_tool_calls": cgen.n_tool_calls,
                "n_tool_ok": sum(1 for res in harness.calls if res.ok),
                "n_oracle": self._oracle_count(cgen.text, harness),
                "think_share": round(think_share(cgen.text), 4),
                "parsed": False,
                "answer_in_obs": False,
            }
            self._tr(
                f"creator[{g}.{i}] adapter={assign.creator} "
                f"difficulty={difficulty:.2f} target={targets[i]:.2f}",
                cgen.text,
                n_tool_calls=roll["n_tool_calls"], n_tool_ok=roll["n_tool_ok"],
                n_oracle=roll["n_oracle"],
            )
            try:
                problem = parse_problem(cgen.text, default_domain=domain)
            except SuiteParseError as e:
                roll["error"] = str(e)[:120]
                self._tr(f"creator[{g}.{i}] PARSE-FAIL", str(e)[:200])
                rollouts.append(roll)
                continue
            problem.difficulty = float(difficulty)
            answer = (problem.answer or "").strip()
            roll["parsed"] = True
            # Did the stated answer literally appear in a successful tool
            # observation? Log-only adoption signal (the strict gate is
            # game.require_tool_use); a multi-step answer assembled from
            # several observations legitimately reads False.
            roll["answer_in_obs"] = bool(answer) and any(
                res.ok and answer in res.output for res in harness.calls)
            rollouts.append(roll)
            problems.append(problem)
            prev_jsons.append(json.dumps({
                "statement": problem.statement,
                "difficulty": problem.difficulty,
                "solution": problem.solution,
                "answer": problem.answer,
                **({"verification": problem.verification}
                   if problem.verification else {}),
            }))
        suite = (ProblemSuite(problems=problems, theme=theme, domain=domain)
                 if problems else None)
        return suite, rollouts

    # ----- one iteration ---------------------------------------------------
    def run_iteration(self, iteration: int) -> dict:
        cfg = self.cfg
        assign = self.roles.assignment(iteration)
        if self.roles.is_swap_iteration(iteration):
            self.roles.maybe_blend(self.adapters, assign)

        domain = self.rng.choice(cfg.game.domains)
        # getattr: scripted test trainers are built via __new__ and may not
        # carry the grounded-theme attribute.
        theme = pick_theme(domain, self.rng,
                           weights=getattr(self, "_theme_weights", None))
        n = cfg.game.n_problems

        self._tr_section(
            f"iter {iteration} | domain={domain} theme={theme} | "
            f"creator={assign.creator} solver={assign.solver}"
        )

        creator_trajs: list[Trajectory] = []
        solver_trajs: list[Trajectory] = []
        suite_summaries: list[dict] = []
        solver_oracle_total = 0  # oracle calls the solver wrote this iteration
        # Think-share telemetry (Sprint 8): fraction of each rollout spent
        # inside <think>. Watch solver trend under brevity pressure (SPIRAL's
        # thinking collapse) and creator pinning at 1.0 (truncation spirals).
        creator_think: list[float] = []
        solver_think: list[float] = []
        # Diversity bookkeeping (Sprint 9): per parsed suite, its id, its
        # problem statements, and its summary dict — consumed after the
        # group loop for the repetition telemetry/penalty.
        parsed_records: list[tuple[str, list[str], dict]] = []
        # Solver accuracy tallies (consistent problems only get a solver rollout).
        solver_problems = 0        # problems posed to the solver
        solver_attempts = 0        # total attempts across those problems
        solver_attempts_solved = 0 # attempts that verified correct
        solver_problems_solved = 0 # problems solved on >=1 attempt

        # Sprint 7: persona-aware system prompts (neutral when personas off —
        # identical to the pre-Sprint-7 constants). Personas are glued to
        # adapters (A=alpha, B=omega) whatever role each plays this iteration;
        # the judge and the held-out benchmark never see them.
        creator_sys = creator_system(
            native_tools=(cfg.tools.protocol == "native"),
            persona=creator_persona(assign.creator) if cfg.game.personas else None,
        )
        solver_sys = solver_system(
            persona=solver_persona(assign.solver) if cfg.game.personas else None,
        )
        # Coding-pipeline variant (Sprint 8): code problems are graded by
        # executing the solver's function against the creator's tests, so the
        # math ANSWER-line contract is replaced per problem below.
        solver_sys_code = solver_system(
            persona=solver_persona(assign.solver) if cfg.game.personas else None,
            code=True,
        )
        # Logic variant (domain expansion): assignment-format ANSWER line.
        solver_sys_logic = solver_system(
            persona=solver_persona(assign.solver) if cfg.game.personas else None,
            logic=True,
        )

        # Prompt transcript entries (train.log_prompts). System prompts go in
        # once per iteration — not once per run, because rotation changes the
        # persona glued to each role — and the solver variants lazily, only if
        # a problem of that kind actually shows up this iteration.
        if cfg.train.log_prompts:
            self._tr("prompt creator_system", creator_sys)
        solver_sys_logged: set[str] = set()

        for g in range(cfg.game.creator_group):
            # --- creator generation --------------------------------------
            # Uniform shape either way: `suite` (or None) + `rollouts`, one
            # bookkeeping dict per creator rollout (1 in suite mode, N in
            # per-problem mode) carrying tokens/mask + tool diagnostics.
            if cfg.game.creator_mode == "per_problem":
                suite, rollouts = self._create_suite_per_problem(
                    assign, domain, theme, n, g, creator_sys
                )
            else:
                cuser = creator_user(domain, theme, n)
                if cfg.train.log_prompts:
                    self._tr(f"prompt creator[{g}]", cuser)
                cgen, charness = self._generate_creator(
                    assign.creator, creator_sys, cuser,
                )
                roll = {
                    "prompt_tokens": cgen.prompt_tokens,
                    "completion_tokens": cgen.completion_tokens,
                    "loss_mask": cgen.loss_mask,
                    "n_tool_calls": cgen.n_tool_calls,
                    "n_tool_ok": sum(1 for res in charness.calls if res.ok),
                    "n_oracle": self._oracle_count(cgen.text, charness),
                    "think_share": round(think_share(cgen.text), 4),
                    "parsed": False,
                    "answer_in_obs": False,
                }
                self._tr(f"creator[{g}] adapter={assign.creator}", cgen.text,
                         n_tool_calls=roll["n_tool_calls"],
                         n_tool_ok=roll["n_tool_ok"], n_oracle=roll["n_oracle"])
                suite = None
                try:
                    suite = parse_suite(cgen.text)
                    roll["parsed"] = True
                except SuiteParseError as e:
                    roll["error"] = str(e)[:120]
                rollouts = [roll]

            n_oracle_c = sum(r["n_oracle"] for r in rollouts)
            n_tool_calls_c = sum(r["n_tool_calls"] for r in rollouts)
            n_tool_ok_c = sum(r["n_tool_ok"] for r in rollouts)
            creator_think.extend(r.get("think_share", 0.0) for r in rollouts)

            # --- parse gate: no usable suite -> fixed low reward, no solver
            if suite is None or not suite.problems:
                r = self.engine.creator_parse_gate()
                detail = rollouts[-1].get("error", "no problems parsed")
                self._tr(f"creator[{g}] PARSE-FAIL", str(detail)[:200],
                         reward=round(r.total, 4))
                for roll in rollouts:
                    creator_trajs.append(Trajectory(
                        roll["prompt_tokens"], roll["completion_tokens"],
                        reward=r.total, loss_mask=roll["loss_mask"],
                        meta={"parsed": False}))
                suite_summaries.append({"parsed": False, "error": str(detail)[:120],
                                        "n_tool_calls": n_tool_calls_c,
                                        "n_tool_ok": n_tool_ok_c,
                                        "n_rollouts": len(rollouts),
                                        "reward": r.total})
                continue

            # Rollout backing each problem: its own rollout in per-problem
            # mode, the single suite rollout otherwise (for the tool gate and
            # the answer-in-obs diagnostic).
            if cfg.game.creator_mode == "per_problem":
                prob_rolls = [r for r in rollouts if r["parsed"]]
            else:
                prob_rolls = [rollouts[0]] * len(suite.problems)

            # --- consistency of the creator's own solutions (verifier-first),
            # behind the optional strict tool gate (Sprint 7): with
            # game.require_tool_use, a problem whose rollout made no successful
            # tool call is voided outright — the "guessed what the tool would
            # return" case. Voids drag R_consistency like any other.
            flags: list[bool] = []
            tool_gated = 0
            for p, roll in zip(suite.problems, prob_rolls):
                if cfg.game.require_tool_use and roll["n_tool_ok"] == 0:
                    flags.append(False)
                    tool_gated += 1
                    self._tr(f"problem[{g}] TOOL-GATED (no successful tool call)",
                             p.statement[:160])
                    continue
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
                p_is_code = is_code_problem(p)
                s_variant = ("code" if p_is_code
                             else "logic" if is_logic_problem(p) else "default")
                s_sys = (solver_sys_code if p_is_code
                         else solver_sys_logic if s_variant == "logic"
                         else solver_sys)
                s_user = solver_user(p)
                if cfg.train.log_prompts:
                    if s_variant not in solver_sys_logged:
                        solver_sys_logged.add(s_variant)
                        self._tr(f"prompt solver_system[{s_variant}]", s_sys)
                    # The K attempts share this one prompt; logged once.
                    self._tr(f"prompt solver[{g}.{i}]", s_user)
                sgens = self._generate_solver_group(assign.solver, s_sys, s_user)
                solver_think.extend(think_share(s.text) for s in sgens)
                for k, sgen in enumerate(sgens):
                    ans = (extract_code_block(sgen.text) if p_is_code
                           else extract_final_answer(sgen.text))
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

            # Per-problem mode (audit 2026-07-03): a parse-failed rank is
            # absent from the suite, so the engine must (a) scale the gradient
            # by the parsed fraction of the INTENDED n (expected_n) and
            # (b) grade each parsed rank against the target it was prompted
            # with (target_by_problem) rather than a re-stretched ramp —
            # otherwise dropping hard ranks out-earns writing them.
            problem_rewards: list[float] | None = None
            if cfg.game.creator_mode == "per_problem":
                rank_targets = ProblemSuite.target_curve(
                    n, cfg.rewards.target_hi, cfg.rewards.target_lo)
                target_by_problem = [rank_targets[r["rank"]] for r in prob_rolls]
                creward = self.engine.creator_reward(
                    suite, solve_rates, flags, n_oracle_c,
                    scored_mask=scored_mask, expected_n=n,
                    target_by_problem=target_by_problem)
                # Credit decomposition (Sprint 8, config-gated): each parsed
                # rank earns its own reward instead of the broadcast total.
                if cfg.game.credit == "per_problem":
                    problem_rewards = self.engine.creator_problem_rewards(
                        suite, solve_rates, flags,
                        scored_mask=scored_mask,
                        target_by_problem=target_by_problem,
                        n_oracle_by_problem=[r["n_oracle"] for r in prob_rolls],
                    )
            else:
                creward = self.engine.creator_reward(
                    suite, solve_rates, flags, n_oracle_c, scored_mask=scored_mask)
            self._tr(f"creator[{g}] reward", reward=round(creward.total, 4),
                     r_gradient=round(creward.r_gradient, 4),
                     r_consistency=round(creward.r_consistency, 4),
                     n_consistent=int(sum(flags)))
            # Broadcast credit (Sprint 7): the suite-level reward is shared by
            # every rollout that contributed a parsed problem — in suite mode
            # that's the single suite rollout (v1 behaviour, unchanged); in
            # per-problem mode all N problem rollouts carry the same reward
            # (and hence the same advantage). A rank that failed to parse gets
            # the parse-gate reward individually instead, so garbage output is
            # penalized at the trajectory that produced it. Per-problem credit
            # decomposition is a queued Sprint-8 ablation.
            gate_total = self.engine.creator_parse_gate().total
            # Reward per rollout: the broadcast suite total (default), or —
            # with game.credit == "per_problem" — that rank's own decomposed
            # reward; parse-failed ranks always get the gate individually.
            if problem_rewards is not None:
                rw = iter(problem_rewards)
                roll_rewards = [next(rw) if r["parsed"] else gate_total
                                for r in rollouts]
            else:
                roll_rewards = [creward.total if r["parsed"] else gate_total
                                for r in rollouts]
            for roll, reward in zip(rollouts, roll_rewards):
                meta = {"parsed": roll["parsed"], "suite_id": suite.suite_id}
                if "rank" in roll:
                    meta["rank"] = roll["rank"]
                creator_trajs.append(Trajectory(
                    roll["prompt_tokens"], roll["completion_tokens"],
                    reward=reward, loss_mask=roll["loss_mask"], meta=meta))
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
                **({"problem_rewards": [round(r, 4) for r in problem_rewards]}
                   if problem_rewards is not None else {}),
                "n_oracle": n_oracle_c, "n_tool_calls": n_tool_calls_c,
                # Sprint 7 tool-adoption watchdogs: successful tool calls, how
                # many stated answers literally appeared in a tool observation,
                # ranks lost to the strict gate / parse failures.
                "n_tool_ok": n_tool_ok_c,
                "n_answer_in_obs": sum(1 for r in prob_rolls if r["answer_in_obs"]),
                "n_tool_gated": tool_gated,
                "n_rollouts": len(rollouts),
                "n_parse_failed": sum(1 for r in rollouts if not r["parsed"]),
                "reward": round(creward.total, 4),
            })
            parsed_records.append((
                suite.suite_id,
                [p.statement for p in suite.problems],
                suite_summaries[-1],
            ))

        # Sprint 9 — diversity: always-on repetition telemetry, config-gated
        # penalty (rewards.w_diversity). Computed across the whole group so a
        # suite that copies another's problems earns less than one that
        # explored; a shared theme baseline cancels in the GRPO advantage.
        problem_similarity = 0.0
        if parsed_records:
            all_statements = [s for _, stmts, _ in parsed_records for s in stmts]
            problem_similarity = round(mean_pairwise_similarity(all_statements), 4)
            penalties = cross_suite_penalties(
                [stmts for _, stmts, _ in parsed_records])
            penalty_by_suite: dict[str, float] = {}
            for (sid, _, summary), pen in zip(parsed_records, penalties):
                summary["repetition"] = round(pen, 4)
                penalty_by_suite[sid] = pen
            if cfg.rewards.w_diversity:
                for t in creator_trajs:
                    sid = t.meta.get("suite_id")
                    if t.meta.get("parsed") and sid in penalty_by_suite:
                        t.reward -= cfg.rewards.w_diversity * penalty_by_suite[sid]

        # Creator GRPO group = every creator trajectory this iteration. In
        # suite mode that's the G_c suites (v1 semantics, unchanged). In
        # per-problem mode it's G_c·N broadcast trajectories: the mean baseline
        # is then a per-trajectory mean, which equals the suite mean whenever
        # every rank parsed (identical math to v1) and reweights only when
        # parse-gate rewards are mixed in.
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
        creator_tool_ok_total = sum(s.get("n_tool_ok", 0) for s in suite_summaries)
        answer_in_obs_total = sum(s.get("n_answer_in_obs", 0) for s in suite_summaries)
        tool_gated_total = sum(s.get("n_tool_gated", 0) for s in suite_summaries)
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
            # Sprint 7 tool-adoption watchdogs (native protocol / strict gate):
            "creator_tool_ok": creator_tool_ok_total,
            "creator_answer_in_obs": answer_in_obs_total,
            "creator_tool_gated": tool_gated_total,
            # Sprint 8 think-share telemetry (see accumulators above):
            "creator_think_share": round(_mean(creator_think), 4),
            "solver_think_share": round(_mean(solver_think), 4),
            # Sprint 9 diversity telemetry: mean pairwise statement similarity
            # across every parsed problem this iteration (collapse watchdog;
            # per-suite nearest-neighbour "repetition" lives in the summaries).
            "problem_similarity": problem_similarity,
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
