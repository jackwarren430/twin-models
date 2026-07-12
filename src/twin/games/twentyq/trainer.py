"""TwentyQTrainer: one 21-questions self-play iteration end-to-end
(twentyq/DESIGN.md §4 Sprint Q5).

Per iteration (creator/guesser assignment from the shared RoleManager):

    creator: N secret rollouts (dictated difficulty rank + target guess rate,
             conditioned on previous secrets' JSONs)
      -> parse gate (unparseable rank -> fixed low reward, no episodes)
      -> judge validity gate (invalid secret -> voided: no episodes,
         unscored, drags consistency)
      -> K episodes per playable secret (guesser = solver adapter asks,
         creator adapter answers, engine referees)
      -> per-episode truthfulness audit (any lie -> episode voided) and,
         for failed episodes, final-state closeness Φ
      -> episode rewards; per-turn solver trajectories with broadcast
         episode advantages, grouped per secret
      -> per-secret creator credit via RewardEngine.creator_problem_rewards
         (guess rates as solve rates; see note below)
      -> two sequential GRPO updates (solver, then creator)
      -> JSONL record; transcript.

Creator credit is PER-SECRET by construction, not broadcast: the self-play
loop gets creator advantage variance from G_c candidate suites per iteration,
but an episode-based iteration can afford only ONE set of N secrets — a
broadcast suite reward would make every creator advantage identically zero
(all-tied group), and the creator would train on nothing but parse failures.
Decomposed per-secret rewards make the N secret rollouts a real GRPO group
(baseline = the iteration's mean secret reward), the same shape as the
solver's K-episode groups.

Update cadence is once per iteration — never per question. Rewards don't
exist until an episode ends, the group baseline needs the K sibling episodes,
and mid-collection updates would break the single-step ratio≡1 assumption
grpo.py rests on (DESIGN §2.1).
"""

from twin.games.twentyq.episode import run_episode
from twin.games.twentyq.judge import (
    judge_answer_audit,
    judge_closeness,
    judge_secret_validity,
)
from twin.games.twentyq.prompts import (
    ANSWERER_SYSTEM,
    CREATOR_SYSTEM,
    GUESSER_SYSTEM,
    JUDGE_SYSTEM,
    answerer_user,
    creator_secret_user,
    guesser_user,
    secret_json_for_conditioning,
)
from twin.games.twentyq.rewards import (
    episode_reward,
    guess_rate,
    per_turn_secret_trajectories,
    secret_turn_trajectories,
    secrets_as_suite,
    shaped_returns,
)
from twin.games.twentyq.schema import SecretParseError, parse_secret
from twin.problems.schema import ProblemSuite
from twin.rl import Trajectory, group_advantages
from twin.think import think_share
from twin.train.base import BaseTrainer


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


class TwentyQTrainer(BaseTrainer):

    # Frozen-base grader for the three NL judge contracts (validity / audit /
    # closeness). Uses the neutral twentyq JUDGE_SYSTEM (not the CAS math
    # grader, whose "VERDICT: CORRECT/INCORRECT" instruction hijacked the
    # CLOSENESS contract — q-shakeout-04) and binds twentyq.judge_thinking
    # (OFF by default; a thinking budget truncated the closeness trace before
    # its verdict line — q-shakeout-03). Passed as the ``oracle`` to the fns.
    def _grade(self, question: str) -> str:
        return self._judge(question, system=JUDGE_SYSTEM,
                           enable_thinking=self.cfg.twentyq.judge_thinking)

    # ----- player closures --------------------------------------------------
    def _make_guesser(self, adapter: str, category: str):
        qcfg = self.cfg.twentyq

        def guesser_fn(qa_pairs, turn_index):
            user = guesser_user(category, qa_pairs, turn_index, qcfg.max_turns)
            return self._generate(
                adapter, GUESSER_SYSTEM, user,
                max_tokens=qcfg.question_max_tokens,
                temp=self.cfg.gen.solver_temp,
                enable_thinking=qcfg.guesser_thinking,
            )
        return guesser_fn

    def _make_answerer(self, adapter: str, secret):
        # Low temperature: answering is a truthfulness task, not exploration
        # (and the audit voids the episode when it drifts anyway).
        def answerer_fn(question, qa_pairs):
            user = answerer_user(secret.secret, secret.category, question)
            gen = self._generate(
                adapter, ANSWERER_SYSTEM, user,
                max_tokens=self.cfg.twentyq.answer_max_tokens,
                temp=self.cfg.gen.oracle_temp,
                enable_thinking=self.cfg.twentyq.answerer_thinking,
            )
            return gen.text
        return answerer_fn

    # ----- one iteration ------------------------------------------------------
    def run_iteration(self, iteration: int) -> dict:
        cfg = self.cfg
        qcfg = cfg.twentyq
        assign = self.roles.assignment(iteration)
        if self.roles.is_swap_iteration(iteration):
            self.roles.maybe_blend(self.adapters, assign)

        category = self.rng.choice(qcfg.categories)
        n = qcfg.n_secrets
        k = qcfg.episodes_per_secret
        targets = ProblemSuite.target_curve(
            n, cfg.rewards.target_hi, cfg.rewards.target_lo)

        self._tr_section(
            f"iter {iteration} | twentyq category={category} | "
            f"creator={assign.creator} guesser={assign.solver}"
        )

        # --- creator: N secret rollouts, dictated difficulty ----------------
        rollouts: list[dict] = []      # bookkeeping per rank, parsed or not
        secrets = []                   # parsed secrets, in rank order
        prev_jsons: list[str] = []
        creator_think: list[float] = []
        for i in range(n):
            difficulty = round(i / (n - 1), 2) if n > 1 else 0.5
            user = creator_secret_user(
                category, i, n, difficulty, targets[i],
                previous=prev_jsons or None,
            )
            if cfg.train.log_prompts:
                self._tr(f"prompt creator[{i}]", user)
            gen = self._generate(
                assign.creator, CREATOR_SYSTEM, user,
                max_tokens=qcfg.secret_max_tokens, temp=cfg.gen.creator_temp,
                enable_thinking=qcfg.creator_thinking,
            )
            roll = {
                "rank": i,
                "prompt_tokens": gen.prompt_tokens,
                "completion_tokens": gen.completion_tokens,
                "parsed": False,
            }
            creator_think.append(think_share(gen.text))
            self._tr(f"creator[{i}] adapter={assign.creator} "
                     f"difficulty={difficulty:.2f} target={targets[i]:.2f}",
                     gen.text)
            try:
                secret = parse_secret(gen.text, default_category=category)
            except SecretParseError as e:
                roll["error"] = str(e)[:120]
                self._tr(f"creator[{i}] PARSE-FAIL", str(e)[:200])
                rollouts.append(roll)
                continue
            secret.difficulty = float(difficulty)   # dictated, not claimed
            roll["parsed"] = True
            roll["secret"] = secret.secret
            rollouts.append(roll)
            secrets.append(secret)
            prev_jsons.append(secret_json_for_conditioning(
                secret.secret, secret.category, difficulty))

        # --- judge validity gate + episodes per playable secret -------------
        solver_trajs: list[Trajectory] = []
        rates = [0.0] * len(secrets)
        consistent = [False] * len(secrets)
        scored = [False] * len(secrets)
        secret_summaries: list[dict] = []
        guesser_think: list[float] = []
        n_guessed = n_episodes = n_void = n_unauditable = 0
        n_format_ended = n_answer_ffails = 0
        phi_vals: list[float] = []

        parsed_ranks = [r["rank"] for r in rollouts if r["parsed"]]
        for si, secret in enumerate(secrets):
            valid = judge_secret_validity(secret, self._grade).correct
            self._tr(f"secret[{si}] '{secret.secret}' valid={valid}")
            summary = {
                "secret": secret.secret, "difficulty": secret.difficulty,
                "target": round(float(targets[parsed_ranks[si]]), 3),
                "valid": bool(valid), "episodes": [],
            }
            if not valid:
                # Voided outright: unscored (rates[si] stays 0, scored False)
                # and inconsistent — drags the creator exactly like an
                # inconsistent problem.
                summary["guess_rate"] = 0.0
                summary["consistent"] = False
                secret_summaries.append(summary)
                continue

            answerer_fn = self._make_answerer(assign.creator, secret)
            guesser_fn = self._make_guesser(assign.solver, category)
            kept_eps, kept_rewards = [], []
            lied = False
            for ki in range(k):
                ep = run_episode(guesser_fn, answerer_fn, secret,
                                 max_turns=qcfg.max_turns)
                n_episodes += 1
                n_answer_ffails += ep.n_answer_format_fails
                if ep.ended == "format":
                    n_format_ended += 1
                guesser_think.extend(think_share(t.raw_text) for t in ep.turns)
                lines = []
                # A format-failed episode is always shown with the raw guesser
                # completion that broke the contract — otherwise its summary is
                # `[format_fail]  -> None`, undebuggable (q-shakeout-01). Under
                # log_prompts, every turn's raw text is shown.
                show_raw = cfg.train.log_prompts or ep.ended == "format"
                for t in ep.turns:
                    lines.append(f"  {t.index}: [{t.kind}] {t.content} -> {t.answer}")
                    if show_raw:
                        lines.append(f"      guesser<< {t.raw_text.strip()[:400]}")
                        if t.answer_raw:
                            lines.append(f"      answerer<< {t.answer_raw.strip()[:200]}")
                transcript = "\n".join(lines)
                # Truthfulness audit over creator-authored answers only.
                audit = judge_answer_audit(secret, ep.audited_pairs, self._grade)
                if audit is None and ep.audited_pairs:
                    n_unauditable += 1
                # Void only when MORE than audit_void_fraction of the answers are
                # flagged as lies (and at least one). The default 0.0 is the
                # historical "any single F voids" gate. A noisy judge over-voids
                # under it: on gemma4-E2B one false-F sank otherwise-perfect
                # episodes (all-truthful Apple games, q-gemma-shakeout-02). A
                # majority threshold tolerates that noise yet still catches a
                # systematically-lying answerer — same fail-open spirit as
                # counting unauditable episodes as consistent.
                void = False
                if audit is not None:
                    n_lies = sum(1 for a in audit if not a)
                    void = n_lies > 0 and n_lies > qcfg.audit_void_fraction * len(audit)
                if void:
                    n_void += 1
                    lied = True
                phi = None
                if not void and not ep.guessed:
                    final_guess = (ep.turns[-1].content
                                   if ep.turns and ep.turns[-1].kind == "guess"
                                   else None)
                    phi = judge_closeness(secret, ep.qa_pairs, self._grade,
                                          final_guess=final_guess)
                    if phi is not None:
                        phi_vals.append(phi)
                rew = episode_reward(
                    qcfg, guessed=ep.guessed, turns_used=ep.turns_used,
                    max_turns=qcfg.max_turns, phi_final=phi,
                    format_fail=(ep.ended == "format"),
                )
                self._tr(
                    f"episode[{si}.{ki}] ended={ep.ended} turns={ep.turns_used}",
                    transcript, guessed=ep.guessed, void=void,
                    phi=phi, reward=round(rew.total, 4),
                )
                summary["episodes"].append({
                    "ended": ep.ended, "turns": ep.turns_used,
                    "guessed": ep.guessed, "void": void,
                    "phi": phi, "reward": round(rew.total, 4),
                })
                if void:
                    continue           # a lied-to guesser's data trains nothing
                if ep.guessed:
                    n_guessed += 1
                kept_eps.append(ep)
                kept_rewards.append(rew)

            if kept_eps:
                if qcfg.credit == "per_turn":
                    # Sprint Q7: judge Φ after every non-terminal turn, shape
                    # into reward-to-go, baseline per turn index across the
                    # sibling episodes. Costs one closeness call per
                    # intermediate state (the per-episode final call above is
                    # unchanged) — the reason this is not the default.
                    returns = []
                    for ep, rew in zip(kept_eps, kept_rewards):
                        inter = [
                            judge_closeness(secret, ep.qa_pairs[:t + 1], self._grade)
                            for t in range(ep.turns_used - 1)
                        ]
                        returns.append(shaped_returns(
                            inter, rew.total,
                            gamma=qcfg.gamma, w_close=qcfg.w_close))
                    solver_trajs.extend(per_turn_secret_trajectories(
                        kept_eps, returns, adv_mode=cfg.train.adv_mode))
                else:
                    solver_trajs.extend(secret_turn_trajectories(
                        kept_eps, kept_rewards, adv_mode=cfg.train.adv_mode))
                rates[si] = guess_rate([e.guessed for e in kept_eps])
                scored[si] = True
            consistent[si] = valid and not lied
            summary["guess_rate"] = round(rates[si], 3)
            summary["consistent"] = consistent[si]
            secret_summaries.append(summary)

        # --- creator rewards: per-secret credit (see module docstring) ------
        creator_trajs: list[Trajectory] = []
        gate_total = self.engine.creator_parse_gate().total
        r_gradient = 0.0
        creator_total = None
        if secrets:
            suite = secrets_as_suite(secrets, category=category)
            target_by_secret = [targets[rank] for rank in parsed_ranks]
            secret_rewards = self.engine.creator_problem_rewards(
                suite, rates, consistent,
                scored_mask=scored, target_by_problem=target_by_secret,
            )
            # Suite-level view for the log (same quantities the self-play
            # r_gradient curve tracks), incl. expected_n parse-drop scaling.
            creward = self.engine.creator_reward(
                suite, rates, consistent,
                scored_mask=scored, expected_n=n,
                target_by_problem=target_by_secret,
            )
            r_gradient = creward.r_gradient
            creator_total = creward.total
            rw = iter(secret_rewards)
            roll_rewards = [next(rw) if r["parsed"] else gate_total
                            for r in rollouts]
        else:
            roll_rewards = [gate_total for _ in rollouts]
        for roll, reward in zip(rollouts, roll_rewards):
            creator_trajs.append(Trajectory(
                roll["prompt_tokens"], roll["completion_tokens"], reward=reward,
                meta={"parsed": roll["parsed"], "rank": roll["rank"]},
            ))
        # Creator GRPO group = the N secret rollouts this iteration.
        for t, a in zip(creator_trajs, group_advantages(
                [t.reward for t in creator_trajs], mode=cfg.train.adv_mode)):
            t.advantage = a

        # --- updates + record -------------------------------------------------
        solver_metrics = self._grpo(assign.solver, solver_trajs)
        creator_metrics = self._grpo(assign.creator, creator_trajs)
        self._tr_section(
            f"iter {iteration} updates | "
            f"solver loss={solver_metrics['loss']:.4f} kl={solver_metrics['kl']:.4f} | "
            f"creator loss={creator_metrics['loss']:.4f} kl={creator_metrics['kl']:.4f}"
        )

        adapter_norm = {nm: round(self.adapters.global_norm(nm), 6)
                        for nm in self.adapters.NAMES}
        adapter_drift = {nm: round(self.adapters.drift_from(nm, self._init_trees[nm]), 6)
                         for nm in self.adapters.NAMES}
        record = {
            "iter": iteration,
            "mode": "twentyq",
            "credit": qcfg.credit,
            "category": category,
            "creator": assign.creator,
            "solver": assign.solver,
            "n_swaps": assign.n_swaps,
            "creator_reward_mean": round(_mean([t.reward for t in creator_trajs]), 4),
            "solver_reward_mean": round(_mean([t.reward for t in solver_trajs]), 4),
            "parse_ok_rate": round(len(secrets) / max(1, n), 3),
            "validity_rate": round(
                sum(1 for s in secret_summaries if s["valid"]) / max(1, len(secret_summaries)), 3),
            "guess_rate_mean": round(_mean([r for r, s in zip(rates, scored) if s]), 4),
            "guess_rates_by_rank": [round(r, 3) for r in rates],
            "target_by_rank": [round(float(t), 3) for t in targets],
            "r_gradient": round(r_gradient, 4),
            "creator_suite_reward": (round(creator_total, 4)
                                     if creator_total is not None else None),
            "episodes": {
                "total": n_episodes, "guessed": n_guessed, "void": n_void,
                "unauditable": n_unauditable, "format_ended": n_format_ended,
                "answer_format_fails": n_answer_ffails,
            },
            "phi_mean": round(_mean(phi_vals), 4),
            "creator_think_share": round(_mean(creator_think), 4),
            "guesser_think_share": round(_mean(guesser_think), 4),
            "adapter_norm": adapter_norm,
            "adapter_drift": adapter_drift,
            "n_solver_trajs": len(solver_trajs),
            "solver_update": solver_metrics,
            "creator_update": creator_metrics,
            "secrets": secret_summaries,
        }
        if self.logger is not None:
            self.logger.log(record)
        return record
