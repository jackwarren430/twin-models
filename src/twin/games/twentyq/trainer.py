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
      -> for failed episodes (non-ensemble credit), final-state closeness Φ
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

import json
from dataclasses import asdict
from pathlib import Path

from twin.games.twentyq.episode import run_episode
from twin.games.twentyq.judge import (
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
    StepReward,
    broadcast_reward_trace,
    ensemble_reward_trace,
    episode_reward,
    guess_rate,
    history_states,
    per_turn_secret_trajectories,
    secret_turn_trajectories,
    secrets_as_suite,
    shaped_reward_trace,
)
from twin.games.twentyq.schema import Secret, SecretParseError, parse_secret
from twin.problems.schema import ProblemSuite
from twin.rl import Trajectory, group_advantages
from twin.think import think_share
from twin.train.base import BaseTrainer


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _solver_adv_map(trajs) -> dict[int, dict[int, float]]:
    """Group solver turn-trajectories back into ``{ep_in_secret: {turn: adv}}``
    for the transcript tree (per-turn credit shows each turn's advantage)."""
    out: dict[int, dict[int, float]] = {}
    for tr in trajs:
        out.setdefault(tr.meta["ep_in_secret"], {})[tr.meta["turn"]] = tr.advantage
    return out


def _reward_signal_report(traces: list[list[StepReward]]) -> dict:
    """Aggregate terminal and dense signals without conflating their scales."""
    traces = [tr for tr in traces if tr]
    terminal = [sum(s.terminal for s in tr) for tr in traces]
    dense = [sum(s.dense for s in tr) for tr in traces]
    immediate = [sum(s.total for s in tr) for tr in traces]
    start_returns = [tr[0].return_ for tr in traces]
    steps = [s for tr in traces for s in tr]
    return {
        "n_episodes": len(traces),
        "n_steps": len(steps),
        "terminal_mean": round(_mean(terminal), 6),
        "terminal_total": round(sum(terminal), 6),
        "dense_immediate_mean": round(_mean(dense), 6),
        "dense_immediate_total": round(sum(dense), 6),
        "combined_immediate_mean": round(_mean(immediate), 6),
        "combined_return_start_mean": round(_mean(start_returns), 6),
        "dense_per_step_mean": round(_mean([s.dense for s in steps]), 6),
        "combined_per_step_mean": round(_mean([s.total for s in steps]), 6),
    }


def _format_reward_trace(trace: list[StepReward], advantages=None) -> str:
    """Plain-text per-turn component table for the flat transcript."""
    advantages = advantages or {}
    lines = [
        "turn | phi_before | phi_after | dense | terminal | immediate | "
        "assigned_return | advantage",
        "-----|------------|-----------|-------|----------|-----------|"
        "-----------------|----------",
    ]
    for i, s in enumerate(trace):
        before = "-" if s.potential_before is None else f"{s.potential_before:+.6f}"
        after = "-" if s.potential_after is None else f"{s.potential_after:+.6f}"
        adv = advantages.get(i) if isinstance(advantages, dict) else advantages
        adv_s = "-" if adv is None else f"{adv:+.6f}"
        lines.append(
            f"{i:>4} | {before:>10} | {after:>9} | {s.dense:+.6f} | "
            f"{s.terminal:+.6f} | {s.total:+.6f} | {s.return_:+.6f} | {adv_s}"
        )
    return "\n".join(lines)


def load_validation_secret_set(path: str | Path) -> tuple[dict, list[Secret]]:
    """Load the versioned stationary secret set used by periodic evaluation."""
    p = Path(path).expanduser()
    if not p.is_absolute() and not p.exists():
        p = Path(__file__).resolve().parents[4] / p
    with open(p) as f:
        payload = json.load(f)
    rows = payload.get("secrets") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"validation secret set has no non-empty 'secrets' list: {p}")
    secrets = [Secret.from_dict(row) for row in rows]
    ids = [s.secret_id for s in secrets]
    if len(ids) != len(set(ids)):
        raise ValueError(f"validation secret ids must be unique: {p}")
    meta = {k: v for k, v in payload.items() if k != "secrets"}
    meta["path"] = str(p)
    meta["n_secrets"] = len(secrets)
    return meta, secrets


def _validation_metrics(rows: list[dict]) -> dict:
    traces = [[StepReward(**s) for s in row["step_rewards"]] for row in rows]
    wins = [row for row in rows if row["guessed"]]
    return {
        "n_episodes": len(rows),
        "guessed": len(wins),
        "guess_rate": round(len(wins) / max(1, len(rows)), 6),
        "mean_turns": round(_mean([row["turns"] for row in rows]), 6),
        "mean_turns_on_success": (round(_mean([row["turns"] for row in wins]), 6)
                                  if wins else None),
        "format_ended": sum(row["ended"] == "format" for row in rows),
        "reward_signals": _reward_signal_report(traces),
    }


class TwentyQTrainer(BaseTrainer):

    # Frozen-base grader for the NL judge contracts (validity and, for
    # non-ensemble credit, closeness Φ; the truthfulness audit was removed
    # 2026-07-13). Uses the neutral twentyq JUDGE_SYSTEM (not the CAS math
    # grader, whose "VERDICT: CORRECT/INCORRECT" instruction hijacked the
    # CLOSENESS contract — q-shakeout-04) and binds twentyq.judge_thinking
    # (OFF by default; a thinking budget truncated the closeness trace before
    # its verdict line — q-shakeout-03). Passed as the ``oracle`` to the fns.
    def _grade(self, question: str) -> str:
        return self._judge(question, system=JUDGE_SYSTEM,
                           enable_thinking=self.cfg.twentyq.judge_thinking)

    # Frozen scoring ensemble for credit="ensemble" (the dense per-turn
    # reward). Loaded lazily and ONCE — the torch/transformers members are
    # heavy and mlx-free, so the import stays out of every other code path.
    def _get_ensemble(self):
        ens = getattr(self, "_ensemble", None)
        if ens is None:
            from twin.games.twentyq.ensemble_reward import EnsembleReward
            qcfg = self.cfg.twentyq
            ens = EnsembleReward.load(
                qcfg.ensemble_models or None,
                device=qcfg.ensemble_device, dtype=qcfg.ensemble_dtype,
            )
            self._ensemble = ens
        return ens

    def _ensemble_potentials(self, ep, secret) -> list[float]:
        """Per-turn ensemble potentials Φ_0..Φ_T for one episode: the frozen
        ensemble's mean log-prob of ``secret`` given each Q/A history prefix."""
        ens = self._get_ensemble()
        return [ens.score_history(state, secret.secret).score
                for state in history_states(ep)]

    # ----- player closures --------------------------------------------------
    def _make_guesser(self, adapter: str, category: str, *,
                      max_turns: int | None = None, temp: float | None = None):
        qcfg = self.cfg.twentyq
        turn_budget = qcfg.max_turns if max_turns is None else max_turns
        sampling_temp = self.cfg.gen.solver_temp if temp is None else temp

        def guesser_fn(qa_pairs, turn_index):
            user = guesser_user(category, qa_pairs, turn_index, turn_budget)
            return self._generate(
                adapter, GUESSER_SYSTEM, user,
                max_tokens=qcfg.question_max_tokens,
                temp=sampling_temp,
                enable_thinking=qcfg.guesser_thinking,
            )
        return guesser_fn

    def _make_answerer(self, adapter: str, secret, *, temp: float | None = None):
        # Low temperature: answering is a truthfulness task, not exploration.
        sampling_temp = self.cfg.gen.oracle_temp if temp is None else temp

        def answerer_fn(question, qa_pairs):
            user = answerer_user(secret.secret, secret.category, question)
            gen = self._generate(
                adapter, ANSWERER_SYSTEM, user,
                max_tokens=self.cfg.twentyq.answer_max_tokens,
                temp=sampling_temp,
                enable_thinking=self.cfg.twentyq.answerer_thinking,
            )
            return gen.text
        return answerer_fn

    def _episode_prompts(self, ep, secret, category: str, max_turns: int) -> dict:
        """Reconstruct the exact per-turn INPUT prompts the players saw, for the
        transcript tree. Deterministic from the finished episode: the guesser at
        turn i saw the history of turns < i (``run_episode`` passes ``ep.qa_pairs``
        of the already-finalized turns), and the answerer prompt depends only on
        the question text (``answerer_user`` ignores the history). No model is
        re-run — this just rebuilds the strings that were rendered."""
        out: dict[int, dict] = {}
        hist: list[tuple[str, str]] = []
        for t in ep.turns:
            entry = {
                "guesser_system": GUESSER_SYSTEM,
                "guesser_user": guesser_user(category, list(hist), t.index, max_turns),
            }
            if t.creator_answered and t.kind == "question":
                entry["answerer_system"] = ANSWERER_SYSTEM
                entry["answerer_user"] = answerer_user(
                    secret.secret, secret.category, t.content)
            out[t.index] = entry
            if t.answer is not None:   # advance history exactly like ep.qa_pairs
                q = f"Is it {t.content}?" if t.kind == "guess" else t.content
                hist.append((q, t.answer))
        return out

    # ----- stationary evaluation -------------------------------------------
    def run_validation(self, step: int) -> dict:
        """Evaluate both trained adapters on a fixed secret set, without GRPO.

        The guesser and frozen-base answerer both decode greedily. That makes
        every checkpoint face the same deterministic games and, importantly,
        avoids advancing the stochastic sampling stream used by training.
        The frozen ensemble is diagnostic only here: it reports the dense
        shaping signal separately from the sparse terminal outcome.
        """
        qcfg = self.cfg.twentyq
        meta, secrets = load_validation_secret_set(qcfg.validation_secret_set)
        max_turns = qcfg.validation_max_turns or qcfg.max_turns
        previous_adapter = getattr(self.adapters, "active", None)
        all_traces: list[list[StepReward]] = []
        by_adapter: dict[str, dict] = {}
        tree_entries: dict[str, list[dict]] = {}
        self._tr_section(
            f"validation step {step} | set={meta.get('name', meta['path'])} | "
            f"guessers={','.join(self.adapters.NAMES)} answerer=base"
        )
        try:
            for adapter in self.adapters.NAMES:
                rows: list[dict] = []
                adapter_tree_entries: list[dict] = []
                for si, secret in enumerate(secrets):
                    ep = run_episode(
                        self._make_guesser(
                            adapter, secret.category, max_turns=max_turns, temp=0.0),
                        self._make_answerer("base", secret, temp=0.0),
                        secret,
                        max_turns=max_turns,
                    )
                    terminal = episode_reward(
                        qcfg,
                        guessed=ep.guessed,
                        turns_used=ep.turns_used,
                        max_turns=max_turns,
                        # Validation deliberately keeps the sparse signal pure:
                        # no judge closeness is folded into a terminal miss.
                        phi_final=None,
                        format_fail=(ep.ended == "format"),
                    )
                    potentials = self._ensemble_potentials(ep, secret)
                    trace = ensemble_reward_trace(
                        potentials,
                        terminal.total,
                        gamma=qcfg.gamma,
                        scale=qcfg.w_ensemble,
                    )
                    all_traces.append(trace)
                    row = {
                        "secret_id": secret.secret_id,
                        "secret": secret.secret,
                        "category": secret.category,
                        "difficulty": secret.difficulty,
                        "guessed": ep.guessed,
                        "ended": ep.ended,
                        "turns": ep.turns_used,
                        "terminal_reward": round(terminal.total, 6),
                        "dense_reward_total": round(sum(s.dense for s in trace), 6),
                        "combined_immediate_total": round(sum(s.total for s in trace), 6),
                        "combined_return_start": round(
                            trace[0].return_ if trace else 0.0, 6),
                        "step_rewards": [asdict(s) for s in trace],
                    }
                    rows.append(row)
                    adapter_tree_entries.append({
                        "ep": ep,
                        "secret": secret.secret,
                        "category": secret.category,
                        "reward": terminal.total,
                        "reward_obj": terminal,
                        "step_rewards": row["step_rewards"],
                        "adv_by_turn": {},
                        "prompts": self._episode_prompts(
                            ep, secret, secret.category, max_turns),
                    })
                    game_lines = [
                        f"  {t.index}: [{t.kind}] {t.content} -> {t.answer}"
                        for t in ep.turns
                    ]
                    self._tr(
                        f"validation[{adapter}.{si}] secret='{secret.secret}' "
                        f"ended={ep.ended} turns={ep.turns_used}",
                        "\n".join(game_lines),
                        guessed=ep.guessed,
                        answerer="base",
                    )
                    self._tr(
                        f"validation[{adapter}.{si}] per-turn reward components",
                        _format_reward_trace(trace),
                        terminal=round(terminal.total, 6),
                        dense=round(sum(s.dense for s in trace), 6),
                    )

                categories: dict[str, dict] = {}
                for category in sorted({row["category"] for row in rows}):
                    categories[category] = _validation_metrics(
                        [row for row in rows if row["category"] == category])
                by_adapter[adapter] = {
                    **_validation_metrics(rows),
                    "by_category": categories,
                    "secrets": rows,
                }
                tree_entries[adapter] = adapter_tree_entries
        finally:
            if previous_adapter is not None:
                self.adapters.activate(previous_adapter)

        record = {
            "step": step,
            "mode": "twentyq_validation",
            "secret_set": meta,
            "max_turns": max_turns,
            "decoding": "greedy",
            "answerer": "base",
            "adapters": by_adapter,
            "reward_signals": _reward_signal_report(all_traces),
        }
        backend = getattr(self, "backend", None)
        if backend is not None:
            peak_gb = backend.peak_memory_gb()
            record["peak_mem_gb"] = round(peak_gb, 2) if peak_gb is not None else None
        if self.logger is not None:
            self.logger.log_validation(record)
        tree = getattr(self, "transcript_tree", None)
        if tree is not None:
            tree.write_validation(
                step=step,
                secret_set=meta,
                answerer="base",
                adapters=tree_entries,
                aggregate=record,
            )
        if backend is not None:
            backend.clear_cache()
            backend.reset_peak_memory()
        self._tr_section(
            f"validation step {step} complete | " + " | ".join(
                f"{name} guess_rate={metrics['guess_rate']:.3f}"
                for name, metrics in by_adapter.items())
        )
        return record

    # ----- one iteration ------------------------------------------------------
    def run_iteration(self, iteration: int) -> dict:
        cfg = self.cfg
        qcfg = cfg.twentyq
        # Optional GRPO-structured transcript tree (injected by the launch
        # script; absent for scripted __new__ test trainers). When present we
        # accumulate per-member / per-episode structured data and write the
        # iteration's folder tree at the end. getattr keeps the zero-overhead,
        # no-sink contract of the flat transcript.
        tree = getattr(self, "transcript_tree", None)
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
                "_user": user,            # for the transcript tree (cheap str refs)
                "_completion": gen.text,
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
        n_guessed = n_episodes = 0
        n_format_ended = n_answer_ffails = 0
        phi_vals: list[float] = []
        reward_traces: list[list[StepReward]] = []
        # Per-secret solver-episode entries for the transcript tree (secret
        # index -> list of episode dicts). Only populated when a tree sink
        # is attached; each dict gathers the Episode plus its per-turn credit.
        ep_entries_by_secret: dict[int, list] = {}

        parsed_ranks = [r["rank"] for r in rollouts if r["parsed"]]
        for si, secret in enumerate(secrets):
            valid = judge_secret_validity(
                secret, self._grade, mode=qcfg.secret_validity).correct
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
            sec_entries: list[dict] = []   # transcript-tree episode entries
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
                # The creator is assumed to answer truthfully: the LLM judge
                # audit of the creator's answers was removed (2026-07-13) — on
                # gemma4-E2B it false-flagged truthful episodes and abstained on
                # many others, voiding a steady chunk of otherwise-good training
                # data (q-full-rot: ~6 void + up to 13/40 unauditable per iter).
                # Every episode now trains; consistency = validity alone.
                phi = None
                # credit="ensemble" supplies its own dense per-turn signal from
                # the frozen ensemble (below); skip the judge closeness call so
                # the two shaping sources never double-count on a failed episode.
                if not ep.guessed and qcfg.credit != "ensemble":
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
                    transcript, guessed=ep.guessed,
                    phi=phi, reward=round(rew.total, 4),
                )
                summary["episodes"].append({
                    "ended": ep.ended, "turns": ep.turns_used,
                    "guessed": ep.guessed,
                    "phi": phi, "reward": round(rew.total, 4),
                })
                if ep.guessed:
                    n_guessed += 1
                kept_eps.append(ep)
                kept_rewards.append(rew)
                if tree is not None:
                    sec_entries.append({
                        "ep": ep, "reward": rew.total, "reward_obj": rew, "phi": phi,
                        "prompts": self._episode_prompts(
                            ep, secret, category, qcfg.max_turns),
                    })

            if kept_eps:
                if qcfg.credit == "ensemble":
                    # Dense per-turn shaping from the frozen ensemble: score the
                    # secret's log-prob at every history prefix, turn it into
                    # r_t = w_ensemble·(γ·Φ_{t+1} − Φ_t) reward-to-go, and add the
                    # sparse terminal reward on the last turn (rewards.py). One
                    # ensemble pass per history state; no per-turn judge call.
                    potentials = [self._ensemble_potentials(ep, secret)
                                  for ep in kept_eps]
                    traces = [ensemble_reward_trace(
                        pots, rew.total, gamma=qcfg.gamma, scale=qcfg.w_ensemble)
                        for pots, rew in zip(potentials, kept_rewards)]
                    returns = [[s.return_ for s in trace] for trace in traces]
                    new_trajs = per_turn_secret_trajectories(
                        kept_eps, returns, adv_mode=cfg.train.adv_mode)
                    solver_trajs.extend(new_trajs)
                    adv_map = _solver_adv_map(new_trajs)
                elif qcfg.credit == "per_turn":
                    # Sprint Q7: judge Φ after every non-terminal turn, shape
                    # into reward-to-go, baseline per turn index across the
                    # sibling episodes. Costs one closeness call per
                    # intermediate state (the per-episode final call above is
                    # unchanged) — the reason this is not the default.
                    traces = []
                    for ep, rew in zip(kept_eps, kept_rewards):
                        inter = [
                            judge_closeness(secret, ep.qa_pairs[:t + 1], self._grade)
                            for t in range(ep.turns_used - 1)
                        ]
                        traces.append(shaped_reward_trace(
                            inter, rew.total,
                            gamma=qcfg.gamma, w_close=qcfg.w_close))
                    returns = [[s.return_ for s in trace] for trace in traces]
                    new_trajs = per_turn_secret_trajectories(
                        kept_eps, returns, adv_mode=cfg.train.adv_mode)
                    solver_trajs.extend(new_trajs)
                    adv_map = _solver_adv_map(new_trajs)
                else:
                    traces = [broadcast_reward_trace(ep.turns_used, rew.total)
                              for ep, rew in zip(kept_eps, kept_rewards)]
                    new_trajs = secret_turn_trajectories(
                        kept_eps, kept_rewards, adv_mode=cfg.train.adv_mode)
                    solver_trajs.extend(new_trajs)
                    badv = group_advantages(
                        [r.total for r in kept_rewards], mode=cfg.train.adv_mode)
                    adv_map = {
                        j: {turn.index: badv[j] for turn in ep.turns}
                        for j, ep in enumerate(kept_eps)
                    }

                # Every transcript gets the same component trace that produced
                # the solver trajectory reward. This is written after all K
                # siblings finish because per-turn GRPO advantages do not exist
                # until the group baseline is known.
                for j, trace in enumerate(traces):
                    reward_traces.append(trace)
                    dense_total = sum(s.dense for s in trace)
                    combined_total = sum(s.total for s in trace)
                    start_return = trace[0].return_ if trace else 0.0
                    summary["episodes"][j].update({
                        "terminal_reward": round(kept_rewards[j].total, 6),
                        "dense_reward_total": round(dense_total, 6),
                        "combined_immediate_total": round(combined_total, 6),
                        "combined_return_start": round(start_return, 6),
                        "step_rewards": [asdict(s) for s in trace],
                    })
                    self._tr(
                        f"episode[{si}.{j}] per-turn reward components",
                        _format_reward_trace(trace, adv_map.get(j, {})),
                        credit=qcfg.credit,
                    )
                    if tree is not None:
                        sec_entries[j]["step_rewards"] = [asdict(s) for s in trace]
                        sec_entries[j]["adv_by_turn"] = adv_map.get(j, {})
                        if qcfg.credit == "broadcast":
                            sec_entries[j]["adv_broadcast"] = badv[j]
                rates[si] = guess_rate([e.guessed for e in kept_eps])
                scored[si] = True
            if tree is not None:
                ep_entries_by_secret[si] = sec_entries
            consistent[si] = valid
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
                "total": n_episodes, "guessed": n_guessed,
                "format_ended": n_format_ended,
                "answer_format_fails": n_answer_ffails,
            },
            "reward_signals": _reward_signal_report(reward_traces),
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
        # Memory: log this iteration's peak (the instrument that caught gemma4's
        # 103GB within-iteration spike — the real fix was train.grpo_microbatch=1
        # so the GRPO batch is scored one trajectory at a time, not ~120 full
        # [T,262144] logit graphs at once), then clear the buffer cache and reset
        # the peak counter at the boundary. Guarded on `backend` so the scripted,
        # __new__-built test trainer (no backend, patched _grpo) still runs the
        # real run_iteration; a real backend always has these protocol methods.
        backend = getattr(self, "backend", None)
        if backend is not None:
            peak_gb = backend.peak_memory_gb()
            record["peak_mem_gb"] = round(peak_gb, 2) if peak_gb is not None else None
            active_fn = getattr(backend, "active_memory_gb", None)
            if active_fn is not None:
                active_gb = active_fn()
                record["active_mem_gb"] = (round(active_gb, 2)
                                           if active_gb is not None else None)
        if self.logger is not None:
            self.logger.log(record)
        # GRPO-structured transcript tree: one creator-member folder per rollout
        # (rank order), each holding its solver-episode files. Assembled from the
        # same rollouts / secret_summaries / creator_trajs the record uses, so it
        # never re-runs a model. See twin.log.transcript_tree.
        if tree is not None:
            si_by_rank = {parsed_ranks[si]: si for si in range(len(secrets))}
            members = []
            for idx, roll in enumerate(rollouts):
                rank = roll["rank"]
                si = si_by_rank.get(rank)
                ctraj = creator_trajs[idx]
                if not roll["parsed"]:
                    member = {"rank": rank, "status": "parse_fail", "secret": None,
                              "difficulty": None, "valid": None,
                              "guess_rate": None, "consistent": None,
                              "episodes": []}
                else:
                    summ = secret_summaries[si]
                    member = {
                        "rank": rank,
                        "status": "ok" if summ["valid"] else "invalid",
                        "secret": summ["secret"],
                        "difficulty": secrets[si].difficulty,
                        "valid": summ["valid"],
                        "guess_rate": summ.get("guess_rate"),
                        "consistent": summ.get("consistent"),
                        "episodes": ep_entries_by_secret.get(si, []),
                    }
                member.update({
                    "target": float(targets[rank]),
                    "system": CREATOR_SYSTEM, "user": roll.get("_user"),
                    "completion": roll.get("_completion"),
                    "parse_error": roll.get("error"),
                    "creator_reward": ctraj.reward,
                    "creator_advantage": ctraj.advantage,
                })
                members.append(member)
            tree.write_iteration(
                iteration=iteration, category=category,
                creator=assign.creator, solver=assign.solver,
                swapped=self.roles.is_swap_iteration(iteration),
                credit=qcfg.credit, members=members, aggregate=record,
            )
        if backend is not None:
            backend.clear_cache()
            backend.reset_peak_memory()
        return record
