"""TwentyQTrainer: one 21-questions self-play iteration end-to-end
(twentyq/DESIGN.md §4 Sprint Q5).

Per iteration (creator/guesser assignment from the shared RoleManager):

    creator: N secret rollouts (dictated difficulty rank + target guess rate,
             excluding recent same-category and current-round secrets)
      -> parse gate (unparseable rank -> fixed low reward, no episodes)
      -> repeat gate (twentyq.repeat_handling: a parsed secret matching the
         in-prompt exclusion list -> voided at repeat_gate_reward, no
         episodes; "retry" then re-decodes the same prompt ONCE with the
         excluded secrets masked to -inf, and a non-repeat retry plays the
         rank's episodes and trains the creator with its real game reward)
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
from collections import deque
from dataclasses import asdict
from pathlib import Path

from twin.games.twentyq.episode import run_episode, run_episodes_batched
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
    repeat_penalties,
    secret_turn_trajectories,
    secrets_as_suite,
    shaped_reward_trace,
)
from twin.games.twentyq.schema import (
    Secret,
    SecretParseError,
    normalize_guess,
    parse_secret,
    repeat_matches,
)
from twin.problems.schema import ProblemSuite
from twin.rl import Trajectory, group_advantages
from twin.think import think_share
from twin.train.base import BaseTrainer


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _frozen_update() -> dict:
    """The update metrics of a role that took no GRPO step this iteration.

    Same keys a real ``_grpo`` result carries (so every downstream reader —
    logs, analysis scripts, the transcript tree — keeps working unchanged),
    plus an explicit ``frozen`` marker so a skipped step is never confused
    with a step that ran and happened to produce a zero loss."""
    return {"n_traj": 0, "n_tokens": 0, "loss": 0.0, "pg": 0.0,
            "kl": 0.0, "grad_norm": 0.0, "frozen": True}


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


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial rate.

    Wilson rather than normal-approximation because validation rates sit near
    0.1 on samples as small as 24, exactly where the normal interval misbehaves
    (it happily returns a negative lower bound). Reported so a step-to-step
    change can be read against its own noise instead of by eye — the v6/v7
    series moved 2,0,2,1,3,2,2 wins out of 24 and was discussed as a trend."""
    if n <= 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _validation_metrics(rows: list[dict]) -> dict:
    traces = [[StepReward(**s) for s in row["step_rewards"]] for row in rows]
    wins = [row for row in rows if row["guessed"]]
    lo, hi = _wilson_ci(len(wins), len(rows))
    return {
        "n_episodes": len(rows),
        "guessed": len(wins),
        "guess_rate": round(len(wins) / max(1, len(rows)), 6),
        "guess_rate_ci95": [round(lo, 6), round(hi, 6)],
        "mean_turns": round(_mean([row["turns"] for row in rows]), 6),
        "mean_turns_on_success": (round(_mean([row["turns"] for row in wins]), 6)
                                  if wins else None),
        "format_ended": sum(row["ended"] == "format" for row in rows),
        "repeat_turns": sum(row.get("repeat_turns", 0) for row in rows),
        "reward_signals": _reward_signal_report(traces),
    }


class TwentyQTrainer(BaseTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_recent_secrets()

    def _recent_window_size(self) -> int:
        return max(0, int(self.cfg.twentyq.recent_secret_window))

    def _reset_recent_secrets(self) -> None:
        """Create the one global history shared by both creator adapters."""
        self._recent_secrets = deque(maxlen=self._recent_window_size())

    def _ensure_recent_secrets(self) -> None:
        # Scripted unit-test trainers are intentionally built via __new__.
        if not hasattr(self, "_recent_secrets"):
            self._reset_recent_secrets()

    def _recent_for_category(self, category: str) -> list[str]:
        self._ensure_recent_secrets()
        category_key = category.strip().casefold()
        return [secret for saved_category, secret in self._recent_secrets
                if saved_category.strip().casefold() == category_key]

    def restore_recent_secrets(
        self,
        path: str | Path,
        *,
        before_iteration: int | None = None,
    ) -> int:
        """Rebuild rolling history from iteration rows in a run JSONL.

        When a log contains retries of the same iteration, the last row wins,
        matching the checkpoint reached by the most recent attempt.  Returning
        the retained count makes resume behavior visible to the launcher.
        """
        self._reset_recent_secrets()
        if not self._recent_window_size():
            return 0

        iterations: dict[int, dict] = {}
        # Read defensively so a crash-truncated final JSONL line cannot prevent
        # restoration of all earlier completed iterations.
        with open(Path(path).expanduser()) as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (record.get("type") != "iteration"
                        or record.get("mode") != "twentyq"):
                    continue
                iteration = record.get("iter")
                if not isinstance(iteration, int):
                    continue
                if (before_iteration is not None
                        and iteration >= before_iteration):
                    continue
                iterations[iteration] = record

        for iteration in sorted(iterations):
            record = iterations[iteration]
            category = str(record.get("category", "")).strip()
            if not category:
                continue
            # ``sampled_secrets`` (every parsed output in sample order, voided
            # repeats and masked retries included) is authoritative — it is
            # exactly what entered the live deque. Older logs predate the
            # field; their summaries ARE their full sample stream.
            sampled = record.get("sampled_secrets")
            if isinstance(sampled, list):
                names = sampled
            else:
                names = [s.get("secret") for s in record.get("secrets", [])]
            for name in names:
                name = str(name or "").strip()
                if name:
                    self._recent_secrets.append((category, name))
        return len(self._recent_secrets)

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

    def _ensemble_scoring_active(self) -> bool:
        """Whether the frozen ensemble can affect any number this run.

        Guards the VALIDATION potential call, which historically ran
        unconditionally. Under ``credit: terminal`` with ``w_ensemble: 0.0``
        every potential it produced was multiplied by zero, yet the run still
        paid to load the 4-model roster and score every history prefix of every
        validation episode (v7: "[ensemble] 4 models resident on cuda", dense
        +0.000). Harmless when validation was 24 greedy games; not harmless once
        validation_episodes multiplies that by K."""
        qcfg = self.cfg.twentyq
        return qcfg.credit == "ensemble" or qcfg.w_ensemble != 0.0

    def _ensemble_potentials(self, ep, secret) -> list[float]:
        """Per-turn ensemble potentials Φ_0..Φ_T for one episode: the frozen
        ensemble's mean log-prob of ``secret`` given each Q/A history prefix.

        Returns flat zeros — without touching the ensemble — when no reward
        path can read them (see :meth:`_ensemble_scoring_active`)."""
        if not self._ensemble_scoring_active():
            return [0.0] * (len(history_states(ep)))
        ens = self._get_ensemble()
        states = history_states(ep)
        size = max(1, int(self.cfg.twentyq.ensemble_batch_size))
        if size == 1:
            return [ens.score_history(state, secret.secret).score
                    for state in states]
        return [score.score for score in ens.score_histories(
            states, [secret.secret] * len(states), batch_size=size)]

    def _ensemble_potentials_many(self, episodes, secret) -> list[list[float]]:
        """Score all history prefixes across sibling episodes in shared chunks."""
        size = max(1, int(self.cfg.twentyq.ensemble_batch_size))
        if size == 1:
            return [self._ensemble_potentials(ep, secret) for ep in episodes]
        states_by_episode = [history_states(ep) for ep in episodes]
        flat_states = [state for states in states_by_episode for state in states]
        scores = self._get_ensemble().score_histories(
            flat_states,
            [secret.secret] * len(flat_states),
            batch_size=size,
        )
        out = []
        cursor = 0
        for states in states_by_episode:
            stop = cursor + len(states)
            out.append([score.score for score in scores[cursor:stop]])
            cursor = stop
        return out

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

    def _make_guesser_batch(self, adapter: str, category: str, *,
                            max_turns: int | None = None,
                            temp: float | None = None):
        qcfg = self.cfg.twentyq
        turn_budget = qcfg.max_turns if max_turns is None else max_turns
        sampling_temp = self.cfg.gen.solver_temp if temp is None else temp

        def guesser_batch_fn(requests):
            users = [guesser_user(category, qa_pairs, turn_index, turn_budget)
                     for qa_pairs, turn_index in requests]
            return self._generate_batch(
                adapter, GUESSER_SYSTEM, users,
                max_tokens=qcfg.question_max_tokens,
                temp=sampling_temp,
                completion_batch_size=qcfg.generation_batch_size,
                enable_thinking=qcfg.guesser_thinking,
            )
        return guesser_batch_fn

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

    def _make_answerer_batch(self, adapter: str, secret, *,
                             temp: float | None = None):
        qcfg = self.cfg.twentyq
        sampling_temp = self.cfg.gen.oracle_temp if temp is None else temp

        def answerer_batch_fn(requests):
            users = [answerer_user(secret.secret, secret.category, question)
                     for question, _qa_pairs in requests]
            generations = self._generate_batch(
                adapter, ANSWERER_SYSTEM, users,
                max_tokens=qcfg.answer_max_tokens,
                temp=sampling_temp,
                completion_batch_size=qcfg.generation_batch_size,
                enable_thinking=qcfg.answerer_thinking,
            )
            return [gen.text for gen in generations]
        return answerer_batch_fn

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

    # ----- frontier bank ----------------------------------------------------
    def _draw_bank_secrets(self, n: int) -> list[Secret]:
        """Draw ``n`` secrets from the pre-measured frontier bank.

        Sampling is WITHOUT replacement within an iteration (two GRPO groups on
        the same secret would share an oracle and correlate) and, by default,
        spread evenly across categories. That second part is not cosmetic: v7
        drew ONE category per iteration i.i.d., and since per-category win rates
        differ by 40x (household 0.120 vs food 0.003) the resulting series of
        per-iteration win rates was dominated by which category came up. Three
        separate "trends" read off it during that run turned out to be sampling
        artifacts. Balancing removes the confound at the source.
        """
        qcfg = self.cfg.twentyq
        bank = getattr(self, "_bank", None)
        if bank is None:
            _meta, bank = load_validation_secret_set(qcfg.bank_path)
            if not bank:
                raise ValueError(f"empty secret bank: {qcfg.bank_path}")
            self._bank = bank
        if not qcfg.bank_balance_categories:
            return self.rng.sample(bank, min(n, len(bank)))
        by_category: dict[str, list[Secret]] = {}
        for secret in bank:
            by_category.setdefault(secret.category, []).append(secret)
        drawn: list[Secret] = []
        # Round-robin the categories so a short draw still spans them.
        order = sorted(by_category)
        self.rng.shuffle(order)
        pools = {c: self.rng.sample(by_category[c], len(by_category[c]))
                 for c in order}
        while len(drawn) < n and any(pools[c] for c in order):
            for c in order:
                if len(drawn) >= n:
                    break
                if pools[c]:
                    drawn.append(pools[c].pop())
        return drawn

    # ----- stationary evaluation -------------------------------------------
    def _validation_episodes(self, adapter: str, secret, max_turns: int):
        """The K validation games for one secret under one adapter.

        K=1 keeps the historical path exactly: one GREEDY game through the
        scalar engine. K>1 switches to SAMPLED decoding at ``gen.solver_temp``
        (greedy would return K identical transcripts and buy no power) and runs
        the games in lockstep through the batched engine, so the extra episodes
        cost roughly one game's wall time rather than K.
        """
        qcfg = self.cfg.twentyq
        k = max(1, int(qcfg.validation_episodes))
        if k == 1:
            return [run_episode(
                self._make_guesser(
                    adapter, secret.category, max_turns=max_turns, temp=0.0),
                self._make_answerer("base", secret, temp=0.0),
                secret,
                max_turns=max_turns,
            )]
        return run_episodes_batched(
            self._make_guesser_batch(
                adapter, secret.category, max_turns=max_turns),
            self._make_answerer_batch("base", secret, temp=0.0),
            secret,
            n_episodes=k,
            max_turns=max_turns,
        )

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
                    # No question_retries here, deliberately, whatever training
                    # uses: retries are a scaffold that spends extra compute to
                    # paper over the policy's repeat lock. Measuring under them
                    # would score the scaffold, and the number would stop being
                    # comparable to the v1..v7 series. Validation is always the
                    # bare policy; w_repeat is what has to move this.
                    eps = self._validation_episodes(adapter, secret, max_turns)
                    for ei, ep in enumerate(eps):
                        terminal = episode_reward(
                            qcfg,
                            guessed=ep.guessed,
                            turns_used=ep.turns_used,
                            max_turns=max_turns,
                            # Validation keeps the sparse signal pure: no judge
                            # closeness is folded into a terminal miss.
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
                            "repeat_turns": ep.n_repeat_turns,
                            "terminal_reward": round(terminal.total, 6),
                            "dense_reward_total": round(
                                sum(s.dense for s in trace), 6),
                            "combined_immediate_total": round(
                                sum(s.total for s in trace), 6),
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
                        # Episode suffix only when there is more than one game
                        # per secret, so K=1 transcripts stay byte-identical to
                        # the v1..v7 series.
                        tag = f"{adapter}.{si}" + (f".{ei}" if len(eps) > 1 else "")
                        self._tr(
                            f"validation[{tag}] secret='{secret.secret}' "
                            f"ended={ep.ended} turns={ep.turns_used}",
                            "\n".join(game_lines),
                            guessed=ep.guessed,
                            answerer="base",
                        )
                        self._tr(
                            f"validation[{tag}] per-turn reward components",
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
            "credit": qcfg.credit,
            "diagnostic_ensemble": True,
            "w_ensemble": qcfg.w_ensemble,
            "secret_set": meta,
            "max_turns": max_turns,
            "decoding": ("greedy" if qcfg.validation_episodes <= 1
                         else f"sampled@{self.cfg.gen.solver_temp}"),
            "episodes_per_secret": max(1, int(qcfg.validation_episodes)),
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
        self._ensure_recent_secrets()
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
        # Difficulty dictation (DESIGN §7.4). Gradient: the historical easy->hard
        # ramp across ranks. Flat: one target rate for every rank, so the round
        # asks for a uniformly-pitched bank. The targets computed here are the
        # SAME vector the creator is prompted with and later scored against —
        # "prompt targets == reward targets" holds in both modes.
        if qcfg.difficulty_mode not in ("gradient", "flat"):
            raise ValueError(
                f"unknown twentyq.difficulty_mode: {qcfg.difficulty_mode!r} "
                "(expected gradient | flat)")
        flat = qcfg.difficulty_mode == "flat"
        flat_rate = float(qcfg.flat_target_rate)
        if flat:
            targets = [flat_rate] * n
        else:
            targets = ProblemSuite.target_curve(
                n, cfg.rewards.target_hi, cfg.rewards.target_lo)
        # A frozen role plays and is scored, but takes no GRPO step — and the
        # work that exists only to feed that step is skipped (§7.3).
        freeze_creator = bool(qcfg.freeze_creator)
        freeze_solver = bool(qcfg.freeze_solver)

        frozen_note = "".join([
            " | creator FROZEN" if freeze_creator else "",
            " | solver FROZEN" if freeze_solver else "",
        ])
        self._tr_section(
            f"iter {iteration} | twentyq category={category} | "
            f"creator={assign.creator} guesser={assign.solver} | "
            f"difficulty={qcfg.difficulty_mode}"
            + (f"@{flat_rate:.2f}" if flat else "")
            + frozen_note
        )

        # --- creator: N secret rollouts, dictated difficulty ----------------
        # Repeat handling (DESIGN §6.5): attempt 0 is always unconstrained (it
        # measures the policy's repeat propensity and gives the repeat gate an
        # on-policy trajectory to train against). Under "void"/"retry" a parsed
        # secret matching the exclusion list it was shown is voided — no
        # episodes, fixed repeat_gate_reward in the creator GRPO group — and
        # "retry" then re-decodes the SAME prompt once at creator_temp with
        # every excluded secret banned at the logits level, so the sample comes
        # from the renormalized non-excluded distribution instead of praying.
        repeat_mode = qcfg.repeat_handling
        if repeat_mode not in ("off", "void", "retry"):
            raise ValueError(
                f"unknown twentyq.repeat_handling: {repeat_mode!r} "
                "(expected off | void | retry)")
        rollouts: list[dict] = []      # bookkeeping per attempt, parsed or not
        secrets = []                   # playable secrets, in sample order
        current_secrets: list[str] = []
        sampled_secrets: list[str] = []   # every parsed output, in sample order
        recent_category_secrets = self._recent_for_category(category)
        exact_repeats: list[bool] = []
        normalized_repeats: list[bool] = []
        n_repeat_voided = n_repeat_retries = n_retry_playable = 0
        creator_think: list[float] = []

        def sample_secret(i, difficulty, user, *, banned=None):
            """One creator rollout -> (roll, parsed Secret | None)."""
            # banned_strings is only passed on masked retries, so scripted
            # test fakes of _generate never need the kwarg on the plain path.
            extra = {"banned_strings": banned} if banned is not None else {}
            gen = self._generate(
                assign.creator, CREATOR_SYSTEM, user,
                max_tokens=qcfg.secret_max_tokens, temp=cfg.gen.creator_temp,
                enable_thinking=qcfg.creator_thinking,
                **extra,
            )
            roll = {
                "rank": i,
                "difficulty": float(difficulty),
                "prompt_tokens": gen.prompt_tokens,
                "completion_tokens": gen.completion_tokens,
                "parsed": False,
                "playable": False,
                "repeat": False,
                "retry": banned is not None,
                "_user": user,            # for the transcript tree (cheap str refs)
                "_completion": gen.text,
            }
            creator_think.append(think_share(gen.text))
            try:
                secret = parse_secret(gen.text, default_category=category)
            except SecretParseError as e:
                roll["error"] = str(e)[:120]
                return roll, None
            secret.difficulty = float(difficulty)   # dictated, not claimed
            roll["parsed"] = True
            roll["secret"] = secret.secret
            return roll, secret

        def keep_playable(roll, secret):
            roll["playable"] = True
            roll["secret_index"] = len(secrets)
            rollouts.append(roll)
            secrets.append(secret)
            current_secrets.append(secret.secret)

        bank_mode = qcfg.secret_source == "bank"
        if qcfg.secret_source not in ("creator", "bank"):
            raise ValueError(
                f"unknown twentyq.secret_source: {qcfg.secret_source!r} "
                "(expected creator | bank)")
        if bank_mode and not freeze_creator:
            # Bank mode produces no creator rollout, so there are no creator
            # trajectories and nothing for its GRPO step to consume. Failing
            # loudly beats silently training the creator on an empty group.
            raise ValueError(
                "twentyq.secret_source='bank' requires freeze_creator=true: "
                "the creator authors no secret, so it has no GRPO group")

        if bank_mode:
            for secret in self._draw_bank_secrets(n):
                keep_playable({
                    "rank": len(secrets), "difficulty": secret.difficulty,
                    "prompt_tokens": None, "completion_tokens": None,
                    "parsed": True, "playable": False, "repeat": False,
                    "retry": False, "_user": "(bank)",
                    "_completion": secret_json_for_conditioning(
                        secret.secret, secret.category, secret.difficulty),
                    "secret": secret.secret, "source": "bank",
                }, secret)
                sampled_secrets.append(secret.secret)
                exact_repeats.append(False)
                normalized_repeats.append(False)
            targets = [1.0 - s.difficulty for s in secrets]

        # Ranks to generate: empty in bank mode, where secrets came from disk.
        for i in ([] if bank_mode else range(n)):
            # Flat mode dictates ONE difficulty to every rank (the complement of
            # the shared target rate), so the rank number orders the round
            # without implying a ramp.
            difficulty = (round(1.0 - flat_rate, 2) if flat
                          else (round(i / (n - 1), 2) if n > 1 else 0.5))
            user = creator_secret_user(
                category, i, n, difficulty, targets[i],
                previous=current_secrets or None,
                recent=recent_category_secrets or None,
                difficulty_mode=qcfg.difficulty_mode,
            )
            if cfg.train.log_prompts:
                self._tr(f"prompt creator[{i}]", user)
            exclusions = recent_category_secrets + current_secrets
            roll, secret = sample_secret(i, difficulty, user)
            self._tr(f"creator[{i}] adapter={assign.creator} "
                     f"difficulty={difficulty:.2f} target={targets[i]:.2f}",
                     roll["_completion"])
            if secret is None:
                self._tr(f"creator[{i}] PARSE-FAIL", roll["error"])
                rollouts.append(roll)
                continue
            sampled_secrets.append(secret.secret)
            # Attempt-0 repeat propensity telemetry — unchanged from v4 so the
            # rates stay comparable across repeat_handling modes.
            exact_repeats.append(any(secret.secret == old for old in exclusions))
            normalized = normalize_guess(secret.secret)
            normalized_repeats.append(bool(normalized) and any(
                normalized == normalize_guess(old) for old in exclusions
            ))
            is_repeat = repeat_mode != "off" and any(
                repeat_matches(secret.secret, old) for old in exclusions)
            if not is_repeat:
                keep_playable(roll, secret)
                continue

            roll["repeat"] = True
            n_repeat_voided += 1
            self._tr(f"creator[{i}] REPEAT-VOID '{secret.secret}'",
                     f"matched the exclusion list shown in-prompt "
                     f"(repeat_handling={repeat_mode})")
            rollouts.append(roll)
            if repeat_mode != "retry":
                continue

            n_repeat_retries += 1
            retry_roll, retry_secret = sample_secret(
                i, difficulty, user, banned=exclusions)
            self._tr(f"creator[{i}] MASKED-RETRY adapter={assign.creator} "
                     f"banned={len(exclusions)} secrets",
                     retry_roll["_completion"])
            if retry_secret is None:
                self._tr(f"creator[{i}] RETRY PARSE-FAIL", retry_roll["error"])
                rollouts.append(retry_roll)
                continue
            sampled_secrets.append(retry_secret.secret)
            if any(repeat_matches(retry_secret.secret, old) for old in exclusions):
                # Tokenization-variant slip past the ban list: a repeat never
                # plays episodes, so the rank is voided outright.
                retry_roll["repeat"] = True
                n_repeat_voided += 1
                self._tr(f"creator[{i}] RETRY REPEAT-VOID "
                         f"'{retry_secret.secret}'")
                rollouts.append(retry_roll)
                continue
            n_retry_playable += 1
            keep_playable(retry_roll, retry_secret)

        # --- judge validity gate + episodes per playable secret -------------
        solver_trajs: list[Trajectory] = []
        rates = [0.0] * len(secrets)
        consistent = [False] * len(secrets)
        scored = [False] * len(secrets)
        secret_summaries: list[dict] = []
        guesser_think: list[float] = []
        n_guessed = n_episodes = 0
        n_format_ended = n_answer_ffails = 0
        # Turn-waste telemetry: questions asked vs. distinct questions asked.
        # The headline win rate cannot distinguish "lost while probing" from
        # "lost while locked", and the lock is the fixable one.
        n_questions = n_repeat_turns = n_retries = 0
        phi_vals: list[float] = []
        reward_traces: list[list[StepReward]] = []
        # Per-secret solver-episode entries for the transcript tree (secret
        # index -> list of episode dicts). Only populated when a tree sink
        # is attached; each dict gathers the Episode plus its per-turn credit.
        ep_entries_by_secret: dict[int, list] = {}

        playable_ranks = [r["rank"] for r in rollouts if r["playable"]]
        for si, secret in enumerate(secrets):
            valid = judge_secret_validity(
                secret, self._grade, mode=qcfg.secret_validity).correct
            self._tr(f"secret[{si}] '{secret.secret}' valid={valid}")
            summary = {
                "secret": secret.secret,
                # Per-secret, because a bank iteration spans categories and the
                # v7 postmortem's biggest confound was a per-category win-rate
                # spread that nothing in the record let you condition on.
                "category": secret.category,
                "difficulty": secret.difficulty,
                "target": round(float(targets[playable_ranks[si]]), 3),
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

            # The SECRET's category, not the iteration's. Identical in creator
            # mode (every rank shares the drawn category) but load-bearing in
            # bank mode, where one iteration deliberately spans categories so a
            # 40x per-category win-rate spread cannot masquerade as a trend.
            secret_category = secret.category
            answerer_fn = self._make_answerer(assign.creator, secret)
            guesser_fn = self._make_guesser(assign.solver, secret_category)
            kept_eps, kept_rewards = [], []
            sec_entries: list[dict] = []   # transcript-tree episode entries
            if qcfg.generation_batch_size > 1:
                episodes = run_episodes_batched(
                    self._make_guesser_batch(assign.solver, secret_category),
                    self._make_answerer_batch(assign.creator, secret),
                    secret,
                    n_episodes=k,
                    max_turns=qcfg.max_turns,
                    question_retries=qcfg.question_retries,
                )
            else:
                episodes = [run_episode(
                    guesser_fn, answerer_fn, secret, max_turns=qcfg.max_turns,
                    question_retries=qcfg.question_retries)
                    for _ in range(k)]
            for ki, ep in enumerate(episodes):
                n_episodes += 1
                n_answer_ffails += ep.n_answer_format_fails
                n_questions += len(ep.turns)
                n_repeat_turns += ep.n_repeat_turns
                n_retries += ep.n_question_retries
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
                # Ensemble supplies its own dense signal; terminal is the clean
                # no-dense control. Neither needs the judge-closeness scorer.
                if not ep.guessed and qcfg.credit not in ("ensemble", "terminal"):
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
                            ep, secret, secret_category, qcfg.max_turns),
                    })

            if kept_eps:
                if freeze_solver:
                    # Frozen solver: no update, so no returns, no advantages,
                    # no trajectories — and, critically, no ensemble scoring.
                    # The dense potentials exist only to build solver returns,
                    # so a frozen solver skips ~K*(T+1) scored histories per
                    # secret across every ensemble member (and never loads the
                    # ensemble at all). Episodes still ran — the creator's
                    # calibration reward is measured from their guess rates —
                    # and the terminal component trace is still built (pure
                    # arithmetic, zero potentials) so reward telemetry keeps
                    # exactly the shape every log reader expects.
                    traces = [ensemble_reward_trace(
                        [0.0] * (ep.turns_used + 1), rew.total,
                        gamma=qcfg.gamma, scale=0.0,
                        local=repeat_penalties(ep, qcfg.w_repeat))
                        for ep, rew in zip(kept_eps, kept_rewards)]
                    adv_map = {}
                elif qcfg.credit == "ensemble":
                    # Dense per-turn shaping from the frozen ensemble: score the
                    # secret's log-prob at every history prefix, turn it into
                    # r_t = w_ensemble·(γ·Φ_{t+1} − Φ_t) reward-to-go, and add the
                    # sparse terminal reward on the last turn (rewards.py). One
                    # ensemble pass per history state; no per-turn judge call.
                    potentials = self._ensemble_potentials_many(kept_eps, secret)
                    traces = [ensemble_reward_trace(
                        pots, rew.total, gamma=qcfg.gamma, scale=qcfg.w_ensemble,
                        local=repeat_penalties(ep, qcfg.w_repeat))
                        for pots, ep, rew in zip(potentials, kept_eps, kept_rewards)]
                    returns = [[s.return_ for s in trace] for trace in traces]
                    new_trajs = per_turn_secret_trajectories(
                        kept_eps, returns, adv_mode=cfg.train.adv_mode)
                    solver_trajs.extend(new_trajs)
                    adv_map = _solver_adv_map(new_trajs)
                elif qcfg.credit == "terminal":
                    # Same reward-to-go and turn-index comparison groups as the
                    # ensemble arm, with identically-zero potentials. This
                    # isolates the ensemble term without loading/scoring it.
                    traces = [ensemble_reward_trace(
                        [0.0] * (ep.turns_used + 1), rew.total,
                        gamma=qcfg.gamma, scale=0.0,
                        local=repeat_penalties(ep, qcfg.w_repeat))
                        for ep, rew in zip(kept_eps, kept_rewards)]
                    returns = [[s.return_ for s in trace] for trace in traces]
                    new_trajs = per_turn_secret_trajectories(
                        kept_eps, returns, adv_mode=cfg.train.adv_mode)
                    for traj in new_trajs:
                        traj.meta["credit"] = "terminal"
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
                        if qcfg.credit == "broadcast" and not freeze_solver:
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
        repeat_gate = float(qcfg.repeat_gate_reward)
        r_gradient = 0.0
        creator_total = None
        secret_rewards: list[float] = []
        if secrets:
            suite = secrets_as_suite(secrets, category=category)
            target_by_secret = [targets[rank] for rank in playable_ranks]
            # Flat mode dictates one difficulty to every rank, which trips the
            # suite's anti-collapse spread check (>= 0.3 between easiest and
            # hardest) — a check written for a RAMPED suite and meaningless for
            # a deliberately flat bank. Left alone it silently zeroes the
            # w_valid term on every secret in flat mode (DESIGN §7.5).
            suite_valid = suite.is_valid(min_spread=0.0 if flat else 0.3)
            secret_rewards = self.engine.creator_problem_rewards(
                suite, rates, consistent, valid=suite_valid,
                scored_mask=scored, target_by_problem=target_by_secret,
            )
            # Suite-level view for the log (same quantities the self-play
            # r_gradient curve tracks), incl. expected_n parse-drop scaling.
            # Repeat-voided ranks are absent from the suite, so they scale
            # r_gradient down exactly like parse drops.
            creward = self.engine.creator_reward(
                suite, rates, consistent, valid=suite_valid,
                scored_mask=scored, expected_n=n,
                target_by_problem=target_by_secret,
            )
            r_gradient = creward.r_gradient
            creator_total = creward.total
        # Voided repeats earn the fixed repeat gate: well-formed but disallowed
        # by the exclusion list they were shown, so below every honest secret
        # while a parse failure stays strictly worse.
        roll_rewards = [
            secret_rewards[r["secret_index"]] if r["playable"]
            else repeat_gate if r["repeat"]
            else gate_total
            for r in rollouts
        ]
        # Rewards are computed either way (pure arithmetic over already-measured
        # guess rates — no model call), so a frozen creator keeps its full
        # calibration telemetry; only the trajectories it would have trained on
        # are skipped.
        if not freeze_creator:
            for roll, reward in zip(rollouts, roll_rewards):
                creator_trajs.append(Trajectory(
                    roll["prompt_tokens"], roll["completion_tokens"], reward=reward,
                    meta={"parsed": roll["parsed"], "rank": roll["rank"],
                          "repeat": roll["repeat"], "retry": roll["retry"]},
                ))
            # Creator GRPO group = this iteration's secret rollouts (the N ranks
            # plus any masked retries — a retry is one more same-policy rollout).
            for t, a in zip(creator_trajs, group_advantages(
                    [t.reward for t in creator_trajs], mode=cfg.train.adv_mode)):
                t.advantage = a

        # --- updates + record -------------------------------------------------
        solver_metrics = (_frozen_update() if freeze_solver
                          else self._grpo(assign.solver, solver_trajs))
        creator_metrics = (_frozen_update() if freeze_creator
                           else self._grpo(assign.creator, creator_trajs))
        self._tr_section(
            f"iter {iteration} updates | "
            + ("solver FROZEN (no update)" if freeze_solver else
               f"solver loss={solver_metrics['loss']:.4f} "
               f"kl={solver_metrics['kl']:.4f}")
            + " | "
            + ("creator FROZEN (no update)" if freeze_creator else
               f"creator loss={creator_metrics['loss']:.4f} "
               f"kl={creator_metrics['kl']:.4f}")
        )

        # Parsed outputs enter history regardless of validity or repeat status
        # (the deque tracks creator sampling, not playable episodes) — masked
        # retries included. Delaying this until the iteration successfully
        # finishes keeps in-memory state aligned with what can be restored
        # from the completed run-log row's ``sampled_secrets`` field.
        for name in sampled_secrets:
            self._recent_secrets.append((category, name))

        adapter_norm = {nm: round(self.adapters.global_norm(nm), 6)
                        for nm in self.adapters.NAMES}
        adapter_drift = {nm: round(self.adapters.drift_from(nm, self._init_trees[nm]), 6)
                         for nm in self.adapters.NAMES}
        record = {
            "iter": iteration,
            "mode": "twentyq",
            "credit": qcfg.credit,
            # In bank mode one iteration deliberately spans categories, so the
            # single drawn category would be a lie. Per-secret categories are
            # in "secrets"; this stays the honest iteration-level summary.
            "category": ("mixed:" + ",".join(sorted({s.category for s in secrets}))
                         if bank_mode and secrets else category),
            "creator": assign.creator,
            "solver": assign.solver,
            "n_swaps": assign.n_swaps,
            "generation_batch_size": qcfg.generation_batch_size,
            "ensemble_batch_size": qcfg.ensemble_batch_size,
            "difficulty_mode": qcfg.difficulty_mode,
            "flat_target_rate": flat_rate if flat else None,
            "frozen": {"creator": freeze_creator, "solver": freeze_solver},
            # Measured over the per-rollout rewards, not the trajectories, so a
            # frozen creator (which builds none) still reports its real mean.
            "creator_reward_mean": round(_mean(roll_rewards), 4),
            "solver_reward_mean": round(_mean([t.reward for t in solver_trajs]), 4),
            "parse_ok_rate": round(
                sum(1 for r in rollouts if not r["retry"] and r["parsed"])
                / max(1, n), 3),
            "playable_rate": round(len(secrets) / max(1, n), 3),
            "repeat_handling": repeat_mode,
            "repeat_voided": n_repeat_voided,
            "repeat_retries": n_repeat_retries,
            "repeat_retry_playable": n_retry_playable,
            "sampled_secrets": list(sampled_secrets),
            "recent_secret_exact_repeat_rate": round(
                sum(exact_repeats) / max(1, len(exact_repeats)), 4),
            "recent_secret_normalized_repeat_rate": round(
                sum(normalized_repeats) / max(1, len(normalized_repeats)), 4),
            "recent_secret_history_size": len(self._recent_secrets),
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
                "turns": n_questions,
                "repeat_turns": n_repeat_turns,
                "repeat_turn_rate": round(n_repeat_turns / max(1, n_questions), 4),
                "question_retries": n_retries,
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
            members = []
            for idx, roll in enumerate(rollouts):
                rank = roll["rank"]
                # A frozen creator builds no trajectories, so there is no
                # advantage to report — the reward it WOULD have trained on is
                # still shown (it is the calibration measurement).
                ctraj = creator_trajs[idx] if creator_trajs else None
                if not roll["parsed"]:
                    member = {"rank": rank, "status": "parse_fail", "secret": None,
                              "difficulty": None, "valid": None,
                              "guess_rate": None, "consistent": None,
                              "episodes": []}
                elif roll["repeat"]:
                    member = {"rank": rank, "status": "repeat_void",
                              "secret": roll["secret"],
                              "difficulty": roll["difficulty"], "valid": None,
                              "guess_rate": None, "consistent": None,
                              "episodes": []}
                else:
                    si = roll["secret_index"]
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
                    "retry": roll["retry"],
                    "system": CREATOR_SYSTEM, "user": roll.get("_user"),
                    "completion": roll.get("_completion"),
                    "parse_error": roll.get("error"),
                    "creator_reward": roll_rewards[idx],
                    "creator_advantage": (ctraj.advantage if ctraj is not None
                                          else None),
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
