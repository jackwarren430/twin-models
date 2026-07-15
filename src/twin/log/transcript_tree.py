"""Hierarchical transcript logging for the 21-questions loop, laid out by GRPO
group structure (twentyq/DESIGN.md).

The flat :class:`~twin.log.transcript.TranscriptLogger` writes one text file per
run — everything in the order the models wrote it. This is its structured
counterpart: a *folder tree* mirroring the two GRPO groups a twentyq iteration
trains, so a run can be browsed group-by-group instead of scrolled::

    <root>/                         one folder per run
      _run.md                       run header (config, arm, model, N/K/T, credit)
      iter_00/
        _iter.md                    per-iteration header + aggregate metrics
        creator_0__dog/             CREATOR GRPO member (rank 0), secret "dog"
          _creator.md               the creator rollout + its reward/advantage
          episode_0__win_t7.md      SOLVER GRPO member (one game) + per-turn credit
          episode_1__miss.md
          ...
        creator_1__axolotl__INVALID/
          _creator.md               parsed but validity-gated: no episodes
        creator_2__PARSE-FAIL/
          _creator.md               rollout didn't parse: gate reward, no episodes

The ordering is creator-over-solver by construction, not preference: a solver
GRPO group is baselined *per secret* (rewards.py ``secret_turn_trajectories`` /
``per_turn_secret_trajectories`` operate on one secret's K episodes), so every
solver group belongs to exactly one creator member.

Passive sink: :class:`TwentyQTranscriptTree` receives already-computed values
from the trainer and only writes files (never runs a model or touches rewards).
It is dependency-free and eager-flushing, like the flat logger.
"""

import re
import time
from pathlib import Path
from typing import Any, Optional


def _slug(text: str, *, maxlen: int = 40) -> str:
    """Filesystem-safe lowercase slug: alnum runs joined by underscores."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", (text or "").strip().lower()).strip("_")
    return (s[:maxlen].rstrip("_") or "unnamed")


def _episode_tag(ep) -> str:
    """Scannable outcome suffix for an episode filename."""
    if ep.guessed:
        return f"win_t{ep.turns_used}"
    if ep.ended == "format":
        return "fmtfail"
    return "miss"


def _fnum(x: Optional[float], nd: int = 3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


class TwentyQTranscriptTree:
    """Writes the per-run GRPO-structured transcript tree. One instance per run;
    :meth:`write_iteration` is called once per completed iteration."""

    def __init__(self, root: str | Path, *, meta: Optional[dict[str, Any]] = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_run_header(meta or {})

    # ----- run header -------------------------------------------------------
    def _write_run_header(self, meta: dict[str, Any]) -> None:
        lines = [f"# Run transcript — {meta.get('run', self.root.name)}", ""]
        lines.append(f"_Written {time.strftime('%Y-%m-%d %H:%M:%S')}._")
        lines.append("")
        lines.append("Folder tree laid out by GRPO group: "
                     "`iter_NN/` → `creator_<rank>__<secret>/` (creator GRPO "
                     "member) → `episode_<k>__<outcome>.md` (solver GRPO member).")
        lines.append("")
        if meta:
            lines.append("| field | value |")
            lines.append("|---|---|")
            for k, v in meta.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")
        self._write(self.root / "_run.md", "\n".join(lines))

    # ----- one iteration ----------------------------------------------------
    def write_iteration(
        self,
        *,
        iteration: int,
        category: str,
        creator: str,
        solver: str,
        swapped: bool,
        credit: str,
        members: list[dict[str, Any]],
        aggregate: Optional[dict[str, Any]] = None,
    ) -> None:
        """``members`` is a list of member dicts in rank order — see the module
        docstring / trainer for the shape. ``aggregate`` is the iteration's
        JSONL record (rolled-up metrics shown in ``_iter.md``). Every input
        prompt and model output is written out (prompts in collapsible
        ``<details>`` so the file stays browsable) — the tree's purpose is full
        visibility, independent of the flat log's ``--log-prompts`` flag."""
        idir = self.root / f"iter_{iteration:02d}"
        idir.mkdir(parents=True, exist_ok=True)
        self._write(idir / "_iter.md",
                    self._render_iter_header(iteration, category, creator, solver,
                                             swapped, credit, members, aggregate))
        for m in members:
            self._write_member(idir, m, category=category, credit=credit)

    def write_validation(
        self,
        *,
        step: int,
        secret_set: dict[str, Any],
        answerer: str,
        adapters: dict[str, list[dict[str, Any]]],
        aggregate: Optional[dict[str, Any]] = None,
    ) -> None:
        """Write deterministic fixed-set evaluation beside training trees."""
        vdir = self.root / f"validation_step_{step:04d}"
        vdir.mkdir(parents=True, exist_ok=True)
        lines = [f"# validation step {step}", "",
                 f"- secret set: `{secret_set.get('name', secret_set.get('path'))}`",
                 f"- frozen answerer: `{answerer}`", "- decoding: `greedy`", "",
                 "| adapter | guessed | total | guess rate | terminal mean | "
                 "dense mean | combined R0 mean |",
                 "|---|---|---|---|---|---|---|"]
        metrics_by_adapter = (aggregate or {}).get("adapters", {})
        for adapter in adapters:
            metrics = metrics_by_adapter.get(adapter, {})
            signals = metrics.get("reward_signals", {})
            lines.append(
                f"| {adapter} | {metrics.get('guessed', '-')} | "
                f"{metrics.get('n_episodes', '-')} | {_fnum(metrics.get('guess_rate'))} | "
                f"{_fnum(signals.get('terminal_mean'))} | "
                f"{_fnum(signals.get('dense_immediate_mean'))} | "
                f"{_fnum(signals.get('combined_return_start_mean'))} |")
        self._write(vdir / "_validation.md", "\n".join(lines))

        for adapter, entries in adapters.items():
            adir = vdir / f"adapter_{_slug(adapter)}"
            adir.mkdir(parents=True, exist_ok=True)
            for j, entry in enumerate(entries):
                ep = entry["ep"]
                member = {"secret": entry["secret"]}
                fname = (f"secret_{j:02d}__{_slug(entry['secret'])}__"
                         f"{_episode_tag(ep)}.md")
                self._write(
                    adir / fname,
                    self._render_episode(
                        j, entry, member,
                        category=entry["category"], credit="validation_ensemble"),
                )

    def _render_iter_header(self, iteration, category, creator, solver, swapped,
                            credit, members, aggregate) -> str:
        L = [f"# iter {iteration} — {category}", ""]
        L.append(f"- **creator** (GRPO group A): `{creator}`  |  "
                 f"**solver** (GRPO group B): `{solver}`"
                 f"{'  |  **ROLE SWAP this iter**' if swapped else ''}")
        L.append(f"- credit: `{credit}`")
        if aggregate:
            ep = aggregate.get("episodes", {})
            signals = aggregate.get("reward_signals", {})
            L += [
                "",
                "## aggregate (this iteration)",
                "",
                "| metric | value |",
                "|---|---|",
                f"| guess_rate_mean | {_fnum(aggregate.get('guess_rate_mean'))} |",
                f"| r_gradient | {_fnum(aggregate.get('r_gradient'))} |",
                f"| creator_reward_mean | {_fnum(aggregate.get('creator_reward_mean'))} |",
                f"| solver_reward_mean | {_fnum(aggregate.get('solver_reward_mean'))} |",
                f"| parse_ok_rate | {_fnum(aggregate.get('parse_ok_rate'))} |",
                f"| validity_rate | {_fnum(aggregate.get('validity_rate'))} |",
                f"| phi_mean | {_fnum(aggregate.get('phi_mean'))} |",
                f"| episodes (guessed/total) | "
                f"{ep.get('guessed', '-')}/{ep.get('total', '-')} |",
                f"| format_ended | {ep.get('format_ended', '-')} |",
                f"| n_solver_trajs | {aggregate.get('n_solver_trajs', '-')} |",
                f"| peak_mem_gb | {aggregate.get('peak_mem_gb', '-')} |",
                f"| terminal reward mean | {_fnum(signals.get('terminal_mean'))} |",
                f"| dense reward mean | {_fnum(signals.get('dense_immediate_mean'))} |",
                f"| combined return at turn 0 mean | "
                f"{_fnum(signals.get('combined_return_start_mean'))} |",
            ]
        L += ["", "## creator GRPO members (by rank)", "",
              "| rank | secret | status | guess_rate | consistent | "
              "creator_reward | creator_adv |", "|---|---|---|---|---|---|---|"]
        for m in members:
            L.append(
                f"| {m['rank']} | {m.get('secret') or '—'} | {m['status']} | "
                f"{_fnum(m.get('guess_rate'))} | {m.get('consistent')} | "
                f"{_fnum(m.get('creator_reward'))} | {_fnum(m.get('creator_advantage'))} |")
        return "\n".join(L)

    # ----- one creator GRPO member (folder) ---------------------------------
    def _write_member(self, idir: Path, m: dict[str, Any], *, category, credit) -> None:
        if m["status"] == "parse_fail":
            name = f"creator_{m['rank']}__PARSE-FAIL"
        else:
            suffix = "__INVALID" if m["status"] == "invalid" else ""
            name = f"creator_{m['rank']}__{_slug(m['secret'])}{suffix}"
        mdir = idir / name
        mdir.mkdir(parents=True, exist_ok=True)
        self._write(mdir / "_creator.md", self._render_creator(m, category=category))
        for j, e in enumerate(m.get("episodes", [])):
            ep = e["ep"]
            fname = f"episode_{j}__{_episode_tag(ep)}.md"
            self._write(mdir / fname,
                        self._render_episode(j, e, m, category=category, credit=credit))

    @staticmethod
    def _details(summary: str, body: str) -> list[str]:
        """A collapsible block for a (possibly large) verbatim prompt/output."""
        return [f"<details><summary>{summary}</summary>", "",
                "```", (body or "").strip(), "```", "", "</details>", ""]

    def _render_creator(self, m, *, category) -> str:
        L = [f"# creator member rank {m['rank']} — "
             f"{m.get('secret') or '(parse fail)'}", ""]
        L += [
            "## creator GRPO member",
            "",
            "| field | value |",
            "|---|---|",
            f"| rank (dictated difficulty) | {m['rank']} ({_fnum(m.get('difficulty'), 2)}) |",
            f"| target guess rate | {_fnum(m.get('target'), 2)} |",
            f"| status | {m['status']} |",
            f"| validity | {m.get('valid')} |",
            f"| realized guess_rate | {_fnum(m.get('guess_rate'))} |",
            f"| consistent | {m.get('consistent')} |",
            f"| **creator reward** | {_fnum(m.get('creator_reward'))} |",
            f"| **creator advantage** (vs N-member group mean) | "
            f"{_fnum(m.get('creator_advantage'))} |",
            "",
        ]
        if m.get("parse_error"):
            L += ["## parse error", "", "```", m["parse_error"], "```", ""]
        L += ["## rollout", "", "### input prompt", ""]
        if m.get("system"):
            L += self._details("creator SYSTEM prompt", m["system"])
        if m.get("user"):
            L += self._details("creator USER prompt", m["user"])
        L += ["### model output (raw completion)", "",
              "```", (m.get("completion") or "").strip(), "```", ""]
        return "\n".join(L)

    # ----- one solver GRPO member (episode file) ----------------------------
    def _render_episode(self, j, e, m, *, category, credit) -> str:
        ep = e["ep"]
        rew = e.get("reward_obj")
        L = [f"# episode {j} — secret '{m.get('secret')}' ({category})", ""]
        L += [
            ("## validation solver episode" if credit == "validation_ensemble"
             else "## solver GRPO member"),
            "",
            "| field | value |",
            "|---|---|",
            f"| outcome | {'GUESSED' if ep.guessed else ep.ended} |",
            f"| turns_used | {ep.turns_used} |",
            f"| episode reward | {_fnum(e.get('reward'))} |",
        ]
        if rew is not None:
            L.append(f"| reward breakdown | guess={rew.guessed} "
                     f"eff={_fnum(rew.r_efficiency, 3)} "
                     f"phi_final={_fnum(rew.phi_final)} "
                     f"format_fail={rew.format_fail} |")
        if e.get("adv_broadcast") is not None:
            L.append(f"| broadcast advantage (vs K siblings) | "
                     f"{_fnum(e['adv_broadcast'])} |")
        L.append(f"| credit | {credit} |")
        L.append("")

        # Per-turn component table for every credit mode. New trainer entries
        # carry ``step_rewards``; the old potential/return fallback keeps
        # historical/test data renderable.
        step_rewards = e.get("step_rewards")
        pots = e.get("potentials")
        rets = e.get("returns")
        adv_by_turn = e.get("adv_by_turn") or {}
        if step_rewards is not None or rets is not None:
            L += ["## per-turn credit", "",
                  "| turn | kind | Φ before | Φ after | dense | terminal | "
                  "immediate | assigned return | advantage |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for t in ep.turns:
                row = (step_rewards[t.index] if step_rewards is not None
                       and t.index < len(step_rewards) else {})
                before = row.get("potential_before")
                after = row.get("potential_after")
                if not row and pots is not None:
                    before = pots[t.index] if t.index < len(pots) else None
                    after = pots[t.index + 1] if t.index + 1 < len(pots) else None
                ret = row.get("return_")
                if ret is None and rets is not None and t.index < len(rets):
                    ret = rets[t.index]
                L.append(
                    f"| {t.index} | {t.kind} | {_fnum(before)} | {_fnum(after)} | "
                    f"{_fnum(row.get('dense'))} | {_fnum(row.get('terminal'))} | "
                    f"{_fnum(row.get('total'))} | {_fnum(ret)} | "
                    f"{_fnum(adv_by_turn.get(t.index))} |")
            L.append("")

        # The game itself: every input prompt (collapsed) and every model output.
        prompts = e.get("prompts") or {}
        L += ["## transcript", ""]
        for t in ep.turns:
            pr = prompts.get(t.index, {})
            L += [f"### turn {t.index} — {t.kind}", ""]
            if step_rewards is not None and t.index < len(step_rewards):
                sr = step_rewards[t.index]
                L += [
                    "**reward** → "
                    f"dense={_fnum(sr.get('dense'))}, "
                    f"terminal={_fnum(sr.get('terminal'))}, "
                    f"immediate={_fnum(sr.get('total'))}, "
                    f"assigned_return={_fnum(sr.get('return_'))}, "
                    f"advantage={_fnum(adv_by_turn.get(t.index))}",
                    "",
                ]
            # Guesser: input prompt then raw output.
            L.append("**guesser input**")
            L.append("")
            if pr.get("guesser_system"):
                L += self._details("guesser SYSTEM prompt", pr["guesser_system"])
            if pr.get("guesser_user"):
                L += self._details("guesser USER prompt", pr["guesser_user"])
            L += [f"**guesser output** → `{t.kind.upper()}: {t.content}`", "",
                  "```", (t.raw_text or "").strip(), "```", ""]
            # Answerer: only creator-authored replies have a prompt+output; the
            # engine referees a wrong GUESS with a ground-truth NO (no model call).
            if t.creator_answered and t.answer is not None:
                L.append("**answerer input**")
                L.append("")
                if pr.get("answerer_system"):
                    L += self._details("answerer SYSTEM prompt", pr["answerer_system"])
                if pr.get("answerer_user"):
                    L += self._details("answerer USER prompt", pr["answerer_user"])
                L += [f"**answerer output** → `ANSWER: {t.answer}`", "",
                      "```", (t.answer_raw or "").strip(), "```", ""]
            elif t.answer is not None:
                L += [f"**answer** → `{t.answer}`  _(engine referee — "
                      f"ground-truth reply to a wrong guess, no model call)_", ""]
        return "\n".join(L)

    # ----- io ---------------------------------------------------------------
    def _write(self, path: Path, text: str) -> None:
        with open(path, "w") as f:
            f.write(text.rstrip("\n") + "\n")
