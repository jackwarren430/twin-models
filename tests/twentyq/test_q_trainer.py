"""Sprint Q5 — TwentyQTrainer end-to-end through the REAL ``run_iteration``:
secret rollouts with dictated difficulty, parse/validity gates, K scripted
episodes per secret, per-secret creator credit, broadcast solver advantages,
rotation + the no-rotation control. Every model call and judge call is stubbed;
the machinery around them is the real code. (The creator is assumed truthful:
the LLM answer-audit / voiding path was removed 2026-07-13.)"""

import json
import math
import random
import re
from collections import deque

import pytest

from twin.config import Config
from twin.games.twentyq.prompts import (
    ANSWERER_SYSTEM,
    CREATOR_SYSTEM,
    GUESSER_SYSTEM,
)
from twin.games.twentyq.trainer import TwentyQTrainer
from twin.models.types import GenResult
from twin.rewards import RewardEngine
from twin.roles import RoleManager

# --------------------------------------------------------------------------- #
# Scripted stand-ins
# --------------------------------------------------------------------------- #

class _FakeAdapters:
    NAMES = ["A", "B"]

    def activate(self, name):
        pass

    def global_norm(self, name):
        return 0.0

    def drift_from(self, name, tree):
        return 0.0


def _secret_json(secret, difficulty):
    return f'{{"secret": "{secret}", "category": "animal", "difficulty": {difficulty}, "notes": "n"}}'


def _config(**overrides):
    d = {
        "twentyq": {"n_secrets": 2, "episodes_per_secret": 2, "max_turns": 2,
                    "categories": ["animal"]},
        "rewards": {"target_hi": 0.9, "target_lo": 0.1},
        "roles": {"swap_interval": 100, "warmup_iterations": 0},
    }
    d.update(overrides)
    return Config.from_dict(d)


def _make_trainer(cfg, creator_script, guesser_script, *,
                  validity=None):
    """A TwentyQTrainer with real reward/role/episode machinery and scripted
    generation + judging. ``creator_script``/``guesser_script`` are consumed
    in order; the answerer always replies ANSWER: YES. ``validity`` maps
    secret text -> bool (default: everything VALID)."""
    t = TwentyQTrainer.__new__(TwentyQTrainer)
    t.cfg = cfg
    t.engine = RewardEngine(cfg.rewards)
    t.roles = RoleManager(
        swap_interval=cfg.roles.swap_interval,
        warmup_iterations=cfg.roles.warmup_iterations,
        injection_rate=0.0, names=["A", "B"],
    )
    t.rng = random.Random(0)
    t.logger = None
    t.transcript = None
    t.adapters = _FakeAdapters()
    t._init_trees = {"A": None, "B": None}
    t.captured = {
        "grpo": {}, "judge": [], "thinking": {}, "creator_users": [],
        "creator_banned": [], "creator_temps": [],
        "generation_batches": [],
    }

    creator_q = list(creator_script)
    guesser_q = list(guesser_script)

    def fake_generate(adapter, system, user, *, max_tokens, temp,
                      enable_thinking=None, banned_strings=None):
        role = ("creator" if system is CREATOR_SYSTEM
                else "guesser" if system is GUESSER_SYSTEM else "answerer")
        t.captured["thinking"][role] = enable_thinking
        if system is CREATOR_SYSTEM:
            t.captured["creator_users"].append(user)
            t.captured["creator_banned"].append(banned_strings)
            t.captured["creator_temps"].append(temp)
            return GenResult(text=creator_q.pop(0),
                             prompt_tokens=[1], completion_tokens=[2, 3])
        if system is GUESSER_SYSTEM:
            return GenResult(text=guesser_q.pop(0),
                             prompt_tokens=[4], completion_tokens=[5, 6])
        assert system is ANSWERER_SYSTEM
        return GenResult(text="ANSWER: YES",
                         prompt_tokens=[7], completion_tokens=[8])

    def fake_judge(question, *, enable_thinking=None, system=None):
        t.captured.setdefault("judge_thinking", []).append(enable_thinking)
        t.captured.setdefault("judge_system", []).append(system)
        t.captured["judge"].append(question)
        if "vetting a secret" in question:
            m = re.search(r"Secret: (.+)", question)
            ok = True if validity is None else validity.get(m.group(1).strip(), True)
            return f"VERDICT: {'VALID' if ok else 'INVALID'}"
        assert "scoring how close" in question
        return "CLOSENESS: 4"

    def fake_grpo(adapter, trajs):
        t.captured["grpo"][adapter] = list(trajs)
        return {"n_traj": len(trajs), "n_tokens": 0, "loss": 0.0, "pg": 0.0,
                "kl": 0.0, "grad_norm": 0.0}

    t._generate = fake_generate

    def fake_generate_batch(adapter, system, users, *, max_tokens, temp,
                            completion_batch_size, enable_thinking=None):
        t.captured["generation_batches"].append((
            "guesser" if system is GUESSER_SYSTEM else "answerer", len(users)))
        out = []
        for _user in users:
            if system is GUESSER_SYSTEM:
                out.append(GenResult(
                    text=guesser_q.pop(0), prompt_tokens=[4],
                    completion_tokens=[5, 6]))
            else:
                assert system is ANSWERER_SYSTEM
                out.append(GenResult(
                    text="ANSWER: YES", prompt_tokens=[7],
                    completion_tokens=[8]))
        return out

    t._generate_batch = fake_generate_batch
    t._judge = fake_judge
    t._grpo = fake_grpo
    return t


# Two parseable secrets; guesser wins both episodes at rank 0 (turn 1, turn 2)
# and loses both at rank 1 (budget, format-fail after a wrong guess).
CREATOR_OK = [_secret_json("dog", 0.0), _secret_json("axolotl", 1.0)]
GUESSER_OK = [
    "GUESS: dog",                              # ep A1: guessed turn 1
    "QUESTION: Is it a pet?", "GUESS: dog",    # ep A2: guessed turn 2
    "QUESTION: Is it alive?", "QUESTION: Is it wet?",   # ep B1: budget fail
    "GUESS: frog", "I give up",                # ep B2: wrong guess, format fail
]


# ----- the happy-path iteration ----------------------------------------------

@pytest.fixture()
def record_and_trainer():
    t = _make_trainer(_config(), CREATOR_OK, GUESSER_OK)
    return t.run_iteration(0), t


def test_iteration_record_shape(record_and_trainer):
    rec, _ = record_and_trainer
    assert rec["mode"] == "twentyq" and rec["category"] == "animal"
    assert rec["parse_ok_rate"] == 1.0 and rec["validity_rate"] == 1.0
    assert rec["guess_rates_by_rank"] == [1.0, 0.0]
    assert rec["target_by_rank"] == [0.9, 0.1]
    # rates [1.0, 0.0] vs targets [0.9, 0.1]: mse = 0.01, exp(-4*0.01)
    assert rec["r_gradient"] == pytest.approx(math.exp(-0.04), abs=1e-4)
    assert rec["episodes"] == {
        "total": 4, "guessed": 2,
        "format_ended": 1, "answer_format_fails": 0,
        # Turn-waste telemetry: the scripted guesser never repeats itself.
        "turns": 7, "repeat_turns": 0, "repeat_turn_rate": 0.0,
        "question_retries": 0,
    }
    assert rec["phi_mean"] == pytest.approx(0.4)


def test_solver_trajectories_grouped_and_broadcast(record_and_trainer):
    _, t = record_and_trainer
    trajs = t.captured["grpo"]["B"]            # solver adapter (iteration 0)
    assert len(trajs) == 7                     # 1 + 2 + 2 + 2 guesser turns
    by_secret = {}
    for traj in trajs:
        by_secret.setdefault(traj.meta["secret_id"], []).append(traj)
    assert sorted(len(v) for v in by_secret.values()) == [3, 4]
    for group in by_secret.values():
        # Broadcast: same advantage within an episode; group centered per secret.
        ep_advs = {}
        for traj in group:
            ep_advs.setdefault(traj.reward, set()).add(round(traj.advantage, 9))
        assert all(len(v) == 1 for v in ep_advs.values())
        total = sum(t_.advantage for r, adv in ep_advs.items()
                    for t_ in group if t_.reward == r) / len(group)
        # episode-level advantages (not per-turn) sum to zero over episodes:
        eps = {r: next(iter(a)) for r, a in ep_advs.items()}
        assert sum(eps.values()) == pytest.approx(0.0)


def test_creator_credit_per_secret(record_and_trainer):
    _, t = record_and_trainer
    trajs = t.captured["grpo"]["A"]            # creator adapter
    assert len(trajs) == 2                     # one per secret rollout
    # Both ranks land 0.1 off their target: identical decomposed rewards
    # (grad exp(-0.04) + consistency 0.5 + suite validity 0.1).
    expected = math.exp(-0.04) + 0.5 + 0.1
    for traj in trajs:
        assert traj.reward == pytest.approx(expected, abs=1e-4)
    assert sum(tr.advantage for tr in trajs) == pytest.approx(0.0)


def test_per_role_thinking_flags_threaded(record_and_trainer):
    _, t = record_and_trainer
    # Defaults (q-shakeout-01/02 fix): all roles emit contract output directly;
    # a thinking budget truncated both guesser and creator into unusable output.
    assert t.captured["thinking"] == {
        "creator": False, "guesser": False, "answerer": False}


def test_thinking_flags_are_configurable():
    from twin.config import Config
    cfg = Config.from_dict({"twentyq": {
        "n_secrets": 2, "episodes_per_secret": 2, "max_turns": 2,
        "categories": ["animal"], "creator_thinking": True,
        "guesser_thinking": True}})
    t = _make_trainer(cfg, list(CREATOR_OK), list(GUESSER_OK))
    t.run_iteration(0)
    assert t.captured["thinking"] == {
        "creator": True, "guesser": True, "answerer": False}


def test_judge_graded_without_thinking_by_default(record_and_trainer):
    _, t = record_and_trainer
    # All three NL contracts grade with thinking OFF (q-shakeout-03 fix).
    assert t.captured["judge_thinking"]
    assert all(x is False for x in t.captured["judge_thinking"])


def test_judge_uses_neutral_twentyq_system(record_and_trainer):
    from twin.games.twentyq.prompts import JUDGE_SYSTEM
    from twin.prompts import JUDGE_SYSTEM as MATH_JUDGE_SYSTEM
    _, t = record_and_trainer
    # Neutral twentyq grader, NOT the CAS math grader whose VERDICT contract
    # hijacked CLOSENESS (q-shakeout-04).
    assert t.captured["judge_system"]
    assert all(s is JUDGE_SYSTEM for s in t.captured["judge_system"])
    assert JUDGE_SYSTEM is not MATH_JUDGE_SYSTEM


def test_judge_called_per_contract(record_and_trainer):
    _, t = record_and_trainer
    calls = t.captured["judge"]
    assert sum(1 for q in calls if "vetting a secret" in q) == 2      # validity
    # The creator-answer audit was removed (2026-07-13): no audit calls.
    assert sum(1 for q in calls if "auditing the answers" in q) == 0
    # closeness only for failed episodes (B1, B2) — v1 budget: 1/episode.
    assert sum(1 for q in calls if "scoring how close" in q) == 2


# ----- gates -------------------------------------------------------------------

def test_parse_failed_rank_gets_gate_reward():
    creator = [_secret_json("dog", 0.0), "I could not decide on a secret."]
    guesser = ["GUESS: dog", "QUESTION: Is it a pet?", "GUESS: dog"]
    t = _make_trainer(_config(), creator, guesser)
    rec = t.run_iteration(0)
    assert rec["parse_ok_rate"] == 0.5
    trajs = t.captured["grpo"]["A"]
    assert len(trajs) == 2
    parsed = [tr for tr in trajs if tr.meta["parsed"]]
    gated = [tr for tr in trajs if not tr.meta["parsed"]]
    assert len(parsed) == 1 and len(gated) == 1
    assert gated[0].reward == pytest.approx(-1.0)
    assert parsed[0].reward > gated[0].reward


def test_invalid_secret_plays_no_episodes_and_drags_consistency():
    t = _make_trainer(_config(), CREATOR_OK, GUESSER_OK[:3],
                      validity={"axolotl": False})
    rec = t.run_iteration(0)
    assert rec["validity_rate"] == 0.5
    assert rec["episodes"]["total"] == 2       # only the valid secret played
    assert rec["secrets"][1]["valid"] is False
    assert rec["secrets"][1]["episodes"] == []
    assert rec["secrets"][1]["consistent"] is False
    # invalid rank: unscored (no gradient) + inconsistent => reward far below
    # the calibrated rank's.
    trajs = sorted(t.captured["grpo"]["A"], key=lambda tr: tr.meta["rank"])
    assert trajs[1].reward < trajs[0].reward - 0.5


def test_truthful_creator_trains_all_episodes():
    # No audit/voiding: every non-format-failed episode trains the solver.
    # A1 (instant guess), A2 (guessed turn 2), B1 (budget fail) and B2
    # (wrong guess then a format fail — still a real turn) all survive.
    t = _make_trainer(_config(), CREATOR_OK, GUESSER_OK)
    rec = t.run_iteration(0)
    assert len(t.captured["grpo"]["B"]) == 7   # 1 + 2 + 2 + 2 guesser turns
    # Consistency now tracks validity alone (both secrets valid here).
    assert rec["secrets"][0]["consistent"] is True
    assert rec["secrets"][1]["consistent"] is True


# ----- rolling secret exclusions ---------------------------------------------

def test_recent_prompt_is_category_filtered_and_variants_are_telemetry_only():
    cfg = _config(twentyq={
        "n_secrets": 2, "episodes_per_secret": 2, "max_turns": 2,
        "categories": ["animal"], "recent_secret_window": 4,
    })
    creator = [_secret_json("cat", 0.0), _secret_json("dog", 1.0)]
    t = _make_trainer(cfg, creator, ["GUESS: cat"] * 2 + ["GUESS: dog"] * 2)
    # One deque spans creator A/B; entries retain category so only the active
    # category reaches the prompt.
    t._recent_secrets = deque(
        [("food", "apple"), ("animal", "The Cat")], maxlen=4)

    rec = t.run_iteration(0)

    first, second = t.captured["creator_users"]
    assert "The Cat" in first
    assert "apple" not in first
    assert "neither an exact repeat nor an obvious variant" in first
    assert "- cat" in second                 # current generation is combined
    assert rec["recent_secret_exact_repeat_rate"] == 0.0
    assert rec["recent_secret_normalized_repeat_rate"] == 0.5
    # No rejection/regeneration/reward gate: the normalized repeat still plays.
    assert rec["parse_ok_rate"] == 1.0
    assert rec["episodes"]["total"] == 4


def test_current_generation_exact_repeat_is_logged_but_not_rejected():
    cfg = _config(twentyq={
        "n_secrets": 2, "episodes_per_secret": 2, "max_turns": 2,
        "categories": ["animal"], "recent_secret_window": 128,
    })
    creator = [_secret_json("dog", 0.0), _secret_json("dog", 1.0)]
    t = _make_trainer(cfg, creator, ["GUESS: dog"] * 4)

    rec = t.run_iteration(0)

    assert rec["recent_secret_exact_repeat_rate"] == 0.5
    assert rec["recent_secret_normalized_repeat_rate"] == 0.5
    assert rec["episodes"]["total"] == 4
    assert len(t.captured["grpo"]["A"]) == 2
    # Mode "off" (default): telemetry only — nothing voided, nothing retried.
    assert rec["repeat_handling"] == "off"
    assert rec["repeat_voided"] == 0 and rec["repeat_retries"] == 0
    assert rec["playable_rate"] == 1.0
    assert rec["sampled_secrets"] == ["dog", "dog"]


def test_lockstep_generation_batches_sibling_episodes():
    cfg = _config(twentyq={
        "n_secrets": 2, "episodes_per_secret": 2, "max_turns": 2,
        "categories": ["animal"], "generation_batch_size": 8,
    })
    # Batch-consumption order: all active siblings at turn t, then t+1.
    guesser = [
        "GUESS: dog", "QUESTION: Is it a pet?", "GUESS: dog",
        "QUESTION: Is it alive?", "QUESTION: Is it aquatic?",
        "QUESTION: Does it have gills?", "Unable to determine anything from these answers",
    ]
    t = _make_trainer(cfg, list(CREATOR_OK), guesser)

    rec = t.run_iteration(0)

    assert rec["guess_rates_by_rank"] == [1.0, 0.0]
    assert rec["episodes"]["total"] == 4
    assert t.captured["generation_batches"] == [
        ("guesser", 2), ("answerer", 1), ("guesser", 1),
        ("guesser", 2), ("answerer", 2), ("guesser", 2),
        ("answerer", 1),
    ]


def test_zero_window_keeps_current_generation_exclusion_only():
    cfg = _config(twentyq={
        "n_secrets": 2, "episodes_per_secret": 2, "max_turns": 2,
        "categories": ["animal"], "recent_secret_window": 0,
    })
    creator = [_secret_json("dog", 0.0), _secret_json("dog", 1.0)]
    t = _make_trainer(cfg, creator, ["GUESS: dog"] * 4)

    rec = t.run_iteration(0)

    assert "- dog" in t.captured["creator_users"][1]
    assert rec["recent_secret_history_size"] == 0
    assert rec["recent_secret_exact_repeat_rate"] == 0.5


def test_restore_recent_secrets_is_bounded_category_aware_and_retry_safe(tmp_path):
    cfg = _config(twentyq={
        "n_secrets": 2, "episodes_per_secret": 2, "max_turns": 2,
        "categories": ["animal"], "recent_secret_window": 3,
    })
    t = _make_trainer(cfg, [], [])
    records = [
        {"type": "meta", "run": "demo"},
        {"type": "iteration", "mode": "twentyq", "iter": 0,
         "creator": "A", "category": "animal",
         "secrets": [{"secret": "dog"}]},
        # A retried iteration replaces the earlier row for checkpoint state.
        {"type": "iteration", "mode": "twentyq", "iter": 0,
         "creator": "A", "category": "animal",
         "secrets": [{"secret": "wolf"}]},
        {"type": "iteration", "mode": "twentyq", "iter": 1,
         "creator": "B", "category": "food",
         "secrets": [{"secret": "apple"}]},
        {"type": "iteration", "mode": "twentyq", "iter": 2,
         "creator": "A", "category": "animal",
         "secrets": [{"secret": "cat"}, {"secret": "fox"}]},
    ]
    path = tmp_path / "run.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records)
                    + '{"type":"iteration"')  # crash-truncated tail is ignored

    assert t.restore_recent_secrets(path, before_iteration=2) == 2
    assert list(t._recent_secrets) == [("animal", "wolf"), ("food", "apple")]

    assert t.restore_recent_secrets(path, before_iteration=3) == 3
    assert list(t._recent_secrets) == [
        ("food", "apple"), ("animal", "cat"), ("animal", "fox")]
    assert t._recent_for_category("ANIMAL") == ["cat", "fox"]


# ----- repeat handling: void + masked resample (DESIGN §6.5) -------------------

def _repeat_cfg(mode, **twentyq_extra):
    tq = {"n_secrets": 2, "episodes_per_secret": 2, "max_turns": 2,
          "categories": ["animal"], "recent_secret_window": 128,
          "repeat_handling": mode}
    tq.update(twentyq_extra)
    return _config(twentyq=tq)


def test_repeat_void_mode_voids_without_retry():
    creator = [_secret_json("dog", 0.0), _secret_json("dog", 1.0)]
    t = _make_trainer(_repeat_cfg("void"), creator, ["GUESS: dog"] * 2)

    rec = t.run_iteration(0)

    # Rank 1's repeat plays NO episodes; only rank 0's dog is played.
    assert rec["episodes"]["total"] == 2
    assert rec["repeat_voided"] == 1 and rec["repeat_retries"] == 0
    assert rec["parse_ok_rate"] == 1.0 and rec["playable_rate"] == 0.5
    assert [s["secret"] for s in rec["secrets"]] == ["dog"]
    assert rec["sampled_secrets"] == ["dog", "dog"]
    # Attempt-0 propensity telemetry is unchanged by enforcement.
    assert rec["recent_secret_exact_repeat_rate"] == 0.5
    trajs = t.captured["grpo"]["A"]
    assert len(trajs) == 2
    repeat = next(tr for tr in trajs if tr.meta["repeat"])
    played = next(tr for tr in trajs if not tr.meta["repeat"])
    assert repeat.reward == pytest.approx(0.0)      # repeat_gate default
    assert repeat.advantage < 0 < played.advantage
    # No masked resample in "void" mode: both creator calls unconstrained.
    assert t.captured["creator_banned"] == [None, None]
    # Voided repeats still enter the rolling history (sampling telemetry).
    assert list(t._recent_secrets) == [("animal", "dog"), ("animal", "dog")]


def test_repeat_retry_masked_resample_plays_and_trains():
    creator = [_secret_json("dog", 0.0), _secret_json("dog", 1.0),
               _secret_json("axolotl", 1.0)]           # the masked retry
    t = _make_trainer(_repeat_cfg("retry"), creator,
                      ["GUESS: dog"] * 2 + ["GUESS: axolotl"] * 2)

    rec = t.run_iteration(0)

    assert rec["repeat_handling"] == "retry"
    assert rec["repeat_voided"] == 1
    assert rec["repeat_retries"] == 1 and rec["repeat_retry_playable"] == 1
    assert rec["playable_rate"] == 1.0
    # The retry's secret plays the rank's episodes at the rank's target.
    assert rec["episodes"]["total"] == 4
    assert [s["secret"] for s in rec["secrets"]] == ["dog", "axolotl"]
    assert rec["secrets"][1]["target"] == 0.1
    assert rec["sampled_secrets"] == ["dog", "dog", "axolotl"]
    # The retry reuses the SAME prompt, bans the exclusion list at the logits
    # level, and keeps creator_temp sampling (no greedy collapse onto #2).
    assert t.captured["creator_banned"] == [None, None, ["dog"]]
    assert t.captured["creator_users"][2] == t.captured["creator_users"][1]
    assert t.captured["creator_temps"][2] == t.cfg.gen.creator_temp
    # Creator GRPO group = 3 rollouts: playable, repeat gate, playable retry.
    trajs = t.captured["grpo"]["A"]
    assert len(trajs) == 3
    repeat = next(tr for tr in trajs if tr.meta["repeat"])
    retry = next(tr for tr in trajs if tr.meta["retry"])
    assert not retry.meta["repeat"]
    assert repeat.reward == pytest.approx(0.0)
    assert retry.reward > 0.5                     # real game reward, trained
    assert repeat.advantage < retry.advantage
    assert sum(tr.advantage for tr in trajs) == pytest.approx(0.0)


def test_repeat_retry_variant_slip_voids_rank():
    # The retry emits a case/plural variant that slipped the token ban;
    # guess_matches still catches it, so the rank voids — a repeat NEVER
    # plays episodes.
    creator = [_secret_json("dog", 0.0), _secret_json("dog", 1.0),
               _secret_json("Dogs", 1.0)]
    t = _make_trainer(_repeat_cfg("retry"), creator, ["GUESS: dog"] * 2)

    rec = t.run_iteration(0)

    assert rec["episodes"]["total"] == 2
    assert rec["repeat_voided"] == 2              # attempt AND retry
    assert rec["repeat_retries"] == 1 and rec["repeat_retry_playable"] == 0
    assert rec["playable_rate"] == 0.5
    assert rec["sampled_secrets"] == ["dog", "dog", "Dogs"]
    trajs = t.captured["grpo"]["A"]
    assert len(trajs) == 3
    gated = [tr for tr in trajs if tr.meta["repeat"]]
    assert len(gated) == 2
    assert all(tr.reward == pytest.approx(0.0) for tr in gated)
    assert list(t._recent_secrets) == [
        ("animal", "dog"), ("animal", "dog"), ("animal", "Dogs")]


def test_unknown_repeat_handling_fails_loudly():
    t = _make_trainer(_repeat_cfg("Retry"), [], [])
    with pytest.raises(ValueError, match="repeat_handling"):
        t.run_iteration(0)


def test_repeat_gate_reward_is_configurable():
    creator = [_secret_json("dog", 0.0), _secret_json("dog", 1.0)]
    t = _make_trainer(_repeat_cfg("void", repeat_gate_reward=-0.5),
                      creator, ["GUESS: dog"] * 2)
    t.run_iteration(0)
    repeat = next(tr for tr in t.captured["grpo"]["A"] if tr.meta["repeat"])
    assert repeat.reward == pytest.approx(-0.5)


def test_repeat_gate_catches_misspelled_ban_evasion():
    # Live v4.5 failure mode: the logits ban squeezes mass into one-letter
    # misspellings ('Okpi' for banned 'Okapi'). The gate's repeat_matches
    # (edit-distance-1, 5+ chars) voids them; the masked retry recovers.
    creator = [_secret_json("Okpi", 0.0), _secret_json("Elephant", 0.0),
               _secret_json("dog", 1.0)]
    t = _make_trainer(_repeat_cfg("retry"), creator,
                      ["GUESS: Elephant"] * 2 + ["GUESS: dog"] * 2)
    t._recent_secrets = deque([("animal", "Okapi")], maxlen=128)

    rec = t.run_iteration(0)

    assert rec["repeat_voided"] == 1
    assert rec["repeat_retries"] == 1 and rec["repeat_retry_playable"] == 1
    assert [s["secret"] for s in rec["secrets"]] == ["Elephant", "dog"]
    assert rec["episodes"]["total"] == 4
    assert t.captured["creator_banned"][1] == ["Okapi"]


def test_repeat_check_matches_cross_iteration_history():
    # "The Cat" in rolling history: guess_matches voids a same-normalized
    # attempt-0 pick even though it is not an exact string repeat.
    creator = [_secret_json("cat", 0.0), _secret_json("dog", 1.0)]
    t = _make_trainer(_repeat_cfg("void"), creator, ["GUESS: dog"] * 2)
    t._recent_secrets = deque([("animal", "The Cat")], maxlen=128)

    rec = t.run_iteration(0)

    assert rec["repeat_voided"] == 1
    assert [s["secret"] for s in rec["secrets"]] == ["dog"]
    assert rec["episodes"]["total"] == 2


def test_restore_prefers_sampled_secrets_over_summaries(tmp_path):
    cfg = _config(twentyq={
        "n_secrets": 2, "episodes_per_secret": 2, "max_turns": 2,
        "categories": ["animal"], "recent_secret_window": 8,
    })
    t = _make_trainer(cfg, [], [])
    records = [
        # New-style row: sampled_secrets (voided repeats + retries included)
        # is authoritative over the playable-only summaries.
        {"type": "iteration", "mode": "twentyq", "iter": 0,
         "category": "animal",
         "sampled_secrets": ["dog", "dog", "axolotl"],
         "secrets": [{"secret": "dog"}, {"secret": "axolotl"}]},
        # Legacy row (pre-repeat-handling): summaries are the sample stream.
        {"type": "iteration", "mode": "twentyq", "iter": 1,
         "category": "food",
         "secrets": [{"secret": "apple"}]},
    ]
    path = tmp_path / "run.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))

    assert t.restore_recent_secrets(path) == 4
    assert list(t._recent_secrets) == [
        ("animal", "dog"), ("animal", "dog"), ("animal", "axolotl"),
        ("food", "apple")]


# ----- rotation ----------------------------------------------------------------

def test_rotation_flips_and_control_never_does():
    cfg_rot = _config(roles={"swap_interval": 1, "warmup_iterations": 0})
    recs = []
    for it in range(2):
        t = _make_trainer(cfg_rot, list(CREATOR_OK), list(GUESSER_OK))
        recs.append(t.run_iteration(it))
    assert (recs[0]["creator"], recs[0]["solver"]) == ("A", "B")
    assert (recs[1]["creator"], recs[1]["solver"]) == ("B", "A")

    cfg_ctl = _config(roles={"swap_interval": 0, "warmup_iterations": 0})
    for it in range(2):
        t = _make_trainer(cfg_ctl, list(CREATOR_OK), list(GUESSER_OK))
        rec = t.run_iteration(it)
        assert (rec["creator"], rec["solver"]) == ("A", "B")


def test_fixed_validation_evaluates_both_adapters_without_grpo(tmp_path):
    secret_set = tmp_path / "validation.json"
    secret_set.write_text(json.dumps({
        "name": "test-set",
        "version": 1,
        "secrets": [{
            "secret_id": "fixed-dog",
            "secret": "dog",
            "category": "animal",
            "difficulty": 0.2,
        }],
    }))
    cfg = _config(twentyq={
        "n_secrets": 2,
        "episodes_per_secret": 2,
        "max_turns": 2,
        "categories": ["animal"],
        "credit": "ensemble",
        "w_ensemble": 0.1,
        "validation_every": 1,
        "validation_secret_set": str(secret_set),
    })
    t = _make_trainer(cfg, [], ["GUESS: dog", "GUESS: dog"])
    t._ensemble_potentials = lambda ep, secret: [-2.0, -1.0]

    rec = t.run_validation(1)

    assert set(rec["adapters"]) == {"A", "B"}
    assert all(m["guess_rate"] == 1.0 for m in rec["adapters"].values())
    assert rec["reward_signals"]["terminal_mean"] > 1.0
    assert rec["reward_signals"]["dense_immediate_mean"] == pytest.approx(0.1)
    assert t.captured["grpo"] == {}


# ----- frozen roles + flat difficulty (DESIGN §7) ------------------------------

def _ensemble_cfg(**twentyq):
    """Ensemble credit, so a frozen solver has an expensive scorer to skip."""
    base = {"n_secrets": 2, "episodes_per_secret": 2, "max_turns": 2,
            "categories": ["animal"], "credit": "ensemble", "w_ensemble": 0.1}
    base.update(twentyq)
    return _config(twentyq=base)


def _ensemble_trainer(cfg, **kw):
    t = _make_trainer(cfg, list(CREATOR_OK), list(GUESSER_OK), **kw)
    t.captured["ensemble_calls"] = 0

    def fake_potentials(episodes, secret):
        t.captured["ensemble_calls"] += 1
        return [[-2.0] * (ep.turns_used + 1) for ep in episodes]

    t._ensemble_potentials_many = fake_potentials
    return t


def test_frozen_creator_skips_its_update_but_keeps_its_telemetry():
    frozen = _ensemble_trainer(_ensemble_cfg(freeze_creator=True)).run_iteration(0)
    trained = _ensemble_trainer(_ensemble_cfg()).run_iteration(0)

    # The creator role (adapter A at iteration 0) took no GRPO step...
    assert frozen["creator_update"] == {
        "n_traj": 0, "n_tokens": 0, "loss": 0.0, "pg": 0.0, "kl": 0.0,
        "grad_norm": 0.0, "frozen": True}
    assert frozen["frozen"] == {"creator": True, "solver": False}
    # ...while every measurement it feeds is byte-identical to the trained run.
    for key in ("creator_reward_mean", "r_gradient", "creator_suite_reward",
                "guess_rates_by_rank", "target_by_rank", "validity_rate"):
        assert frozen[key] == trained[key], key
    # The solver is untouched by the creator's freeze.
    assert frozen["solver_update"]["n_traj"] == trained["solver_update"]["n_traj"] > 0


def test_frozen_creator_does_not_call_grpo_for_the_creator_adapter():
    t = _ensemble_trainer(_ensemble_cfg(freeze_creator=True))
    rec = t.run_iteration(0)
    assert rec["creator"] == "A" and rec["solver"] == "B"
    assert set(t.captured["grpo"]) == {"B"}          # A never reached _grpo


def test_frozen_solver_skips_ensemble_scoring_and_its_update():
    t = _ensemble_trainer(_ensemble_cfg(freeze_solver=True))
    rec = t.run_iteration(0)

    assert t.captured["ensemble_calls"] == 0          # the expensive part
    assert set(t.captured["grpo"]) == {"A"}           # creator still trains
    assert rec["solver_update"]["frozen"] is True
    assert rec["n_solver_trajs"] == 0
    assert rec["frozen"] == {"creator": False, "solver": True}
    # Episodes still ran and are still scored — the creator's reward needs them.
    assert rec["episodes"]["total"] == 4 and rec["episodes"]["guessed"] == 2
    assert rec["reward_signals"]["n_episodes"] == 4
    assert rec["reward_signals"]["dense_immediate_total"] == 0.0
    assert rec["reward_signals"]["terminal_mean"] == pytest.approx(
        _ensemble_trainer(_ensemble_cfg()).run_iteration(0)
        ["reward_signals"]["terminal_mean"])


def test_both_frozen_trains_nothing_but_still_plays():
    t = _ensemble_trainer(_ensemble_cfg(freeze_creator=True, freeze_solver=True))
    rec = t.run_iteration(0)
    assert t.captured["grpo"] == {}
    assert t.captured["ensemble_calls"] == 0
    assert rec["episodes"]["total"] == 4
    assert rec["guess_rates_by_rank"] == [1.0, 0.0]


def test_flat_mode_dictates_one_target_to_every_rank():
    cfg = _ensemble_cfg(difficulty_mode="flat", flat_target_rate=0.5)
    t = _ensemble_trainer(cfg)
    rec = t.run_iteration(0)

    assert rec["difficulty_mode"] == "flat" and rec["flat_target_rate"] == 0.5
    assert rec["target_by_rank"] == [0.5, 0.5]
    # Prompts: one shared target, no per-rank difficulty scale, round framing
    # and the exclusion machinery untouched.
    for i, user in enumerate(t.captured["creator_users"]):
        assert f"Pick secret {i + 1} of 2" in user
        assert "Every secret this round has the SAME target" in user
        assert "succeed on about 50% of games" in user
        assert "on a 0-1 scale" not in user
    # Reward follows the mode: rates [1.0, 0.0] vs flat target 0.5.
    assert rec["r_gradient"] == pytest.approx(math.exp(-4 * 0.25), abs=1e-4)


def test_flat_mode_keeps_the_validity_term_despite_zero_difficulty_spread():
    """All ranks share one dictated difficulty, which would trip the suite's
    anti-collapse spread check and silently zero w_valid on every secret."""
    cfg = _ensemble_cfg(difficulty_mode="flat", flat_target_rate=0.5)
    rec = _ensemble_trainer(cfg).run_iteration(0)
    # Per-secret reward = w_grad*exp(-4*(p-0.5)^2) + w_cons*1 + w_valid*1.
    expected = [1.0 * math.exp(-4 * 0.25) + 0.5 + 0.1] * 2
    assert rec["creator_reward_mean"] == pytest.approx(
        sum(expected) / len(expected), abs=1e-4)


def test_gradient_mode_prompt_is_unchanged():
    t = _ensemble_trainer(_ensemble_cfg())
    t.run_iteration(0)
    for user in t.captured["creator_users"]:
        assert "on a 0-1 scale" in user
        assert "Every secret this round" not in user


def test_unknown_difficulty_mode_raises():
    t = _ensemble_trainer(_ensemble_cfg(difficulty_mode="ramp"))
    with pytest.raises(ValueError, match="difficulty_mode"):
        t.run_iteration(0)


# ----- validation statistical power (multi-episode) ----------------------------

def _validation_cfg(tmp_path, **twentyq):
    secret_set = tmp_path / "validation.json"
    secret_set.write_text(json.dumps({
        "name": "test-set", "version": 1,
        "secrets": [{"secret_id": "fixed-dog", "secret": "dog",
                     "category": "animal", "difficulty": 0.2}],
    }))
    # w_ensemble 0.0 with credit=terminal is the real no-ensemble posture, and
    # it is what keeps _ensemble_potentials from loading a 4-model roster in a
    # unit test (see TwentyQTrainer._ensemble_scoring_active).
    base = {"n_secrets": 2, "episodes_per_secret": 2, "max_turns": 3,
            "categories": ["animal"], "credit": "terminal", "w_ensemble": 0.0,
            "validation_every": 1, "validation_secret_set": str(secret_set)}
    base.update(twentyq)
    return _config(twentyq=base)


def test_validation_single_episode_stays_greedy_and_scalar(tmp_path):
    cfg = _validation_cfg(tmp_path, validation_episodes=1)
    t = _make_trainer(cfg, [], ["GUESS: dog"] * 8)
    rec = t.run_validation(1)
    assert rec["decoding"] == "greedy"
    assert rec["episodes_per_secret"] == 1
    assert all(m["n_episodes"] == 1 for m in rec["adapters"].values())


def test_validation_multi_episode_samples_and_reports_ci(tmp_path):
    # One turn per game, alternating right/wrong: a real 0.5 rate on n=4.
    cfg = _validation_cfg(tmp_path, validation_episodes=4, max_turns=1)
    t = _make_trainer(cfg, [], ["GUESS: dog", "GUESS: cat"] * 4)
    rec = t.run_validation(1)

    assert rec["decoding"].startswith("sampled@")
    assert rec["episodes_per_secret"] == 4
    for metrics in rec["adapters"].values():
        assert metrics["n_episodes"] == 4
        lo, hi = metrics["guess_rate_ci95"]
        assert lo < metrics["guess_rate"] < hi
        # 4 samples cannot pin a rate down; the interval must say so.
        assert hi - lo > 0.5


def test_wilson_ci_brackets_rate_and_never_leaves_unit_interval():
    from twin.games.twentyq.trainer import _wilson_ci
    lo, hi = _wilson_ci(3, 24)          # the v6/v7 validation regime
    assert 0.0 <= lo < 3 / 24 < hi <= 1.0
    # The historical instrument's honest width: ~±13 points at n=24.
    assert hi - lo > 0.20
    # Tightens as n grows at the same rate.
    lo2, hi2 = _wilson_ci(24, 192)
    assert (hi2 - lo2) < (hi - lo)
    # Degenerate inputs stay in range rather than producing negative bounds.
    assert _wilson_ci(0, 24)[0] == 0.0
    assert _wilson_ci(24, 24)[1] == 1.0
    assert _wilson_ci(0, 0) == (0.0, 0.0)


# ----- frontier bank as the secret source --------------------------------------

def _bank_file(tmp_path, entries):
    p = tmp_path / "bank.json"
    p.write_text(json.dumps({
        "name": "test-bank", "version": 1,
        "secrets": [
            {"secret_id": f"bank-{i:02d}", "secret": s, "category": c,
             "difficulty": d, "measured_guess_rate": round(1 - d, 3)}
            for i, (s, c, d) in enumerate(entries)
        ],
    }))
    return p


def _bank_cfg(tmp_path, entries, **twentyq):
    base = {"n_secrets": 2, "episodes_per_secret": 2, "max_turns": 2,
            "categories": ["animal", "food"], "credit": "terminal",
            "w_ensemble": 0.0, "secret_source": "bank",
            "freeze_creator": True,
            "bank_path": str(_bank_file(tmp_path, entries))}
    base.update(twentyq)
    return _config(twentyq=base)


BANK_ENTRIES = [("dog", "animal", 0.4), ("cat", "animal", 0.6),
                ("pizza", "food", 0.3), ("bread", "food", 0.7)]


def test_bank_mode_plays_bank_secrets_and_calls_no_creator(tmp_path):
    cfg = _bank_cfg(tmp_path, BANK_ENTRIES)
    # Empty creator script: any creator rollout would IndexError on pop.
    t = _make_trainer(cfg, [], ["GUESS: dog"] * 40)
    rec = t.run_iteration(0)

    assert rec["episodes"]["total"] == 4          # 2 secrets x 2 episodes
    played = {s["secret"] for s in rec["secrets"]}
    assert played <= {e[0] for e in BANK_ENTRIES}
    assert len(played) == 2                       # without replacement
    assert t.captured["creator_users"] == []      # creator never prompted


def test_bank_mode_targets_come_from_measured_difficulty(tmp_path):
    cfg = _bank_cfg(tmp_path, BANK_ENTRIES)
    t = _make_trainer(cfg, [], ["GUESS: dog"] * 40)
    rec = t.run_iteration(0)
    # target == 1 - difficulty, i.e. the MEASURED win rate, not a dictated ramp.
    for summary, target in zip(rec["secrets"], rec["target_by_rank"]):
        expected = {e[0]: round(1 - e[2], 3) for e in BANK_ENTRIES}
        assert target == pytest.approx(expected[summary["secret"]], abs=1e-6)


def test_bank_mode_balances_categories_within_an_iteration(tmp_path):
    cfg = _bank_cfg(tmp_path, BANK_ENTRIES, n_secrets=2)
    cats = []
    for seed in range(8):
        t = _make_trainer(cfg, [], ["GUESS: dog"] * 40)
        t.rng = random.Random(seed)
        rec = t.run_iteration(0)
        cats.append(tuple(sorted(s["category"] for s in rec["secrets"])))
    # Every iteration draws one animal and one food, never two of a kind.
    assert set(cats) == {("animal", "food")}


def test_bank_mode_unbalanced_draw_is_opt_out(tmp_path):
    cfg = _bank_cfg(tmp_path, BANK_ENTRIES, bank_balance_categories=False)
    seen = set()
    for seed in range(12):
        t = _make_trainer(cfg, [], ["GUESS: dog"] * 40)
        t.rng = random.Random(seed)
        rec = t.run_iteration(0)
        seen.add(tuple(sorted(s["category"] for s in rec["secrets"])))
    # Unbalanced sampling can produce same-category pairs; balanced cannot.
    assert any(c[0] == c[1] for c in seen)


def test_bank_mode_uses_each_secrets_own_category_in_the_guesser_prompt(tmp_path):
    cfg = _bank_cfg(tmp_path, BANK_ENTRIES, generation_batch_size=1)
    t = _make_trainer(cfg, [], ["GUESS: dog"] * 40)
    seen_categories = []
    original = t._make_guesser

    def spy(adapter, category, **kw):
        seen_categories.append(category)
        return original(adapter, category, **kw)

    t._make_guesser = spy
    rec = t.run_iteration(0)
    played = {s["secret"]: s["category"] for s in rec["secrets"]}
    assert sorted(seen_categories) == sorted(played.values())
    assert len(set(seen_categories)) == 2     # both categories in ONE iteration


def test_bank_mode_requires_a_frozen_creator(tmp_path):
    cfg = _bank_cfg(tmp_path, BANK_ENTRIES, freeze_creator=False)
    t = _make_trainer(cfg, [], ["GUESS: dog"] * 40)
    with pytest.raises(ValueError, match="requires freeze_creator"):
        t.run_iteration(0)


def test_unknown_secret_source_is_rejected(tmp_path):
    cfg = _bank_cfg(tmp_path, BANK_ENTRIES, secret_source="wishful")
    t = _make_trainer(cfg, [], ["GUESS: dog"] * 40)
    with pytest.raises(ValueError, match="unknown twentyq.secret_source"):
        t.run_iteration(0)


def test_creator_mode_is_unchanged_by_the_bank_addition():
    """The v1..v7 path must be byte-identical when secret_source defaults."""
    t = _make_trainer(_config(), CREATOR_OK, GUESSER_OK)
    rec = t.run_iteration(0)
    assert rec["guess_rates_by_rank"] == [1.0, 0.0]
    assert rec["target_by_rank"] == [0.9, 0.1]
    assert len(t.captured["creator_users"]) == 2


def test_bank_mode_records_mixed_category_honestly(tmp_path):
    cfg = _bank_cfg(tmp_path, BANK_ENTRIES)
    t = _make_trainer(cfg, [], ["GUESS: dog"] * 40)
    rec = t.run_iteration(0)
    # The iteration-level label must not claim a single category it did not play.
    assert rec["category"] == "mixed:animal,food"
    assert {s["category"] for s in rec["secrets"]} == {"animal", "food"}
