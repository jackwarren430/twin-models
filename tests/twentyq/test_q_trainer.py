"""Sprint Q5 — TwentyQTrainer end-to-end through the REAL ``run_iteration``:
secret rollouts with dictated difficulty, parse/validity gates, K scripted
episodes per secret, truthfulness-audit voiding, per-secret creator credit,
broadcast solver advantages, rotation + the no-rotation control. Every model
call and judge call is stubbed; the machinery around them is the real code."""

import math
import random
import re

import pytest

from twin.config import Config
from twin.games.twentyq.prompts import (
    ANSWERER_SYSTEM,
    CREATOR_SYSTEM,
    GUESSER_SYSTEM,
)
from twin.games.twentyq.trainer import TwentyQTrainer
from twin.models.base import GenResult
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
                  validity=None, audit_lies=False):
    """A TwentyQTrainer with real reward/role/episode machinery and scripted
    generation + judging. ``creator_script``/``guesser_script`` are consumed
    in order; the answerer always replies ANSWER: YES. ``validity`` maps
    secret text -> bool (default: everything VALID). ``audit_lies``: the
    audit flags every creator-answered pair False (lying creator)."""
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
    t.captured = {"grpo": {}, "judge": [], "thinking": {}}

    creator_q = list(creator_script)
    guesser_q = list(guesser_script)

    def fake_generate(adapter, system, user, *, max_tokens, temp,
                      enable_thinking=None):
        role = ("creator" if system is CREATOR_SYSTEM
                else "guesser" if system is GUESSER_SYSTEM else "answerer")
        t.captured["thinking"][role] = enable_thinking
        if system is CREATOR_SYSTEM:
            return GenResult(text=creator_q.pop(0),
                             prompt_tokens=[1], completion_tokens=[2, 3])
        if system is GUESSER_SYSTEM:
            return GenResult(text=guesser_q.pop(0),
                             prompt_tokens=[4], completion_tokens=[5, 6])
        assert system is ANSWERER_SYSTEM
        return GenResult(text="ANSWER: YES",
                         prompt_tokens=[7], completion_tokens=[8])

    def fake_judge(question):
        t.captured["judge"].append(question)
        if "vetting a secret" in question:
            m = re.search(r"Secret: (.+)", question)
            ok = True if validity is None else validity.get(m.group(1).strip(), True)
            return f"VERDICT: {'VALID' if ok else 'INVALID'}"
        if "auditing the answers" in question:
            n_pairs = len(re.findall(r"^\d+\. Q:", question, re.MULTILINE))
            tok = "F" if audit_lies else "T"
            return "AUDIT: " + " ".join([tok] * n_pairs)
        assert "scoring how close" in question
        return "CLOSENESS: 4"

    def fake_grpo(adapter, trajs):
        t.captured["grpo"][adapter] = list(trajs)
        return {"n_traj": len(trajs), "n_tokens": 0, "loss": 0.0, "pg": 0.0,
                "kl": 0.0, "grad_norm": 0.0}

    t._generate = fake_generate
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
        "total": 4, "guessed": 2, "void": 0, "unauditable": 0,
        "format_ended": 1, "answer_format_fails": 0,
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


def test_judge_called_per_contract(record_and_trainer):
    _, t = record_and_trainer
    calls = t.captured["judge"]
    assert sum(1 for q in calls if "vetting a secret" in q) == 2      # validity
    # audits only for episodes WITH creator-answered pairs: A2 and B1 (A1 is
    # an instant guess; B2's only pair is the engine-answered wrong guess).
    assert sum(1 for q in calls if "auditing the answers" in q) == 2
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


def test_lying_creator_voids_episodes_and_solver_data():
    t = _make_trainer(_config(), CREATOR_OK, GUESSER_OK, audit_lies=True)
    rec = t.run_iteration(0)
    # Episodes with creator-answered pairs (A2, B1) are voided; A1 (instant
    # guess) and B2 (only an engine-answered wrong guess) have nothing to
    # audit and survive to train the solver: 1 + 2 turn trajectories.
    assert rec["episodes"]["void"] == 2
    assert len(t.captured["grpo"]["B"]) == 3
    assert rec["secrets"][0]["consistent"] is False
    assert rec["secrets"][1]["consistent"] is False


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
