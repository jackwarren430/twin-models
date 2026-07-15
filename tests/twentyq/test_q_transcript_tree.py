"""The GRPO-structured transcript tree (twin.log.transcript_tree). Pure I/O over
already-computed values — mlx-free, so it runs anywhere. Verifies the folder
layout (run → iter → creator GRPO member → solver GRPO member) and that edge
statuses (INVALID / PARSE-FAIL) render as childless member folders."""

from twin.games.twentyq.episode import Episode, Turn
from twin.games.twentyq.rewards import QEpisodeReward
from twin.games.twentyq.schema import Secret
from twin.log.transcript_tree import TwentyQTranscriptTree, _episode_tag, _slug


def _secret(text="dog"):
    return Secret(secret=text, category="animal", difficulty=0.0, notes="n")


def _turn(index, kind, content, answer, raw, answer_raw=""):
    return Turn(index=index, kind=kind, content=content, answer=answer,
                raw_text=raw, prompt_tokens=[1], completion_tokens=[2],
                answer_raw=answer_raw, creator_answered=True)


def _win_episode():
    return Episode(
        secret=_secret(),
        turns=[_turn(0, "question", "Is it a mammal?", "YES",
                     "QUESTION: Is it a mammal?", "ANSWER: YES"),
               _turn(1, "guess", "dog", None, "GUESS: dog")],
        guessed=True, ended="guessed")


def _miss_episode():
    return Episode(
        secret=_secret(),
        turns=[_turn(0, "question", "Is it big?", "NO",
                     "QUESTION: Is it big?", "ANSWER: NO")],
        guessed=False, ended="budget")


def _member_ok():
    ep0, ep1 = _win_episode(), _miss_episode()
    rew = QEpisodeReward(total=1.3, guessed=True, r_efficiency=0.6,
                         phi_final=None, format_fail=False)
    return {
        "rank": 0, "status": "ok", "secret": "dog", "difficulty": 0.0,
        "valid": True, "guess_rate": 0.5, "consistent": True, "target": 0.9,
        "system": "SYS", "user": "pick a secret", "completion": '{"secret":"dog"}',
        "parse_error": None, "creator_reward": 1.2, "creator_advantage": 0.3,
        "episodes": [
            {"ep": ep0, "reward": 1.3, "reward_obj": rew, "phi": None,
             "potentials": [-3.0, -2.1, -1.0], "returns": [0.9, 0.4],
             "step_rewards": [
                 {"dense": 0.9, "terminal": 0.0, "total": 0.9,
                  "return_": 0.9, "potential_before": -3.0,
                  "potential_after": -2.1},
                 {"dense": 1.1, "terminal": 1.3, "total": 2.4,
                  "return_": 0.4, "potential_before": -2.1,
                  "potential_after": -1.0},
             ],
             "adv_by_turn": {0: 0.2, 1: -0.1},
             "prompts": {
                 0: {"guesser_system": "GUESS-SYS", "guesser_user": "GUESS-USER-0",
                     "answerer_system": "ANS-SYS", "answerer_user": "ANS-USER-0"},
                 1: {"guesser_system": "GUESS-SYS", "guesser_user": "GUESS-USER-1"}}},
            {"ep": ep1, "reward": 0.0,
             "reward_obj": QEpisodeReward(0.0, False, 0.0, None, False),
             "phi": None, "potentials": [-3.0, -2.8], "returns": [0.1],
             "adv_by_turn": {0: 0.0},
             "prompts": {0: {"guesser_system": "GUESS-SYS", "guesser_user": "GUESS-USER",
                             "answerer_system": "ANS-SYS", "answerer_user": "ANS-USER"}}},
        ],
    }


def _member_invalid():
    return {"rank": 1, "status": "invalid", "secret": "water", "difficulty": 0.5,
            "valid": False, "guess_rate": 0.0, "consistent": False, "target": 0.5,
            "system": "SYS", "user": "pick", "completion": '{"secret":"water"}',
            "parse_error": None, "creator_reward": 0.1, "creator_advantage": -0.2,
            "episodes": []}


def _member_parsefail():
    return {"rank": 2, "status": "parse_fail", "secret": None, "difficulty": None,
            "valid": None, "guess_rate": None, "consistent": None, "target": 0.1,
            "system": "SYS", "user": "pick", "completion": "I cannot decide",
            "parse_error": "no JSON object found", "creator_reward": -1.0,
            "creator_advantage": -0.1, "episodes": []}


def _write(root, **overrides):
    tree = TwentyQTranscriptTree(root, meta={"run": "demo", "credit": "ensemble"})
    kw = dict(iteration=0, category="animal", creator="A", solver="B",
              swapped=False, credit="ensemble",
              members=[_member_ok(), _member_invalid(), _member_parsefail()],
              aggregate={"guess_rate_mean": 0.25, "r_gradient": 0.5,
                         "episodes": {"guessed": 1, "total": 3, "format_ended": 0}})
    kw.update(overrides)
    tree.write_iteration(**kw)
    return root


# ----- helpers ---------------------------------------------------------------

def test_slug_and_episode_tag():
    assert _slug("household object!!") == "household_object"
    assert _slug("") == "unnamed"
    assert _episode_tag(_win_episode()) == "win_t2"
    assert _episode_tag(_miss_episode()) == "miss"
    fmt = Episode(secret=_secret(), turns=[], guessed=False, ended="format")
    assert _episode_tag(fmt) == "fmtfail"


# ----- structure -------------------------------------------------------------

def test_run_and_iter_scaffold(tmp_path):
    root = _write(tmp_path / "run.transcript")
    assert (root / "_run.md").is_file()
    assert (root / "iter_00" / "_iter.md").is_file()


def test_creator_member_folders_named_by_rank_and_status(tmp_path):
    root = _write(tmp_path / "run.transcript")
    it = root / "iter_00"
    assert (it / "creator_0__dog").is_dir()
    assert (it / "creator_1__water__INVALID").is_dir()
    assert (it / "creator_2__PARSE-FAIL").is_dir()
    for d in ("creator_0__dog", "creator_1__water__INVALID", "creator_2__PARSE-FAIL"):
        assert (it / d / "_creator.md").is_file()


def test_solver_members_are_outcome_tagged_episode_files(tmp_path):
    root = _write(tmp_path / "run.transcript")
    dog = root / "iter_00" / "creator_0__dog"
    eps = sorted(p.name for p in dog.glob("episode_*.md"))
    assert eps == ["episode_0__win_t2.md", "episode_1__miss.md"]
    # Childless members (no episodes) carry only _creator.md.
    assert list((root / "iter_00" / "creator_2__PARSE-FAIL").glob("episode_*.md")) == []


def test_episode_file_shows_per_turn_credit_and_transcript(tmp_path):
    root = _write(tmp_path / "run.transcript")
    txt = (root / "iter_00" / "creator_0__dog" / "episode_0__win_t2.md").read_text()
    assert "## solver GRPO member" in txt
    assert "## per-turn credit" in txt
    assert "GUESS: dog" in txt            # the guesser's raw completion
    assert "ANSWER: YES" in txt           # the answerer's reply
    assert "0.900" in txt and "0.200" in txt   # a return and an advantage
    assert "dense" in txt and "terminal" in txt and "assigned return" in txt
    assert "**reward**" in txt             # inline on each turn, not table-only


def test_all_prompts_and_outputs_visible(tmp_path):
    # Full visibility: every input prompt (guesser + answerer, system + user) and
    # every model output appears in the episode file, prompts in <details>.
    root = _write(tmp_path / "run.transcript")
    txt = (root / "iter_00" / "creator_0__dog" / "episode_0__win_t2.md").read_text()
    assert "guesser SYSTEM prompt" in txt and "guesser USER prompt" in txt
    assert "GUESS-USER-0" in txt and "GUESS-USER-1" in txt   # per-turn guesser input
    assert "answerer SYSTEM prompt" in txt and "ANS-USER-0" in txt
    assert "**guesser output**" in txt and "**answerer output**" in txt
    assert "<details>" in txt
    # A wrong-guess turn would be an engine referee; here turn 1 is a winning
    # guess (episode ends), so no answerer block on it.
    creator = (root / "iter_00" / "creator_0__dog" / "_creator.md").read_text()
    assert "creator SYSTEM prompt" in creator and "creator USER prompt" in creator


def test_creator_file_shows_grpo_reward_and_advantage(tmp_path):
    root = _write(tmp_path / "run.transcript")
    txt = (root / "iter_00" / "creator_0__dog" / "_creator.md").read_text()
    assert "creator reward" in txt and "1.200" in txt
    assert "creator advantage" in txt and "0.300" in txt
    parse = (root / "iter_00" / "creator_2__PARSE-FAIL" / "_creator.md").read_text()
    assert "no JSON object found" in parse


def test_engine_referee_answer_has_no_answerer_block(tmp_path):
    # A wrong GUESS is answered by the engine (ground-truth NO), not a model —
    # it must render as an engine-referee note, not an answerer input/output.
    wrong = Turn(index=0, kind="guess", content="cat", answer="NO",
                 raw_text="GUESS: cat", prompt_tokens=[1], completion_tokens=[2],
                 answer_raw="", creator_answered=False)
    ep = Episode(secret=_secret(), turns=[wrong], guessed=False, ended="budget")
    m = _member_ok()
    m["episodes"] = [{"ep": ep, "reward": 0.0,
                      "reward_obj": QEpisodeReward(0.0, False, 0.0, None, False),
                      "phi": None, "prompts": {0: {"guesser_user": "u"}}}]
    root = _write(tmp_path / "run.transcript", members=[m])
    txt = next((root / "iter_00").glob("creator_0__dog/episode_*.md")).read_text()
    assert "engine referee" in txt
    assert "**answerer output**" not in txt


def test_validation_tree_contains_per_turn_rewards(tmp_path):
    tree = TwentyQTranscriptTree(tmp_path / "run.transcript", meta={"run": "demo"})
    entry = dict(_member_ok()["episodes"][0])
    entry.update(secret="dog", category="animal")
    aggregate = {
        "adapters": {"A": {
            "guessed": 1,
            "n_episodes": 1,
            "guess_rate": 1.0,
            "reward_signals": {
                "terminal_mean": 1.3,
                "dense_immediate_mean": 2.0,
                "combined_return_start_mean": 0.9,
            },
        }},
    }
    tree.write_validation(
        step=5,
        secret_set={"name": "fixed-v1"},
        answerer="base",
        adapters={"A": [entry]},
        aggregate=aggregate,
    )
    root = tmp_path / "run.transcript" / "validation_step_0005"
    assert (root / "_validation.md").is_file()
    txt = next((root / "adapter_a").glob("secret_*.md")).read_text()
    assert "validation_ensemble" in txt
    assert "**reward**" in txt
    assert "dense=0.900" in txt
