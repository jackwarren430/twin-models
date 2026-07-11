#!/usr/bin/env python
"""Targeted probe: does the neutral, non-thinking twentyq judge emit each
contract line correctly with the real MLX base? (q-shakeout-04 fix check.)"""
import sys
from pathlib import Path
ROOT = Path("/Users/jackwarren430/Documents/Programming/RL/twin-models")
sys.path.insert(0, str(ROOT / "src"))

from twin.backends import get_backend
from twin.config import Config
from twin.games.twentyq.judge import (
    judge_answer_audit, judge_closeness, judge_secret_validity)
from twin.games.twentyq.schema import Secret
from twin.games.twentyq.trainer import TwentyQTrainer

cfg = Config.from_yaml(str(ROOT / "configs" / "twentyq-tiny.yaml"))
backend = get_backend(cfg.compute.backend)
print("loading base...", flush=True)
base = backend.load_base(cfg.model, cfg.compute)
adapters = backend.build_adapters(base, cfg.lora)
t = TwentyQTrainer(base, adapters, cfg, backend=backend)

secret = Secret(secret="bread", category="food", difficulty=1.0)
qa = [("Is it a fruit?", "NO"), ("Is it a vegetable?", "NO"),
      ("Is it a grain?", "YES"), ("Is it a legume?", "NO"),
      ("Is it a cereal grain?", "NO"), ("Is it Rice?", "NO")]

print("\n--- validity (expect VALID) ---", flush=True)
v = judge_secret_validity(secret, t._grade)
print("parsed:", v.correct, "|", v.detail)

water = Secret(secret="water", category="food", difficulty=0.0)
print("\n--- validity 'water' as food (expect INVALID) ---", flush=True)
v2 = judge_secret_validity(water, t._grade)
print("parsed:", v2.correct, "|", v2.detail)

print("\n--- audit (expect all True for truthful bread answers) ---", flush=True)
a = judge_answer_audit(secret, qa, t._grade)
print("parsed:", a)

print("\n--- closeness (expect an integer 0-10, NOT None) ---", flush=True)
phi = judge_closeness(secret, qa, t._grade, final_guess="Rice")
print("parsed phi:", phi)

print("\nRESULT:",
      "PASS" if (v.correct and a is not None and phi is not None) else "FAIL",
      flush=True)
