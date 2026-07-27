#!/usr/bin/env python3
"""Is the creator's low secret diversity a PROMPTING limit or a KNOWLEDGE limit?

This decides what to do about the binding constraint. Building validation-v3
took 1200 raw creator draws and yielded only 78 distinct novel candidates (6.5%,
down from 15.7% on the bank-v1 build) once bank-v1 and validation-v2 were held
out. Diversity — not calibration yield — now bounds both the training bank AND
the evaluation set, so it limits training and measurement at the same time.

Two futures follow, and they need opposite work:

  KNOWLEDGE-LIMITED  the base model simply does not know many more animals /
                     foods / household objects that a small solver could ever
                     guess. Then training the creator cannot conjure vocabulary
                     it does not have — RL on the creator would concentrate an
                     already-narrow distribution, making it worse. The answer
                     is more CATEGORIES.

  PROMPTING-LIMITED  the vocabulary is there and the one-at-a-time prompt with
                     a growing exclusion list is failing to reach it. Then
                     creator training is worth doing, because there is headroom
                     for a policy to find.

The discriminating test is cheap: ask for the same entities a different way and
count distinct names. If a different prompt reaches materially more of the
vocabulary, the limit is the prompt.

Arms, all at matched generation budget:
  serial     the production path — one secret per call, rolling exclusions
  listed     one call asks for a numbered list of N distinct entities
  hinted     serial, but each call carries a random subcategory hint

Reports distinct names, and how many are NOVEL against everything already
spoken for (bank-v1 + validation-v2 + validation-v3), which is the number that
actually matters for building the next set.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twin.backends import get_backend  # noqa: E402
from twin.config import Config  # noqa: E402
from twin.games.twentyq.prompts import (  # noqa: E402
    CREATOR_SYSTEM,
    creator_secret_user,
)
from twin.games.twentyq.schema import (  # noqa: E402
    SecretParseError,
    normalize_guess,
    parse_secret,
)
from twin.games.twentyq.trainer import load_validation_secret_set  # noqa: E402

HINTS = {
    "animal": ["that lives in water", "that can fly", "found in a rainforest",
               "kept as a pet", "from Africa", "that is an insect",
               "from the Arctic", "that is a reptile", "that lives underground",
               "found on a farm"],
    "food": ["eaten for breakfast", "that is a vegetable", "from Asia",
             "that is a dessert", "eaten at a restaurant", "that is a fruit",
             "from Mexico", "that is a drink", "made from grain",
             "eaten with the hands"],
    "household object": ["found in a kitchen", "found in a bathroom",
                         "used for cleaning", "made of metal",
                         "found in a garage", "that uses electricity",
                         "found in a bedroom", "used for writing",
                         "found in a garden", "that is furniture"],
}

LIST_PROMPT = (
    "Name {n} different {category}s that would each make a good secret answer "
    "in a game of 21 questions. They must be well-known enough that a player "
    "could plausibly guess them from yes/no questions.\n\n"
    "Reply with a numbered list of {n} lines. Each line is just the name, "
    "nothing else. Do not repeat any of these: {exclusions}."
)


def reserved_names() -> set[str]:
    out: set[str] = set()
    for name in ("data/twentyq-bank-v1.json", "data/twentyq-validation-v2.json",
                 "data/twentyq-validation-v3.json"):
        p = ROOT / name
        if not p.exists():
            continue
        _, secrets = load_validation_secret_set(p)
        out |= {normalize_guess(s.secret) for s in secrets}
    return {k for k in out if k}


def parse_list_reply(text: str) -> list[str]:
    """Names out of a numbered list, tolerant of bullets and stray prose."""
    names = []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^(?:\d+[.)]|[-*•])\s*(.+)$", line)
        if not m:
            continue
        name = m.group(1).strip().strip('"').strip("*").split(" - ")[0]
        name = re.sub(r"\s*\(.*?\)\s*$", "", name).strip()
        if name and len(name) < 40:
            names.append(name)
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/twentyq-v8-bank-solver.yaml")
    ap.add_argument("--calls", type=int, default=60,
                    help="generation calls per arm per category")
    ap.add_argument("--list-size", type=int, default=20)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "tq-runs/creator-diversity-probe.json")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    qcfg = cfg.twentyq
    backend = get_backend(cfg.compute.backend)
    print(f"loading {cfg.model.path}", flush=True)
    base = backend.load_base(cfg.model, cfg.compute)
    backend.seed(args.seed)
    rng = random.Random(args.seed)

    spoken_for = reserved_names()
    print(f"{len(spoken_for)} names already spoken for "
          f"(bank-v1 + validation-v2 + validation-v3)\n", flush=True)

    results: dict = {"calls_per_arm": args.calls, "list_size": args.list_size,
                     "reserved": len(spoken_for), "arms": {}}

    def gen(prompts, max_tokens):
        return base.generate_batch(
            prompts, max_tokens=max_tokens, temp=cfg.gen.creator_temp,
            top_p=cfg.gen.top_p,
            completion_batch_size=qcfg.generation_batch_size,
            suppress_thinking=not qcfg.creator_thinking,
        )

    for arm in ("serial", "listed", "hinted"):
        per_category: dict[str, dict] = {}
        for category in qcfg.categories:
            seen: list[str] = []
            keys: set[str] = set()
            n_raw = 0

            if arm == "listed":
                # Same total budget, spent as fewer, longer completions.
                rounds = max(1, (args.calls * 1) // args.list_size)
                for _ in range(rounds):
                    excl = ", ".join(seen[-40:]) if seen else "none"
                    user = LIST_PROMPT.format(n=args.list_size,
                                              category=category,
                                              exclusions=excl)
                    prompt = base.render(user, system=CREATOR_SYSTEM,
                                         enable_thinking=qcfg.creator_thinking)
                    for g in gen([prompt], 48 * args.list_size):
                        for name in parse_list_reply(g.text):
                            n_raw += 1
                            k = normalize_guess(name)
                            if k and k not in keys:
                                keys.add(k)
                                seen.append(name)
            else:
                done = 0
                while done < args.calls:
                    batch = min(qcfg.generation_batch_size, args.calls - done)
                    prompts = []
                    for i in range(batch):
                        cat = category
                        if arm == "hinted":
                            cat = f"{category} {rng.choice(HINTS[category])}"
                        user = creator_secret_user(
                            cat, rank=0, n_secrets=1, difficulty=0.2,
                            target_rate=0.9, recent=seen[-250:],
                            difficulty_mode=qcfg.difficulty_mode)
                        prompts.append(base.render(
                            user, system=CREATOR_SYSTEM,
                            enable_thinking=qcfg.creator_thinking))
                    for g in gen(prompts, qcfg.secret_max_tokens):
                        n_raw += 1
                        try:
                            sec = parse_secret(g.text)
                        except SecretParseError:
                            continue
                        k = normalize_guess(sec.secret)
                        if k and k not in keys:
                            keys.add(k)
                            seen.append(sec.secret)
                    done += batch

            novel = [n for n in seen if normalize_guess(n) not in spoken_for]
            per_category[category] = {
                "raw": n_raw, "distinct": len(seen), "novel": len(novel),
                "names": seen,
            }
            print(f"  {arm:<7} {category:<18} raw {n_raw:>4}  "
                  f"distinct {len(seen):>3}  novel {len(novel):>3}", flush=True)

        tot_raw = sum(v["raw"] for v in per_category.values())
        tot_d = sum(v["distinct"] for v in per_category.values())
        tot_n = sum(v["novel"] for v in per_category.values())
        results["arms"][arm] = {"by_category": per_category, "raw": tot_raw,
                                "distinct": tot_d, "novel": tot_n}
        print(f"  {arm:<7} TOTAL              raw {tot_raw:>4}  "
              f"distinct {tot_d:>3}  novel {tot_n:>3}\n", flush=True)

    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.out}\n")
    print("VERDICT")
    s = results["arms"]["serial"]
    for arm in ("listed", "hinted"):
        a = results["arms"][arm]
        if s["novel"]:
            print(f"  {arm:<7} novel {a['novel']:>3} vs serial {s['novel']:>3}"
                  f"  = {a['novel'] / s['novel']:.2f}x")
    print("  >1.5x on either arm => PROMPTING-limited (creator training has "
          "headroom)")
    print("  ~1x on both         => KNOWLEDGE-limited (add categories instead)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
