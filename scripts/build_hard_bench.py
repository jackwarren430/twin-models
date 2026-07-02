#!/usr/bin/env python
"""Build the *hard* benchmark tier by adapting open-source benchmarks.

The hand-authored core set (`data/bench/*.json`) is at ceiling for an 8B base —
useless as a training-progress signal. This script pulls genuinely hard items
from established public benchmarks and converts them into our schema + graders,
writing `data/bench/hard/{math,coding,knowledge,reasoning}.json`:

    math       <- MATH-500 (Hendrycks competition math), level>=3, numeric answers
                 -> math_numeric (SymPy verify_math)
    coding     <- MBPP (test split), hidden assert tests
                 -> code (sandboxed verify_code)
    knowledge  <- MMLU-Pro (10-way multiple choice, graduate level)
                 -> mcq
    reasoning  <- BIG-Bench Hard (logical deduction, dates, boolean, ...)
                 -> mcq (letter tasks) / exact (boolean & word tasks)

Data is fetched dependency-free via the HF *datasets-server* JSON API (rows
endpoint), so no `datasets`/`pyarrow` install is needed; the resulting JSON is
committed and the benchmark itself stays dependency-free. These sets are used to
*evaluate* our own model (not train on, not redistribute), the standard internal
use; GPQA is intentionally excluded (gated + contamination canary).

    conda run -n twin-models python scripts/build_hard_bench.py            # all, 20/category
    conda run -n twin-models python scripts/build_hard_bench.py --per-category 30 --seed 1
    conda run -n twin-models python scripts/build_hard_bench.py --only math,coding

Then measure where the base lands:
    conda run -n twin-models python scripts/benchmark.py --config configs/base.yaml \
        --data data/bench/hard
"""

import argparse
import json
import random
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twin.verifiers.math_verifier import _is_seq, _parse  # noqa: E402

OUT_DIR = ROOT / "data" / "bench" / "hard"
API = "https://datasets-server.huggingface.co/rows"


# --------------------------------------------------------------------------- #
# HF datasets-server fetch (no deps)
# --------------------------------------------------------------------------- #
def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "twin-bench-build"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def fetch_window(dataset: str, config: str, split: str, offset: int, length: int) -> dict:
    q = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split,
         "offset": offset, "length": min(length, 100)}
    )
    return _get(f"{API}?{q}")


def fetch_pool(dataset: str, config: str, split: str, *, want: int, seed: int) -> list[dict]:
    """Grab a diverse row pool: first probe for the total, then pull windows at
    random offsets (datasets are often grouped, so sequential reads are skewed)."""
    head = fetch_window(dataset, config, split, 0, 100)
    total = head.get("num_rows_total") or len(head["rows"])
    rows = [r["row"] for r in head["rows"]]
    rng = random.Random(seed)
    # pull extra windows until we have a comfortable pool (>= 6x want, capped)
    target_pool = min(total, max(want * 6, 200))
    seen_offsets = {0}
    guard = 0
    while len(rows) < target_pool and guard < 40:
        guard += 1
        off = rng.randrange(0, max(1, total - 1))
        off = (off // 100) * 100
        if off in seen_offsets:
            continue
        seen_offsets.add(off)
        try:
            w = fetch_window(dataset, config, split, off, 100)
        except Exception as e:  # noqa: BLE001
            print(f"   (window {off} failed: {type(e).__name__}); continuing", file=sys.stderr)
            continue
        rows.extend(r["row"] for r in w["rows"])
    return rows


# --------------------------------------------------------------------------- #
# converters: source row -> our BenchItem dict (or None to skip)
# --------------------------------------------------------------------------- #
def _math_answer_numeric(ans: str) -> bool:
    """Keep only answers SymPy can grade as a number or tuple of numbers (drops
    \\sqrt, \\pi, intervals, matrices, text — which verify_math can't compare)."""
    try:
        p = _parse(ans)
    except Exception:  # noqa: BLE001
        return False
    try:
        if _is_seq(p):
            return len(list(p)) > 0 and all(not getattr(e, "free_symbols", set()) for e in p)
        return not getattr(p, "free_symbols", set())
    except Exception:  # noqa: BLE001
        return False


def conv_math(row: dict) -> dict | None:
    if int(row.get("level", 0) or 0) < 3:
        return None
    ans = str(row.get("answer", "")).strip()
    if not _math_answer_numeric(ans):
        return None
    uid = str(row.get("unique_id", "")).replace("/", "_").replace(".json", "")
    return {
        "id": f"math500-{uid}" if uid else None,
        "subcategory": str(row.get("subject", "")),
        "prompt": str(row["problem"]).strip(),
        "answer": ans,
        "verification": {"type": "math_numeric"},
        "metadata": {"source": "MATH-500", "level": row.get("level")},
    }


_ENTRY_RE = re.compile(r"assert\s+([A-Za-z_]\w*)\s*\(")


def conv_mbpp(row: dict) -> dict | None:
    tests = list(row.get("test_list", []) or [])
    if not tests:
        return None
    desc = str(row.get("text") or row.get("prompt") or "").strip()
    setup = str(row.get("test_setup_code", "") or "")
    m = _ENTRY_RE.search(tests[0])
    entry = m.group(1) if m else ""
    prompt = desc
    if entry:
        prompt += f"\n\nName the function `{entry}`."
    return {
        "id": f"mbpp-{row.get('task_id')}",
        "subcategory": "mbpp",
        "prompt": prompt,
        "entry_point": entry,
        "verification": {"type": "code", "tests": "\n".join(tests), "setup": setup},
        "metadata": {"source": "MBPP", "task_id": row.get("task_id")},
    }


def conv_mmlu_pro(row: dict) -> dict | None:
    options = list(row.get("options", []) or [])
    answer = str(row.get("answer", "")).strip().upper()
    if len(options) < 2 or not answer:
        return None
    if answer not in {chr(ord("A") + i) for i in range(len(options))}:
        return None
    return {
        "id": f"mmlupro-{row.get('question_id')}",
        "subcategory": str(row.get("category", "")),
        "prompt": str(row["question"]).strip(),
        "choices": [str(o) for o in options],
        "answer": answer,
        "verification": {"type": "mcq"},
        "metadata": {"source": "MMLU-Pro", "category": row.get("category")},
    }


_BBH_LETTER = re.compile(r"^\(([A-Z])\)$")


def conv_bbh(row: dict, task: str, idx: int) -> dict | None:
    inp = str(row.get("input", "")).strip()
    target = str(row.get("target", "")).strip()
    if not inp or not target:
        return None
    base = {"id": f"bbh-{task}-{idx}", "subcategory": task,
            "metadata": {"source": "BBH", "task": task}}
    m = _BBH_LETTER.match(target)
    if m:
        # a "(X)" answer is multiple-choice: only usable if we can parse the
        # option list into proper choices, else it can't be graded -> skip.
        if "Options:" not in inp:
            return None
        stem, _, opts_block = inp.partition("Options:")
        choices, letters = [], []
        for line in opts_block.splitlines():
            mo = re.match(r"\s*\(([A-Z])\)\s*(.*)", line)
            if mo:
                letters.append(mo.group(1))
                choices.append(mo.group(2).strip())
        if (len(choices) >= 2
                and letters == [chr(ord("A") + i) for i in range(len(choices))]
                and m.group(1) in letters):
            return {**base, "prompt": stem.strip(), "choices": choices,
                    "answer": m.group(1), "verification": {"type": "mcq"}}
        return None
    # boolean / word / number answers -> exact (normalized match handles case/parens)
    if len(target) > 40:
        return None  # long free-form targets don't grade reliably as exact
    return {**base, "prompt": inp, "answer": target, "verification": {"type": "exact"}}


# BBH tasks chosen to grade cleanly (letter-MCQ or short boolean/word/number).
BBH_TASKS = [
    "logical_deduction_three_objects", "logical_deduction_five_objects",
    "date_understanding", "reasoning_about_colored_objects",
    "temporal_sequences", "tracking_shuffled_objects_three_objects",
    "boolean_expressions", "causal_judgement", "formal_fallacies",
    "navigate", "web_of_lies", "object_counting",
]


# --------------------------------------------------------------------------- #
# build one category
# --------------------------------------------------------------------------- #
def _sample(items: list[dict], k: int, seed: int) -> list[dict]:
    items = [it for it in items if it]
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:k]


def build_math(k, seed):
    rows = fetch_pool("HuggingFaceH4/MATH-500", "default", "test", want=k, seed=seed)
    return _sample([conv_math(r) for r in rows], k, seed)


def build_coding(k, seed):
    rows = fetch_pool("google-research-datasets/mbpp", "full", "test", want=k, seed=seed)
    return _sample([conv_mbpp(r) for r in rows], k, seed)


def build_knowledge(k, seed):
    rows = fetch_pool("TIGER-Lab/MMLU-Pro", "default", "test", want=k, seed=seed)
    return _sample([conv_mmlu_pro(r) for r in rows], k, seed)


def build_reasoning(k, seed):
    rng = random.Random(seed)
    per_task = max(2, (k // max(1, len(BBH_TASKS))) + 2)
    pool: list[dict] = []
    for task in BBH_TASKS:
        try:
            head = fetch_window("lukaemon/bbh", task, "test", 0, 100)
        except Exception as e:  # noqa: BLE001
            print(f"   (BBH task {task} failed: {type(e).__name__})", file=sys.stderr)
            continue
        rows = [r["row"] for r in head["rows"]]
        rng.shuffle(rows)
        got = 0
        for i, r in enumerate(rows):
            it = conv_bbh(r, task, i)
            if it:
                pool.append(it)
                got += 1
            if got >= per_task:
                break
    return _sample(pool, k, seed)


BUILDERS = {
    "math": build_math,
    "coding": build_coding,
    "knowledge": build_knowledge,
    "reasoning": build_reasoning,
}

SOURCES = {
    "math": "MATH-500 (Hendrycks et al.), MIT license — competition math, level>=3, numeric answers only.",
    "coding": "MBPP (Austin et al.), CC-BY-4.0 — Python function tasks graded by hidden assert tests.",
    "knowledge": "MMLU-Pro (TIGER-Lab), MIT license — 10-way multiple-choice, graduate level.",
    "reasoning": "BIG-Bench Hard (Suzgun et al.), Apache-2.0 — logical deduction / dates / boolean / counting.",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-category", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None, help="comma list: math,coding,knowledge,reasoning")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cats = args.only.split(",") if args.only else list(BUILDERS)

    for cat in cats:
        if cat not in BUILDERS:
            ap.error(f"unknown category {cat!r}")
        print(f"building {cat} (target {args.per_category}) ...", flush=True)
        items = BUILDERS[cat](args.per_category, args.seed)
        # ensure ids exist + unique
        for i, it in enumerate(items):
            if not it.get("id"):
                it["id"] = f"{cat}_hard-{i:03d}"
        payload = {
            "category": cat,
            "description": f"HARD tier — {SOURCES[cat]}",
            "items": items,
        }
        path = out_dir / f"{cat}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"  wrote {len(items)} items -> {path}")
    print("\nNext: conda run -n twin-models python scripts/benchmark.py "
          f"--config configs/base.yaml --data {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
