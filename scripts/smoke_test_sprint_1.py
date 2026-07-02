"""Sprint 1 end-to-end smoke test (loads the real 6-bit Qwen3-8B base).

Validates the core mechanism claimed in DESIGN.md §3:
  1. base loads; chat template renders; greedy generation runs
  2. one base, two LoRA adapter trees, swapped via model.update()
  3. zeroed adapter ('base') reproduces base behaviour exactly  (oracle/ref)
  4. a perturbed adapter A changes the model's logits; B stays == base
  5. swapping is independent (perturbing A does not affect B)
  6. save(A) / load into B round-trips exactly

Run:  conda run -n twin-models python scripts/smoke_test.py
Exits non-zero on any failed invariant.
"""

import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twin.config import Config                      # noqa: E402
from twin.models import Adapters, TwinBase          # noqa: E402

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
failures = 0


def check(name: str, ok: bool, detail: str = ""):
    global failures
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures += 1


def perturb_b(tree, sigma=0.02):
    """Return a copy of `tree` with all lora_b leaves set to small random
    values (simulating a trained adapter; lora_b starts at zero)."""
    flat = dict(tree_flatten(tree))
    for k in list(flat):
        if k.endswith("lora_b"):
            flat[k] = mx.random.normal(flat[k].shape) * sigma
    return tree_unflatten(list(flat.items()))


def maxdiff(a, b) -> float:
    return float(mx.max(mx.abs(a - b)))


def main():
    cfg = Config.from_yaml(ROOT / "configs" / "tiny.yaml")
    print(f"Loading base: {cfg.model.path}")
    base = TwinBase(cfg.model.path)
    print("  loaded.\n")

    print("== generation sanity ==")
    prompt = base.render("In one short sentence, what is reinforcement learning?")
    out = base.generate(prompt, max_tokens=40, temp=0.0)  # greedy
    check("greedy generation produced text", len(out.text.strip()) > 0,
          detail=out.text.strip()[:80].replace("\n", " "))

    print("\n== attach two LoRA adapters over one base ==")
    ad = Adapters.from_config(base.model, cfg.lora)
    nparams = ad.num_params("A")
    check("adapter has trainable params", nparams > 0, detail=f"{nparams:,} params/adapter")

    # Fixed token sequence for deterministic logit comparison.
    ids = base.tokenizer.encode(base.render("What is 2+2? Answer with a number."))

    ad.activate("A")
    lp_A_init = base.token_logprobs(ids)
    ad.activate("base")
    lp_base = base.token_logprobs(ids)
    ad.activate("B")
    lp_B_init = base.token_logprobs(ids)

    print("\n== invariant: at init, A == B == base (lora_b is zero) ==")
    check("A(init) == base", maxdiff(lp_A_init, lp_base) < 1e-4, detail=f"maxdiff={maxdiff(lp_A_init, lp_base):.2e}")
    check("B(init) == base", maxdiff(lp_B_init, lp_base) < 1e-4, detail=f"maxdiff={maxdiff(lp_B_init, lp_base):.2e}")

    print("\n== perturb adapter A only ==")
    ad.trees["A"] = perturb_b(ad.trees["A"], sigma=0.03)
    mx.eval(ad.trees["A"])
    ad.activate("A")
    lp_A = base.token_logprobs(ids)
    ad.activate("B")
    lp_B = base.token_logprobs(ids)
    ad.activate("base")
    lp_base2 = base.token_logprobs(ids)

    check("perturbed A differs from base", maxdiff(lp_A, lp_base) > 1e-3, detail=f"maxdiff={maxdiff(lp_A, lp_base):.2e}")
    check("B unchanged (still == base)", maxdiff(lp_B, lp_base) < 1e-4, detail=f"maxdiff={maxdiff(lp_B, lp_base):.2e}")
    check("base adapter stable across swaps", maxdiff(lp_base, lp_base2) < 1e-6)

    print("\n== using('base') context manager restores previous adapter ==")
    ad.activate("A")
    with ad.using("base"):
        inside = base.token_logprobs(ids)
    after = base.token_logprobs(ids)
    check("inside using('base') == base", maxdiff(inside, lp_base) < 1e-4)
    check("after context, A restored", maxdiff(after, lp_A) < 1e-6, detail=f"active={ad.active}")

    print("\n== save(A) / load into B round-trips ==")
    ckpt = ROOT / "checkpoints"
    ckpt.mkdir(exist_ok=True)
    p = str(ckpt / "_smoke_A.safetensors")
    ad.save("A", p)
    ad.load("B", p)
    ad.activate("B")
    lp_B_loaded = base.token_logprobs(ids)
    check("loaded B == perturbed A", maxdiff(lp_B_loaded, lp_A) < 1e-5, detail=f"maxdiff={maxdiff(lp_B_loaded, lp_A):.2e}")
    Path(p).unlink(missing_ok=True)

    print()
    if failures:
        print(f"{FAIL}: {failures} invariant(s) failed.")
        sys.exit(1)
    print(f"{PASS}: all Sprint-1 invariants hold.")


if __name__ == "__main__":
    main()
