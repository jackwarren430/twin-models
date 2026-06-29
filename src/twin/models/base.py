"""TwinBase: the single shared frozen base model.

One instance of the 6-bit Qwen3-8B MLX model is loaded here and used for three
things, distinguished only by which LoRA adapter is active (see
``twin.models.adapters.Adapters``):

  * Model A  -> adapter "A" active
  * Model B  -> adapter "B" active
  * Oracle / KL reference -> the zeroed adapter (== base behaviour)

This class owns the model+tokenizer and provides role-agnostic primitives:
rendering chat prompts, sampling completions, and computing token log-probs
(needed later for GRPO). It does NOT know about adapters; the caller activates
the desired adapter via ``Adapters`` before calling generate()/logprobs().
"""

import os
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


@dataclass
class GenResult:
    text: str
    prompt_tokens: list[int]
    completion_tokens: list[int]


class TwinBase:
    def __init__(self, model_path: str):
        self.model, self.tokenizer = load(os.path.expanduser(model_path))
        # mlx-lm models default to eval; be explicit so dropout etc. are off
        # for inference paths. Training code re-enables train() as needed.
        self.model.eval()

    # ----- prompting -------------------------------------------------------
    def render(
        self,
        user: str,
        system: str | None = None,
        *,
        enable_thinking: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        """Render a chat prompt to a string using the model's chat template.

        ``enable_thinking`` is a Qwen3 template flag (forwarded as a kwarg);
        kept False by default for throughput (see DESIGN.md §2)."""
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    # ----- generation ------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temp: float = 0.7,
        top_p: float = 0.95,
        seed: int | None = None,
    ) -> GenResult:
        """Sample a completion from the model with the *currently active*
        adapter. Returns text plus the prompt/completion token ids (useful for
        log-prob scoring without re-tokenising)."""
        if seed is not None:
            mx.random.seed(seed)
        sampler = make_sampler(temp=temp, top_p=top_p)
        prompt_tokens = self.tokenizer.encode(prompt)
        text = generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        # NOTE: this re-encodes the decoded text, which is an *approximation* of
        # the exact ids that were sampled (detokenize->encode is not always a
        # round-trip). Fine for Sprint 1. Sprint 3's GRPO will capture the true
        # sampled ids via stream_generate to score the exact trajectory.
        completion_tokens = self.tokenizer.encode(text)
        return GenResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    # ----- scoring (used by GRPO in Sprint 3) ------------------------------
    def token_logprobs(self, token_ids: list[int]) -> mx.array:
        """Per-position log-prob of the *actual next token* for a single
        sequence. Returns shape ``[len(token_ids) - 1]`` where element t is
        ``log p(token_ids[t+1] | token_ids[:t+1])``.

        This is the building block for sequence log-probs and GRPO. Runs a
        single forward pass under whatever adapter is currently active."""
        ids = mx.array(token_ids)[None]            # [1, T]
        logits = self.model(ids)[0]                # [T, V]
        logits = logits.astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        targets = mx.array(token_ids[1:])[:, None]  # [T-1, 1]
        chosen = mx.take_along_axis(logp[:-1], targets, axis=-1)[:, 0]  # [T-1]
        return chosen

    def sequence_logprob(self, prompt_tokens: list[int], completion_tokens: list[int]) -> mx.array:
        """Total log-prob of ``completion_tokens`` given ``prompt_tokens``
        under the active adapter (sum over the completion region)."""
        seq = list(prompt_tokens) + list(completion_tokens)
        per_tok = self.token_logprobs(seq)         # [len(seq)-1]
        start = len(prompt_tokens) - 1             # first completion target index
        return per_tok[start:].sum()

    # ----- oracle ----------------------------------------------------------
    def oracle(self, question: str, *, max_tokens: int = 512, temp: float = 0.2) -> str:
        """Convenience: a base-only factual query. The CALLER is responsible
        for activating the zeroed adapter first (Adapters.activate('base')),
        so this method makes no assumption and just generates."""
        prompt = self.render(
            question,
            system=(
                "You are a precise reference. Answer the question factually and "
                "concisely. If unsure, say so."
            ),
        )
        return self.generate(prompt, max_tokens=max_tokens, temp=temp).text
