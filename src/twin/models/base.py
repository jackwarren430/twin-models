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
from typing import Callable, Optional

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler


@dataclass
class GenResult:
    text: str
    prompt_tokens: list[int]
    completion_tokens: list[int]


@dataclass
class ReactResult:
    """A multi-turn (ReAct) rollout: the model's text interleaved with injected
    tool observations, plus the token bookkeeping GRPO needs. ``completion_tokens``
    is the FULL spliced sequence (policy tokens + injected ``<obs>`` tokens), so
    the forward pass conditions on the observations; ``loss_mask`` marks which of
    those are policy-sampled (1) vs injected (0)."""

    text: str
    prompt_tokens: list[int]
    completion_tokens: list[int]
    loss_mask: list[int]
    n_rounds: int = 0
    n_tool_calls: int = 0


def assemble_react(segments: list[tuple[list[int], bool]]) -> tuple[list[int], list[int]]:
    """Flatten ``(tokens, is_policy)`` segments into ``(completion_ids,
    loss_mask)``. Pure bookkeeping, factored out so the masking is unit-testable
    without loading the model."""
    ids: list[int] = []
    mask: list[int] = []
    for tokens, is_policy in segments:
        ids.extend(tokens)
        mask.extend([1 if is_policy else 0] * len(tokens))
    return ids, mask


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
        tools: list[dict] | None = None,
    ) -> str:
        """Render a chat prompt to a string using the model's chat template.

        ``enable_thinking`` is a Qwen3 template flag (forwarded as a kwarg);
        kept False by default for throughput (see DESIGN.md §2).
        ``tools`` (JSON function signatures, e.g. ``twin.tools.tool_schemas``)
        switches the template into native function-calling mode: it declares
        the tools in the system block and the model emits ``<tool_call>``
        JSON — the protocol Qwen3 was trained on (Sprint 7)."""
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        kwargs: dict = {}
        if tools:
            kwargs["tools"] = tools
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            **kwargs,
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
        adapter. Returns text plus the prompt/completion token ids.

        The completion ids are the **exact** token ids that were sampled
        (captured from ``stream_generate``), NOT a re-encoding of the decoded
        text — so GRPO (Sprint 3) scores the true trajectory it generated. The
        EOS that ends generation is included if the model emitted it."""
        if seed is not None:
            mx.random.seed(seed)
        sampler = make_sampler(temp=temp, top_p=top_p)
        prompt_tokens = self.tokenizer.encode(prompt)
        completion_tokens: list[int] = []
        pieces: list[str] = []
        for resp in stream_generate(
            self.model,
            self.tokenizer,
            prompt_tokens,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            completion_tokens.append(int(resp.token))
            pieces.append(resp.text)
        return GenResult(
            text="".join(pieces),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int = 512,
        temp: float = 0.7,
        top_p: float = 0.95,
        seed: int | None = None,
        completion_batch_size: int = 32,
    ) -> list[GenResult]:
        """Sample completions for ``prompts`` in ONE continuous-batching pass
        under the *currently active* adapter (Sprint 8: batched solver
        generation — decode is memory-bandwidth-bound on Apple silicon, so a
        batch of B reuses each weight read ~B times).

        Semantics match :meth:`generate` per prompt: ``completion_tokens`` are
        the exact sampled ids (the terminal EOS included when the model
        emitted one); ``text`` is the decoded completion without the EOS.
        The RNG *stream* differs from B sequential calls — same sampling
        distribution, different draws — so a batched run is statistically,
        not bitwise, equivalent to a sequential one (documented in
        DESIGN_V2.md; the solve-rate measurement is unaffected).

        ``completion_batch_size`` caps concurrent decode sequences: KV cache
        is ~150KB/token for Qwen3-8B, so B sequences at a 4096 budget
        worst-case ~0.6GB each — keep B modest on 32GB."""
        if seed is not None:
            mx.random.seed(seed)
        sampler = make_sampler(temp=temp, top_p=top_p)
        gen = BatchGenerator(
            self.model,
            stop_tokens=[[t] for t in self.tokenizer.eos_token_ids],
            sampler=sampler,
            completion_batch_size=completion_batch_size,
        )
        prompt_ids = [self.tokenizer.encode(p) for p in prompts]
        uids = gen.insert(prompt_ids, [max_tokens] * len(prompt_ids))
        toks: dict = {uid: [] for uid in uids}
        try:
            while responses := gen.next_generated():
                for r in responses:
                    toks[r.uid].append(int(r.token))
        finally:
            gen.close()
        eos = set(self.tokenizer.eos_token_ids)
        out: list[GenResult] = []
        for pid, uid in zip(prompt_ids, uids):
            ids = toks[uid]
            text_ids = ids[:-1] if (ids and ids[-1] in eos) else ids
            out.append(GenResult(
                text=self.tokenizer.decode(text_ids),
                prompt_tokens=pid,
                completion_tokens=ids,
            ))
        return out

    # ----- inline ReAct generation (tool use mid-rollout) ------------------
    def _encode_no_special(self, text: str) -> list[int]:
        """Encode ``text`` as raw continuation tokens (no BOS/special added) so
        injected observations splice cleanly into an in-progress completion."""
        try:
            return self.tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            ids = self.tokenizer.encode(text)
            bos = getattr(self.tokenizer, "bos_token_id", None)
            if bos is not None and ids and ids[0] == bos:
                ids = ids[1:]
            return ids

    def _gen_segment(self, seq: list[int], sampler, budget: int, stop: str):
        """Sample one segment from ``seq`` (full token context), stopping after
        ``stop`` appears in the segment text, on EOS, or at ``budget`` tokens.
        Returns ``(tokens, text, hit_stop, hit_eos)``."""
        tokens: list[int] = []
        text = ""
        hit_stop = False
        for resp in stream_generate(
            self.model, self.tokenizer, seq, max_tokens=budget, sampler=sampler
        ):
            tokens.append(int(resp.token))
            text += resp.text
            if stop and stop in text:
                hit_stop = True
                break
        # Ended without our stop string and short of budget => the model emitted
        # EOS (its own natural stop). At/over budget it was truncated.
        hit_eos = (not hit_stop) and len(tokens) < budget
        return tokens, text, hit_stop, hit_eos

    def generate_react(
        self,
        prompt: str,
        *,
        tool_runner: Callable[[str], Optional[str]],
        max_tokens: int = 1024,
        temp: float = 0.7,
        top_p: float = 0.95,
        max_rounds: int = 4,
        stop: str = "</tool>",
        seed: int | None = None,
    ) -> ReactResult:
        """Generate with inline tool use under the *currently active* adapter.

        The model generates until it emits ``stop`` (the close of a
        ``<tool>...</tool>`` call); ``tool_runner(segment_text)`` then returns the
        observation text to splice in (or ``None`` to finish). Injected tokens
        are recorded with mask 0 so GRPO ignores them. ``max_tokens`` bounds the
        *model-generated* tokens across all rounds; injected obs tokens are free.
        The whole exchange is one logical completion scored as one trajectory."""
        if seed is not None:
            mx.random.seed(seed)
        sampler = make_sampler(temp=temp, top_p=top_p)
        prompt_tokens = self.tokenizer.encode(prompt)
        seq = list(prompt_tokens)
        segments: list[tuple[list[int], bool]] = []
        pieces: list[str] = []
        remaining = max_tokens
        n_tool_calls = 0
        rounds = 0
        while rounds < max_rounds and remaining > 0:
            rounds += 1
            seg_tokens, seg_text, hit_stop, hit_eos = self._gen_segment(
                seq, sampler, remaining, stop
            )
            segments.append((seg_tokens, True))
            pieces.append(seg_text)
            seq.extend(seg_tokens)
            remaining -= len(seg_tokens)
            if hit_eos or not hit_stop or remaining <= 0:
                break
            obs = tool_runner(seg_text)
            if obs is None:
                break
            n_tool_calls += 1
            obs_tokens = self._encode_no_special(obs)
            segments.append((obs_tokens, False))
            pieces.append(obs)
            seq.extend(obs_tokens)

        completion_tokens, loss_mask = assemble_react(segments)
        return ReactResult(
            text="".join(pieces),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            loss_mask=loss_mask,
            n_rounds=rounds,
            n_tool_calls=n_tool_calls,
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

    def completion_logprobs(
        self,
        prompt_tokens: list[int],
        completion_tokens: list[int],
        *,
        logit_chunk: int | None = None,
    ) -> mx.array:
        """Per-token log-probs of ``completion_tokens`` given ``prompt_tokens``
        under the *currently active* adapter. Returns shape
        ``[len(completion_tokens)]`` where element t is
        ``log p(completion_tokens[t] | prompt_tokens + completion_tokens[:t])``.

        This is the GRPO scoring primitive (DESIGN.md §8): one forward pass,
        **differentiable w.r.t. the active LoRA tree** (the only trainable
        params), so it works both for the grad-enabled policy pass and the
        no-grad reference pass (base adapter). ``token_logprobs`` returns *all*
        positions; this slices to the completion region without summing, which
        GRPO needs for its per-token policy-gradient and KL terms.

        ``logit_chunk`` (Sprint 8, from the mini-02 memory probe): the plain
        path materializes fp32 ``[T, V]`` logits AND log-softmax (~5.3GB each
        at 8.7k tokens), both pinned live by ``value_and_grad`` — a 4096-budget
        worst case peaks at 55GB and survives only on macOS swap. Chunked, the
        transformer runs once for ``[T, H]`` hidden states (H=4096 ≪ V=152k),
        and the LM head + log-softmax run over the completion region in
        ``mx.checkpoint``-ed chunks of this many positions — backward
        rematerializes one chunk's ``[chunk, V]`` at a time instead of holding
        ``[T, V]``. CAVEAT: ``mx.checkpoint`` differentiates only through the
        chunk function's array *arguments*; the closed-over head weights are
        treated as constants. That is exactly right here — the LoRA tree
        (attention/MLP projections) is the only trainable set and it sits
        upstream of the hidden states — but do NOT enable this if the head or
        embeddings are ever made trainable. ``None``/0 = plain v1 path."""
        n = len(completion_tokens)
        if n == 0:
            return mx.zeros((0,))
        seq = list(prompt_tokens) + list(completion_tokens)
        ids = mx.array(seq)[None]                  # [1, T]
        start = len(prompt_tokens) - 1             # logits[start] predicts completion[0]
        targets = mx.array(list(completion_tokens))
        if not logit_chunk:
            logits = self.model(ids)[0].astype(mx.float32)   # [T, V]
            logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            sel = logp[start : start + n]              # [n, V]
            return mx.take_along_axis(sel, targets[:, None], axis=-1)[:, 0]  # [n]

        hidden = self.model.model(ids)[0]          # [T, H] pre-head states
        sel_h = hidden[start : start + n]          # [n, H]
        head = self._lm_head_fn()

        def chunk_logprobs(h, t):
            logits = head(h).astype(mx.float32)                       # [k, V]
            logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            return mx.take_along_axis(logp, t[:, None], axis=-1)[:, 0]

        ckpt = mx.checkpoint(chunk_logprobs)
        outs = [
            ckpt(sel_h[i : i + logit_chunk], targets[i : i + logit_chunk])
            for i in range(0, n, logit_chunk)
        ]
        return outs[0] if len(outs) == 1 else mx.concatenate(outs)

    def _lm_head_fn(self):
        """The model's vocabulary projection as a callable ``[.., H] -> [.., V]``
        (plain or tied-embedding head, quantized or not)."""
        if getattr(getattr(self.model, "args", None), "tie_word_embeddings", False):
            return self.model.model.embed_tokens.as_linear
        return self.model.lm_head

    def sequence_logprob(self, prompt_tokens: list[int], completion_tokens: list[int]) -> mx.array:
        """Total log-prob of ``completion_tokens`` given ``prompt_tokens``
        under the active adapter (sum over the completion region)."""
        return self.completion_logprobs(prompt_tokens, completion_tokens).sum()

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
