"""PyTorch / CUDA backend — the NVIDIA DGX Spark native path.

Target hardware: GB10 Grace Blackwell (sm_121), 128 GB unified LPDDR5x, aarch64
Ubuntu, CUDA 13. Software: PyTorch (nightly for sm_121) + HuggingFace
``transformers`` + ``peft``. See DGX_SPARK.md for setup.

This module mirrors the MLX ``twin.models.TwinBase`` / ``twin.models.Adapters`` /
``twin.rl.grpo_update`` surface with a PyTorch implementation that produces and
consumes the SAME backend-neutral value types (``GenResult`` / ``ReactResult``
from :mod:`twin.models.types`, ``Trajectory`` from :mod:`twin.rl.core`) — so the
self-play loop is byte-for-byte the same code on either backend.

Design choices (locked with the user):
  * base loaded **bf16 full precision** (~16 GB; trivially fits 128 GB) + bf16
    LoRA — no quantization; best gradient quality for autograd-through-LoRA.
  * generation via native ``transformers.generate`` (batched, SDPA /
    FlashAttention-2). Policy LoRA weights hot-swap for free every step via
    PEFT ``set_adapter`` — no engine re-sync (the reason vLLM was deferred).

torch/transformers are imported at module load (this module is only imported
when ``compute.backend == 'torch'`` is selected, via
:func:`twin.backends.get_backend`), so a Mac that selects the mlx backend never
pays for — or needs — them. peft/safetensors import lazily where used.
"""

import bisect
import os
import re

try:
    import torch
    from torch.utils.checkpoint import checkpoint
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        LogitsProcessorList,
        StoppingCriteria,
        StoppingCriteriaList,
    )
except ImportError as e:  # pragma: no cover - exercised only on a torch-less host
    raise ImportError(
        "compute.backend='torch' needs PyTorch + transformers (+ peft, "
        "safetensors) installed. On the DGX Spark install the sm_121 build per "
        "DGX_SPARK.md, e.g. `pip install torch --index-url "
        "https://download.pytorch.org/whl/nightly/cu130` then `pip install "
        f"transformers peft accelerate safetensors`. Original error: {e}"
    ) from e

from twin.models.types import GenResult, ReactResult, assemble_react
from twin.rl.core import Trajectory  # noqa: F401  (re-exported for parity/tests)


def banned_phrase_variants(phrase: str) -> list[str]:
    """Surface variants of one banned phrase for logits-level masking.

    Token-sequence bans are literal, so cover the codings a creator actually
    emits: case variants (as-is / lower / Title / First-upper), bare plural /
    singular (mirroring ``guess_matches``'s one-trailing-'s' tolerance), each
    with and without a leading space (different token ids in BPE vocabs).
    A wrong variant bans a token sequence that never occurs — harmless."""
    phrase = (phrase or "").strip()
    if not phrase:
        return []
    bases = {phrase, phrase.lower(), phrase.upper(), phrase.title(),
             phrase[:1].upper() + phrase[1:].lower()}
    for b in list(bases):
        bases.add(b + "s" if not b.endswith("s") else b[:-1])
    out = []
    for b in bases:
        out.extend((b, " " + b))
    return sorted(set(out))


def banned_token_sequences(tokenizer, phrases: list[str]) -> list[list[int]]:
    """``bad_words_ids`` for ``phrases``: the deduplicated token sequences of
    every :func:`banned_phrase_variants` expansion, encoded without special
    tokens (they ban mid-completion continuations, not full prompts).

    NOTE (2026-07-16, q-fullv45-ctrl-terminal iters 3/7): a token-sequence ban
    is self-defeating as the ONLY guard — masking the final token of the
    banned path reroutes the sampler onto an alternate BPE segmentation of the
    SAME surface string ('Black'+' Card'+'am'+'om' after 'amom' was banned).
    Generation therefore uses :class:`BannedStringsProcessor`, which masks at
    the string level; this encoding survives for telemetry/tests."""
    seqs: dict[tuple[int, ...], list[int]] = {}
    for phrase in phrases:
        for variant in banned_phrase_variants(phrase):
            ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
            if ids:
                seqs.setdefault(tuple(ids), list(ids))
    return list(seqs.values())


# Per-tokenizer sorted (casefolded token surface, id) table for prefix-range
# lookups. Built once per tokenizer (~seconds for a 260k vocab), then every
# banned-string mask is one binary search + slice per active remainder.
_VOCAB_TABLES: dict = {}


def _vocab_table(tokenizer):
    key = getattr(tokenizer, "name_or_path", None) or id(tokenizer)
    table = _VOCAB_TABLES.get(key)
    if table is None:
        surfaces = tokenizer.batch_decode([[i] for i in range(len(tokenizer))])
        pairs = sorted((s.casefold(), i) for i, s in enumerate(surfaces))
        _VOCAB_TABLES[key] = table = (
            [text for text, _ in pairs],           # sorted surfaces
            [tid for _, tid in pairs],             # ids in the same order
        )
    return table


class BannedStringsProcessor:
    """STRING-level phrase ban for ``generate`` (a ``LogitsProcessor``).

    At each step the generated-so-far text of each row is decoded; whenever it
    ends with a (possibly empty) prefix of a banned phrase, every vocab token
    whose surface begins the remaining characters is masked to -inf. The
    decoded completion therefore can never contain a banned phrase under ANY
    tokenization — closing the alternate-segmentation leak that pure
    ``bad_words_ids`` sequences have (observed live: 3/15 masked retries
    emitted the exact banned secret through a rerouted BPE split). Matching is
    casefolded; plural forms come in via :func:`banned_phrase_variants`."""

    def __init__(self, tokenizer, phrases: list[str], prompt_len: int):
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_len)
        self.phrases = sorted({
            variant.strip().casefold()
            for phrase in phrases
            for variant in banned_phrase_variants(phrase)
            if variant.strip()
        })
        self._max_chars = max((len(p) for p in self.phrases), default=0)
        self._surfaces, self._ids = _vocab_table(tokenizer)

    def _ban_range(self, scores, row, remainder: str) -> None:
        lo = bisect.bisect_left(self._surfaces, remainder)
        hi = bisect.bisect_left(self._surfaces, remainder + "￿")
        if hi > lo:
            ids = torch.as_tensor(self._ids[lo:hi], device=scores.device)
            scores[row, ids] = float("-inf")

    def __call__(self, input_ids, scores):
        if not self.phrases:
            return scores
        # Decode only a bounded tail: a prefix of a banned phrase can span at
        # most max_chars characters, and one token rarely exceeds ~16 chars.
        max_tail_tokens = max(8, self._max_chars // 2 + 8)
        for row in range(input_ids.shape[0]):
            generated = input_ids[row, self.prompt_len:]
            tail = ""
            if generated.numel():
                tail = self.tokenizer.decode(
                    generated[-max_tail_tokens:].tolist()).casefold()
            for phrase in self.phrases:
                # EVERY matched prefix bans its remainder — periodic prefixes
                # ("aa" of "aab") can match at several lengths at once, and
                # k=0 (empty prefix) always applies, catching whole-phrase
                # single tokens like ' cardamom'.
                start = min(len(phrase) - 1, len(tail))
                for k in range(start, -1, -1):
                    if k == 0 or tail.endswith(phrase[:k]):
                        self._ban_range(scores, row, phrase[k:])
        return scores

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


class _StopOnString(StoppingCriteria):
    """Fire once ``stop`` appears in the text generated *after* ``prompt_len``
    tokens — the torch analog of mlx-lm's stop-string break in ``_gen_segment``.
    Decoding only the new suffix each step keeps this cheap for our short
    tool-call segments."""

    def __init__(self, tokenizer, prompt_len: int, stop: str):
        self.tok = tokenizer
        self.plen = prompt_len
        self.stop = stop
        self.hit = False

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if not self.stop:
            return False
        text = self.tok.decode(input_ids[0, self.plen:], skip_special_tokens=False)
        if self.stop in text:
            self.hit = True
            return True
        return False


class TorchTwinBase:
    """The frozen base model + tokenizer on CUDA, with the same primitives as
    ``twin.models.TwinBase``. ``self.model`` is reassigned to the PEFT-wrapped
    model by :class:`TorchAdapters`; all methods read ``self.model`` live, so
    the active-adapter selection flows through transparently."""

    def __init__(self, model_path: str, compute_cfg=None):
        cc = compute_cfg
        dtype_name = getattr(cc, "dtype", "bfloat16")
        self._dtype = _DTYPES.get(str(dtype_name).lower(), torch.bfloat16)
        attn = getattr(cc, "attn_impl", "sdpa")
        want_device = getattr(cc, "device", "cuda")
        self._compile = bool(getattr(cc, "compile", False))
        self._grad_ckpt = bool(getattr(cc, "grad_checkpointing", False))

        # Blackwell perf: TF32 matmul + tensor-core float32 path.
        if getattr(cc, "tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision(getattr(cc, "matmul_precision", "high"))

        path = os.path.expanduser(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        # Decoder-only batched generation needs LEFT padding so completions line
        # up at the sequence tail.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=self._dtype, attn_implementation=attn
        )
        cuda_ok = torch.cuda.is_available()
        self.device = torch.device(
            want_device if (want_device != "cuda" or cuda_ok) else "cpu"
        )
        model.to(self.device)
        model.eval()
        if self._grad_ckpt:
            # HF checkpointing only fires in train() mode; completion_logprobs
            # flips to train() around the grad-enabled forward (LoRA/base
            # dropout are 0, so this changes nothing but activation caching).
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.model = model
        self._eos = self._collect_eos()
        self._pad = self.tokenizer.pad_token_id

    # ----- tokenization helpers -------------------------------------------
    def _encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    # The rendered chat prompt already carries all control tokens as text, so we
    # never re-add specials (matches the mlx path's continuation encoding).
    _encode_no_special = _encode

    def _collect_eos(self) -> set[int]:
        ids: set[int] = set()

        def _add(v):
            if isinstance(v, (list, tuple)):
                ids.update(int(x) for x in v)
            elif v is not None:
                ids.add(int(v))

        _add(self.tokenizer.eos_token_id)
        _add(getattr(getattr(self.model, "generation_config", None), "eos_token_id", None))
        for tok in ("<|im_end|>", "<|endoftext|>"):
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                ids.add(tid)
        return ids

    def _first_eos_cut(self, comp: list[int]) -> list[int]:
        """Trim ``comp`` at (and including) the first EOS — mirrors mlx-lm, and
        drops the padding tail transformers appends after early-stopping."""
        for k, t in enumerate(comp):
            if t in self._eos:
                return comp[: k + 1]
        return comp

    def _decode(self, ids) -> str:
        """Decode to text with the terminal EOS stripped but other tokens
        (``<think>``, tool tags) preserved, matching the mlx completion text."""
        ids = [int(t) for t in ids if int(t) not in self._eos]
        return self.tokenizer.decode(ids, skip_special_tokens=False)

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
    @torch.no_grad()
    def generate(
        self, prompt: str, *, max_tokens: int = 512, temp: float = 0.7,
        top_p: float = 0.95, top_k: int | None = None,
        min_p: float | None = None, seed: int | None = None,
        banned_strings: list[str] | None = None,
    ) -> GenResult:
        return self.generate_batch(
            [prompt], max_tokens=max_tokens, temp=temp, top_p=top_p,
            top_k=top_k, min_p=min_p, seed=seed, completion_batch_size=1,
            banned_strings=banned_strings,
        )[0]

    @torch.no_grad()
    def generate_batch(
        self, prompts: list[str], *, max_tokens: int = 512, temp: float = 0.7,
        top_p: float = 0.95, top_k: int | None = None,
        min_p: float | None = None, seed: int | None = None,
        completion_batch_size: int = 32,
        banned_strings: list[str] | None = None,
    ) -> list[GenResult]:
        """Batched decode under the active adapter. ``completion_batch_size``
        caps concurrent sequences (KV-cache bound). Semantics match
        :meth:`generate` per prompt: ``completion_tokens`` are the exact sampled
        ids up to and including the first EOS; ``text`` excludes that EOS.

        ``banned_strings`` bans each phrase at the STRING level
        (:class:`BannedStringsProcessor`): any token that would make the
        decoded completion contain a banned phrase — under any BPE
        segmentation — is masked to -inf, so the decoder takes the
        next-most-likely non-banned continuation and sampling draws from the
        renormalized masked distribution."""
        if seed is not None:
            torch.manual_seed(seed)
        do_sample = temp > 0
        # ``None`` means UNSET: the kwarg is omitted so HF falls back to the
        # model's generation_config (gemma-4-E2B ships top_k=64 — the implicit
        # historical sampler for every run in the lineage). Passing None
        # through would instead OVERRIDE the config and disable the warper;
        # use top_k=0 to request that explicitly.
        sampler_overrides: dict = {}
        if do_sample and top_k is not None:
            sampler_overrides["top_k"] = top_k
        if do_sample and min_p is not None:
            sampler_overrides["min_p"] = min_p
        eos = list(self._eos) or None
        bs = max(1, completion_batch_size)
        out: list[GenResult] = []
        for i in range(0, len(prompts), bs):
            chunk = prompts[i : i + bs]
            enc = self.tokenizer(
                chunk, return_tensors="pt", padding=True, add_special_tokens=False
            )
            input_ids = enc["input_ids"].to(self.device)
            attn = enc["attention_mask"].to(self.device)
            processors = None
            if banned_strings:
                processors = LogitsProcessorList([BannedStringsProcessor(
                    self.tokenizer, banned_strings,
                    prompt_len=input_ids.shape[1])])
            gen = self.model.generate(
                input_ids=input_ids, attention_mask=attn,
                max_new_tokens=max_tokens, do_sample=do_sample,
                temperature=(temp if do_sample else None),
                top_p=(top_p if do_sample else None),
                pad_token_id=self._pad, eos_token_id=eos,
                logits_processor=processors,
                **sampler_overrides,
            )
            plen = input_ids.shape[1]
            for j, p in enumerate(chunk):
                comp = self._first_eos_cut([int(x) for x in gen[j, plen:].tolist()])
                text_ids = comp[:-1] if (comp and comp[-1] in self._eos) else comp
                out.append(GenResult(self._decode(text_ids), self._encode(p), comp))
        return out

    # ----- inline ReAct generation (tool use mid-rollout) ------------------
    @torch.no_grad()
    def _gen_segment(self, seq_ids: list[int], budget: int, temp: float,
                     top_p: float, stop: str):
        """Sample one segment from ``seq_ids`` (full token context), stopping
        after ``stop`` appears, on EOS, or at ``budget`` tokens. Returns
        ``(tokens, text, hit_stop, hit_eos)`` — the mlx ``_gen_segment`` contract."""
        seq = torch.tensor([seq_ids], device=self.device)
        plen = seq.shape[1]
        do_sample = temp > 0
        crit = StoppingCriteriaList([_StopOnString(self.tokenizer, plen, stop)]) if stop else None
        gen = self.model.generate(
            input_ids=seq, max_new_tokens=budget, do_sample=do_sample,
            temperature=(temp if do_sample else None),
            top_p=(top_p if do_sample else None),
            pad_token_id=self._pad, eos_token_id=(list(self._eos) or None),
            stopping_criteria=crit,
        )
        new = [int(x) for x in gen[0, plen:].tolist()]
        if new and new[-1] in self._eos:
            new = self._first_eos_cut(new)
        hit_stop = bool(crit and crit[0].hit)
        hit_eos = (not hit_stop) and len(new) > 0 and new[-1] in self._eos
        return new, self._decode(new), hit_stop, hit_eos

    @torch.no_grad()
    def generate_react(
        self, prompt: str, *, tool_runner, max_tokens: int = 1024,
        temp: float = 0.7, top_p: float = 0.95, max_rounds: int = 4,
        stop: str = "</tool>", seed: int | None = None,
    ) -> ReactResult:
        """Inline tool use under the active adapter — same contract and masking
        as ``twin.models.TwinBase.generate_react`` (injected obs tokens carry
        mask 0 so GRPO ignores them)."""
        if seed is not None:
            torch.manual_seed(seed)
        prompt_tokens = self._encode(prompt)
        seq = list(prompt_tokens)
        segments: list[tuple[list[int], bool]] = []
        pieces: list[str] = []
        remaining = max_tokens
        n_tool_calls = 0
        rounds = 0
        while rounds < max_rounds and remaining > 0:
            rounds += 1
            seg_tokens, seg_text, hit_stop, hit_eos = self._gen_segment(
                seq, remaining, temp, top_p, stop
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

    # ----- scoring (GRPO primitive) ----------------------------------------
    def _causal_lm(self):
        """The underlying ``*ForCausalLM`` (unwrapping the PEFT wrapper if any)."""
        m = self.model
        base = getattr(m, "base_model", None)
        if base is not None and hasattr(base, "model"):
            return base.model  # PeftModel.base_model.model
        return m

    def _transformer(self):
        """The decoder stack (returns ``last_hidden_state``) — LoRA layers live
        inside it, so calling it directly still applies the active adapter."""
        return self._causal_lm().model

    def _lm_head(self):
        return self._causal_lm().get_output_embeddings()

    def completion_logprobs(
        self, prompt_tokens: list[int], completion_tokens: list[int], *,
        logit_chunk: int | None = None,
    ):
        """Per-token log-probs of ``completion_tokens`` given ``prompt_tokens``
        under the active adapter — differentiable w.r.t. the active LoRA params.
        Shape ``[len(completion_tokens)]``.

        Grad follows the ambient torch mode: the policy pass (inside
        :func:`torch_grpo_update`, grad enabled) gets a graph; the reference
        pass (loop wraps it in ``backend.no_grad()``) does not.

        ``logit_chunk`` is the torch analog of the mlx chunked LM head: run the
        transformer once for ``[T, H]`` hidden states, then the (LoRA-free) head
        + log-softmax over the completion region in ``torch.utils.checkpoint``-ed
        chunks — backward rematerializes one ``[chunk, V]`` at a time instead of
        pinning ``[T, V]``. Rarely needed on 128 GB; ``None``/0 = plain path."""
        n = len(completion_tokens)
        if n == 0:
            return torch.zeros(0, device=self.device, dtype=torch.float32)
        ids = torch.tensor([list(prompt_tokens) + list(completion_tokens)], device=self.device)
        start = len(prompt_tokens) - 1
        targets = torch.tensor(list(completion_tokens), device=self.device, dtype=torch.long)

        grad_ckpt = self._grad_ckpt and torch.is_grad_enabled()
        if grad_ckpt:
            self.model.train()
        try:
            if not logit_chunk:
                logits = self.model(input_ids=ids).logits[0].float()
                logp = torch.log_softmax(logits, dim=-1)
                sel = logp[start : start + n]
                return sel.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

            hidden = self._transformer()(input_ids=ids).last_hidden_state[0]  # [T,H]
            sel_h = hidden[start : start + n]
            head = self._lm_head()

            def chunk_logprobs(h, t):
                logits = head(h).float()
                logp = torch.log_softmax(logits, dim=-1)
                return logp.gather(-1, t.unsqueeze(-1)).squeeze(-1)

            outs = []
            for i in range(0, n, logit_chunk):
                h = sel_h[i : i + logit_chunk]
                t = targets[i : i + logit_chunk]
                if torch.is_grad_enabled():
                    outs.append(checkpoint(chunk_logprobs, h, t, use_reentrant=False))
                else:
                    outs.append(chunk_logprobs(h, t))
            return outs[0] if len(outs) == 1 else torch.cat(outs)
        finally:
            if grad_ckpt:
                self.model.eval()

    def token_logprobs(self, token_ids: list[int]):
        ids = torch.tensor([list(token_ids)], device=self.device)
        logits = self.model(input_ids=ids).logits[0].float()
        logp = torch.log_softmax(logits, dim=-1)
        targets = torch.tensor(list(token_ids[1:]), device=self.device, dtype=torch.long)
        return logp[:-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    def sequence_logprob(self, prompt_tokens: list[int], completion_tokens: list[int]):
        return self.completion_logprobs(prompt_tokens, completion_tokens).sum()

    # ----- oracle ----------------------------------------------------------
    def oracle(self, question: str, *, max_tokens: int = 512, temp: float = 0.2) -> str:
        prompt = self.render(
            question,
            system=(
                "You are a precise reference. Answer the question factually and "
                "concisely. If unsure, say so."
            ),
        )
        return self.generate(prompt, max_tokens=max_tokens, temp=temp).text


class TorchAdapters:
    """Two LoRA adapters ('A','B') + a zeroed 'base' over one frozen bf16 model,
    via PEFT. Mirrors ``twin.models.Adapters``.

    PEFT keeps each adapter's weights in the model keyed by name, so — unlike the
    mlx tree-swap — ``activate`` is ``set_adapter`` (pointer flip), 'base' is
    ``disable_adapter_layers`` (LoRA contributes 0 == the frozen base, the KL
    reference), and ``capture`` is a no-op (``optimizer.step`` already wrote the
    active adapter's weights in place). Each adapter owns distinct params, so a
    per-adapter optimizer never bleeds moments across A/B."""

    NAMES = ("A", "B")
    _DEFAULT_LEAVES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    def __init__(self, base, *, num_layers: int, rank: int, alpha: float,
                 dropout: float = 0.0, keys: list[str] | None = None):
        from peft import LoraConfig, get_peft_model

        model = base.model
        cfg = model.config
        # Multimodal checkpoints (e.g. Gemma4ForConditionalGeneration) nest the
        # text stack's depth under `text_config` and put the decoder layers under
        # `...language_model.layers`, sharing leaf names (q_proj, ...) with vision
        # /audio towers. A flat `num_hidden_layers` + leaf-only target would then
        # (a) KeyError here and (b) risk adapting a tower. So resolve the text
        # depth from text_config when the top level lacks it, and — only then —
        # pin LoRA to the text decoder's last N layers by module-name regex.
        n_layers_total = getattr(cfg, "num_hidden_layers", None)
        nested = n_layers_total is None
        if nested:
            text_cfg = getattr(cfg, "text_config", None)
            n_layers_total = getattr(text_cfg, "num_hidden_layers", None)
            if n_layers_total is None:
                raise ValueError(
                    f"cannot find num_hidden_layers on {type(cfg).__name__} "
                    "(nor its text_config) to place LoRA layers")
        # Adapt only the LAST num_layers blocks (matches the mlx target set).
        layers = list(range(max(0, n_layers_total - num_layers), n_layers_total))
        leaves = self._leaves(keys)
        if nested:
            prefix = self._decoder_layers_prefix(model, n_layers_total)
            idx_alt = "|".join(str(i) for i in layers)
            leaf_alt = "|".join(re.escape(l) for l in leaves)
            # PEFT regex-fullmatches module names when target_modules is a str.
            target_modules = (rf"{re.escape(prefix)}\.(?:{idx_alt})\..*\."
                              rf"(?:{leaf_alt})")
            lc = LoraConfig(
                r=rank, lora_alpha=alpha, lora_dropout=dropout,
                target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
            )
        else:
            lc = LoraConfig(
                r=rank, lora_alpha=alpha, lora_dropout=dropout,
                target_modules=leaves, layers_to_transform=layers,
                bias="none", task_type="CAUSAL_LM",
            )
        peft_model = get_peft_model(model, lc, adapter_name="A")
        peft_model.add_adapter("B", lc)
        if getattr(base, "_compile", False):
            # Experimental: torch.compile the scoring forward. May recompile on
            # adapter switch; opt-in via compute.compile. OptimizedModule proxies
            # set_adapter/disable_adapter_layers to the wrapped PeftModel.
            peft_model = torch.compile(peft_model)
        base.model = peft_model
        self.base = base
        self.model = peft_model
        self._disabled = False
        self.active: str | None = None
        self.activate("A")

    @classmethod
    def from_config(cls, base, lora_cfg) -> "TorchAdapters":
        # PEFT scaling == lora_alpha / r; pin it to the shared effective_scale so
        # torch and mlx apply the same LoRA magnitude for one config.
        return cls(
            base,
            num_layers=lora_cfg.num_layers,
            rank=lora_cfg.rank,
            alpha=lora_cfg.effective_scale * lora_cfg.rank,
            dropout=lora_cfg.dropout,
            keys=lora_cfg.keys,
        )

    @classmethod
    def _leaves(cls, keys):
        if not keys:
            return list(cls._DEFAULT_LEAVES)
        return sorted({k.split(".")[-1] for k in keys})

    @staticmethod
    def _decoder_layers_prefix(model, n_layers_total: int) -> str:
        """Module-name prefix of the TEXT decoder's layer list (e.g.
        ``model.language_model.layers``) on a multimodal model. Chosen as the
        ``*.layers`` container whose block count equals the text depth — so the
        vision/audio towers (different depths, same leaf names) are excluded.
        Prefers a ``language_model`` container on ties."""
        counts: dict[str, set[int]] = {}
        for name, _ in model.named_modules():
            m = re.match(r"(.*\blayers)\.(\d+)$", name)
            if m:
                counts.setdefault(m.group(1), set()).add(int(m.group(2)))
        exact = [p for p, idxs in counts.items() if len(idxs) == n_layers_total]
        for cands in (exact, list(counts)):
            if not cands:
                continue
            lm = [p for p in cands if "language_model" in p]
            return (lm or cands)[0]
        raise ValueError("no '*.layers' decoder container found for LoRA placement")

    # ----- per-adapter param access ---------------------------------------
    def _named_adapter_params(self, name: str):
        for pn, p in self.model.named_parameters():
            segs = pn.split(".")
            if name in segs and any(s.startswith("lora_") for s in segs):
                yield pn, p

    def adapter_params(self, name: str) -> list:
        return [p for _, p in self._named_adapter_params(name)]

    # ----- activation ------------------------------------------------------
    def activate(self, name: str) -> None:
        if name == "base":
            if not self._disabled:
                self.model.disable_adapter_layers()
                self._disabled = True
            self.active = "base"
            return
        if self._disabled:
            self.model.enable_adapter_layers()
            self._disabled = False
        self.model.set_adapter(name)
        self.active = name

    def using(self, name: str):
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            prev = self.active
            self.activate(name)
            try:
                yield
            finally:
                if prev is not None:
                    self.activate(prev)

        return _ctx()

    def capture(self, name: str) -> None:
        # PEFT updates the active adapter's params in place on optimizer.step();
        # nothing to copy back (kept for interface parity with the mlx Adapters).
        if name not in self.NAMES:
            raise KeyError(name)

    # ----- persistence -----------------------------------------------------
    def save(self, name: str, path: str) -> None:
        from safetensors.torch import save_file

        tensors = {pn: p.detach().cpu().contiguous()
                   for pn, p in self._named_adapter_params(name)}
        save_file(tensors, path)

    def load(self, name: str, path: str) -> None:
        from safetensors.torch import load_file

        state = load_file(path)
        own = dict(self._named_adapter_params(name))
        with torch.no_grad():
            for k, v in state.items():
                if k in own:
                    own[k].copy_(v.to(own[k].device, own[k].dtype))
        if self.active == name:
            self.activate(name)

    # ----- introspection ---------------------------------------------------
    def num_params(self, name: str = "A") -> int:
        return sum(p.numel() for _, p in self._named_adapter_params(name))

    def global_norm(self, name: str = "A") -> float:
        if name == "base":
            return 0.0
        total = 0.0
        for _, p in self._named_adapter_params(name):
            total += float((p.detach().float() ** 2).sum())
        return total ** 0.5

    def snapshot(self, name: str):
        return {pn: p.detach().float().clone()
                for pn, p in self._named_adapter_params(name)}

    def drift_from(self, name: str, ref_tree) -> float:
        cur = dict(self._named_adapter_params(name))
        total = 0.0
        for k, v in ref_tree.items():
            if k in cur:
                d = cur[k].detach().float() - v.float()
                total += float((d ** 2).sum())
        return total ** 0.5


# ----- GRPO update (torch) -------------------------------------------------
def _k3_kl(logp_theta, logp_ref):
    """Per-token k3 KL estimator of KL(π_θ ‖ π_ref): ``exp(δ) − δ − 1`` with
    ``δ = logp_ref − logp_theta`` (matches twin.rl.grpo.k3_kl)."""
    delta = logp_ref - logp_theta
    return torch.exp(delta) - delta - 1.0


def torch_grpo_update(model, optimizer, batch, score_fn, *,
                      kl_beta: float = 0.0, grad_clip=1.0,
                      microbatch_size: int = 0) -> dict:
    """One GRPO optimizer step over ``batch`` (advantages pre-assigned) — the
    torch mirror of :func:`twin.rl.grpo.grpo_update`, returning the same metrics.

    ``model`` must have the policy adapter active; ``optimizer`` holds that
    adapter's params. Each micro-batch chunk's ``(loss / total_tokens)`` is
    backpropagated, **accumulating** normalized grads in ``.grad`` (linear, so
    identical to one big graph, with peak memory bounded by one chunk). Injected
    (mask==0) positions are index-dropped from both PG and KL before reducing."""
    batch = [t for t in batch if t.n_train_tokens > 0]
    total_tokens = sum(t.n_train_tokens for t in batch)
    if total_tokens == 0:
        return {"loss": 0.0, "pg": 0.0, "kl": 0.0, "grad_norm": 0.0,
                "n_traj": 0, "n_tokens": 0}

    chunk_size = microbatch_size if (microbatch_size and microbatch_size > 0) else len(batch)
    optimizer.zero_grad(set_to_none=True)
    pg_total = 0.0
    kl_total = 0.0
    for start in range(0, len(batch), chunk_size):
        chunk = batch[start : start + chunk_size]
        with torch.enable_grad():
            pg = None
            kl = None
            for t in chunk:
                logp = score_fn(model, t.prompt_ids, t.completion_ids)  # [n], grad
                ref = t.ref_logprobs
                idx = t.train_indices()
                if idx is not None:                     # drop injected positions
                    sel = torch.as_tensor(idx, device=logp.device, dtype=torch.long)
                    logp = logp.index_select(0, sel)
                    ref = ref.index_select(0, sel) if ref is not None else None
                term = (-float(t.advantage)) * logp.sum()
                pg = term if pg is None else pg + term
                if kl_beta and ref is not None:
                    kterm = _k3_kl(logp, ref).sum()
                    kl = kterm if kl is None else kl + kterm
            if pg is None:
                continue
            kl_val = kl if kl is not None else torch.zeros((), device=pg.device)
            loss = pg + kl_beta * kl_val
            (loss / total_tokens).backward()
        pg_total += float(pg.detach())
        kl_total += float(kl_val.detach())

    params = [p for g in optimizer.param_groups for p in g["params"]]
    if grad_clip:
        raw_norm = float(torch.nn.utils.clip_grad_norm_(params, grad_clip))
    else:
        sq = sum(float((p.grad.detach() ** 2).sum())
                 for p in params if p.grad is not None)
        raw_norm = sq ** 0.5
    optimizer.step()

    return {
        "loss": (pg_total + kl_beta * kl_total) / total_tokens,
        "pg": pg_total / total_tokens,
        "kl": kl_total / total_tokens,
        "grad_norm": raw_norm,
        "n_traj": len(batch),
        "n_tokens": total_tokens,
    }


class TorchBackend:
    name = "torch"

    def load_base(self, model_cfg, compute_cfg=None):
        return TorchTwinBase(model_cfg.path, compute_cfg)

    def build_adapters(self, base, lora_cfg):
        return TorchAdapters.from_config(base, lora_cfg)

    def make_optimizer(self, adapters, name: str, lr: float):
        return torch.optim.AdamW(
            adapters.adapter_params(name), lr=lr, weight_decay=0.0
        )

    def seed(self, n: int) -> None:
        torch.manual_seed(n)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(n)

    def realize(self, *xs) -> None:
        # Eager execution — tensors are already materialized. (No-op; kept so the
        # loop's mx.eval sites are backend-uniform.)
        return None

    def no_grad(self):
        return torch.no_grad()

    def grpo_update(self, model, optimizer, batch, score_fn, *,
                    kl_beta: float = 0.0, grad_clip=1.0,
                    microbatch_size: int = 0) -> dict:
        return torch_grpo_update(
            model, optimizer, batch, score_fn,
            kl_beta=kl_beta, grad_clip=grad_clip, microbatch_size=microbatch_size,
        )

    def peak_memory_gb(self) -> float | None:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
        return None

    def reset_peak_memory(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def clear_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
