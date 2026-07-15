"""Ensemble log-probability reward for 21-questions (ISOLATED — not wired into
the training pipeline yet).

Idea
----
Instead of an LLM *judge* emitting a discrete score, we ask an ensemble of
*frozen base* LLMs how likely they find the secret answer, given the game's
question/answer history. A model that has correctly narrowed the entity assigns
the answer string a high sequence log-probability; the reward is that likelihood.

For one ensemble member the score is::

    render:  <chat template>( system, user=<Q/A history + instruction>,
                              assistant="ANSWER: <answer>" )
    span:    the tokens of <answer> only (the "ANSWER:" scaffold and the closing
             turn/EOS control tokens are excluded)
    logp_i:  mean over the span of  log P(token_t | everything before it)

The per-member score is therefore a *length-normalised* mean token log-prob
(the "sum, then divide by #answer tokens" the spec asks for; predictions are
offset by -1 internally — token t is scored from the logits at position t-1).
The ensemble score is the mean of the members' scores (sum / #models). Higher
(closer to 0) ⇒ the ensemble finds the answer more probable.

Each member applies its **own** chat template and tokenizer, so the "ANSWER:"
span is located by character offsets in the rendered string and mapped back to
tokens — no per-model prompt string is hard-coded.

Backend: this is the DGX-Spark (torch/CUDA) path. It imports only torch +
transformers (no mlx, no `twin` internals) so it stays importable on the Spark
and can be poked in isolation. See twentyq/DESIGN.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Model registry.  Friendly name -> HF repo id.  All five are ungated as of
# 2026-07; swap a repo id here (e.g. to the gated official mistralai/meta repos
# after `hf auth login`) without touching any other code.
# ---------------------------------------------------------------------------
ENSEMBLE_MODELS: dict[str, str] = {
    "gemma-4-e2b": "google/gemma-4-E2B-it",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "smollm3-3b": "HuggingFaceTB/SmolLM3-3B",
    "llama-3.2-3b": "unsloth/Llama-3.2-3B-Instruct",
}

_DTYPES = {
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
    "float32": torch.float32, "fp32": torch.float32,
}

DEFAULT_SCAFFOLD = "ANSWER:"
DEFAULT_INSTRUCTION = (
    "Given this question/answer history for the game 21 questions, guess the "
    "secret. Return your answer in this format: ANSWER: <guess>."
)
DEFAULT_SYSTEM = "You are playing the game 21 questions. Deduce the secret entity."
# Sentinel for the state BEFORE any question has been asked — the reward's
# score(history_0) baseline. Must match the probe's H_EMPTY exactly so the
# trainer and the isolated probe compute the same starting potential.
EMPTY_HISTORY = "No questions have been asked yet."


def format_qa_history(qa_pairs: list[tuple[str, str]]) -> str:
    """Render a (question, answer) history into the flat block the ensemble
    conditions on — one ``Q: <q> A: <a>`` line per pair (the probe's H_* form).
    An empty history collapses to :data:`EMPTY_HISTORY` (the score(history_0)
    baseline)."""
    if not qa_pairs:
        return EMPTY_HISTORY
    return "\n".join(f"Q: {q} A: {a}" for q, a in qa_pairs)


def history_user(qa_pairs: list[tuple[str, str]], *,
                 instruction: str = DEFAULT_INSTRUCTION) -> str:
    """The full ``user`` string an ensemble member scores against: the guess
    instruction followed by the rendered Q/A history. Identical construction to
    the probe's ``hist(...)`` helper so scores line up 1:1."""
    return f"{instruction}\n\nHistory:\n{format_qa_history(qa_pairs)}"


@dataclass
class MemberScore:
    """One ensemble member's verdict on one (history, answer) pair."""

    name: str
    logprob: float           # mean per-token log-prob of the answer span
    n_tokens: int            # answer-span token count the mean was taken over
    answer_tokens: list[str] = field(default_factory=list)  # decoded span tokens
    token_logprobs: list[float] = field(default_factory=list)  # per-token log P
    full_text: str = ""      # the exact rendered string (special tokens included)

    @property
    def perplexity(self) -> float:
        return math.exp(-self.logprob) if self.n_tokens else float("inf")

    @property
    def token_probs(self) -> list[float]:
        """Per-token probabilities exp(log P) — the answer-span tokens' P(t|<t)."""
        return [math.exp(lp) for lp in self.token_logprobs]

    @property
    def prob(self) -> float:
        """Length-normalised probability: exp(mean log P) = geometric mean of the
        per-token probabilities. This is `.logprob` in probability space."""
        return math.exp(self.logprob) if self.n_tokens else 0.0


@dataclass
class EnsembleScore:
    """Aggregate result: `.score` is the reward (mean of member logprobs)."""

    score: float
    members: list[MemberScore]

    @property
    def per_model(self) -> dict[str, float]:
        return {m.name: m.logprob for m in self.members}

    def __repr__(self) -> str:  # compact, readable in a REPL / test print
        parts = ", ".join(f"{m.name}={m.logprob:+.3f}" for m in self.members)
        return f"EnsembleScore(score={self.score:+.4f} | {parts})"


class EnsembleMember:
    """A single frozen scoring model + its tokenizer, held resident on device."""

    def __init__(self, name: str, model_path: str, *, device: str = "cuda",
                 dtype: str = "bfloat16", attn_impl: str = "sdpa"):
        self.name = name
        self.model_path = model_path
        torch_dtype = _DTYPES.get(str(dtype).lower(), torch.bfloat16)
        cuda_ok = torch.cuda.is_available()
        self.device = torch.device(device if (device != "cuda" or cuda_ok) else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, attn_implementation=attn_impl,
        )
        model.to(self.device)
        model.eval()
        self.model = model

    # -- chat rendering ----------------------------------------------------
    def _apply_template(self, msgs: list[dict]) -> str:
        """Render `msgs` with thinking pinned OFF. Reasoning-capable templates
        (Qwen3, SmolLM3) accept `enable_thinking`; setting it False keeps their
        assistant turn a bare answer instead of a reasoning trace. Templates that
        don't know the kwarg (Gemma, Llama, Mistral) ignore an unused Jinja
        variable, but if one strictly rejects it we retry without — so scoring is
        never conditioned on a `<think>` block, whatever the template default."""
        try:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
            )

    def _render_full(self, system: str | None, user: str, assistant: str) -> str:
        """Render the whole 3-turn conversation to text. Falls back to folding
        the system message into the user turn for templates (e.g. Gemma) that
        reject a `system` role."""
        base = [{"role": "user", "content": user},
                {"role": "assistant", "content": assistant}]
        msgs = ([{"role": "system", "content": system}] + base) if system else base
        try:
            return self._apply_template(msgs)
        except Exception:
            if not system:
                raise
            folded = [{"role": "user", "content": f"{system}\n\n{user}"},
                      {"role": "assistant", "content": assistant}]
            return self._apply_template(folded)

    # -- span location -----------------------------------------------------
    def _answer_char_span(self, full_text: str, scaffold: str, answer: str,
                          assistant: str) -> tuple[int, int]:
        """Character range of the answer within the rendered conversation. We
        anchor on the assistant turn (last occurrence of the assistant content)
        and step past the scaffold, so an answer word that also appears in the
        Q/A history is never mistaken for the span."""
        a_pos = full_text.rfind(assistant)
        if a_pos < 0:  # rendering altered the content (whitespace) — best effort
            a_pos = full_text.rfind(answer)
            return a_pos, a_pos + len(answer)
        offset = len(scaffold) + 1 if scaffold else 0  # +1 for the space
        start = a_pos + offset
        return start, start + len(answer)

    def _span_token_indices(self, full_text: str, char_start: int, char_end: int):
        """Tokenize `full_text` (specials already present as literal text, so no
        add_special_tokens) and return (ids, first_idx, last_idx_exclusive) of
        the tokens overlapping [char_start, char_end)."""
        if self.tokenizer.is_fast:
            enc = self.tokenizer(full_text, add_special_tokens=False,
                                 return_offsets_mapping=True)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            idx = [i for i, (a, b) in enumerate(offs)
                   if a < char_end and b > char_start and b > a]
            if not idx:
                raise RuntimeError(f"[{self.name}] empty answer span at "
                                   f"chars [{char_start},{char_end})")
            return ids, idx[0], idx[-1] + 1

        # Slow-tokenizer fallback: prefix-diff (no offsets available).
        prefix_ids = self.tokenizer(full_text[:char_start],
                                    add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text[:char_end],
                                  add_special_tokens=False)["input_ids"]
        start = len(prefix_ids)
        # back off over any shared boundary token that got re-merged
        while start > 0 and full_ids[start - 1] != prefix_ids[start - 1]:
            start -= 1
        return full_ids, start, len(full_ids)

    # -- scoring -----------------------------------------------------------
    @torch.no_grad()
    def _span_logprobs(self, ids: list[int], start: int, end: int) -> list[float]:
        """Per-token log P(token_t | tokens_<t) over the answer span [start,end)."""
        input_ids = torch.tensor([ids], device=self.device)
        logits = self.model(input_ids=input_ids).logits[0].float()
        logp = torch.log_softmax(logits, dim=-1)
        # token at position t is predicted by the logits at position t-1
        sel = logp[start - 1:end - 1]
        targets = torch.tensor(ids[start:end], device=self.device, dtype=torch.long)
        tok_logp = sel.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return [float(x) for x in tok_logp]

    def score(self, user: str, answer: str, *, system: str | None = None,
              scaffold: str = DEFAULT_SCAFFOLD) -> MemberScore:
        assistant = f"{scaffold} {answer}".strip() if scaffold else answer
        full_text = self._render_full(system, user, assistant)
        cs, ce = self._answer_char_span(full_text, scaffold, answer, assistant)
        ids, start, end = self._span_token_indices(full_text, cs, ce)
        tok_logp = self._span_logprobs(ids, start, end)
        mean_lp = sum(tok_logp) / len(tok_logp) if tok_logp else 0.0
        span_toks = [self.tokenizer.decode([t]) for t in ids[start:end]]
        return MemberScore(self.name, mean_lp, end - start, span_toks,
                           token_logprobs=tok_logp, full_text=full_text)


class EnsembleReward:
    """Holds the frozen ensemble; `.score(...)` returns the aggregate reward."""

    def __init__(self, members: list[EnsembleMember]):
        if not members:
            raise ValueError("ensemble needs at least one member")
        self.members = members

    @classmethod
    def load(cls, names: list[str] | None = None, *, device: str = "cuda",
             dtype: str = "bfloat16", registry: dict[str, str] | None = None,
             verbose: bool = True) -> "EnsembleReward":
        registry = registry or ENSEMBLE_MODELS
        names = names or list(registry.keys())
        members = []
        for n in names:
            if n not in registry:
                raise KeyError(f"unknown ensemble model '{n}'; "
                               f"known: {list(registry)}")
            if verbose:
                print(f"[ensemble] loading {n} <- {registry[n]} ...", flush=True)
            members.append(EnsembleMember(n, registry[n], device=device, dtype=dtype))
        if verbose:
            print(f"[ensemble] {len(members)} models resident on {device}",
                  flush=True)
        return cls(members)

    def score(self, user: str, answer: str, *, system: str | None = None,
              scaffold: str = DEFAULT_SCAFFOLD) -> EnsembleScore:
        member_scores = [m.score(user, answer, system=system, scaffold=scaffold)
                         for m in self.members]
        agg = sum(m.logprob for m in member_scores) / len(member_scores)
        return EnsembleScore(agg, member_scores)

    def score_history(self, qa_pairs: list[tuple[str, str]], answer: str, *,
                      system: str | None = DEFAULT_SYSTEM,
                      scaffold: str = DEFAULT_SCAFFOLD,
                      instruction: str = DEFAULT_INSTRUCTION) -> EnsembleScore:
        """Score the secret ``answer`` given a 21-questions Q/A history — the
        game-facing entry point (the trainer's per-turn potential Φ). Empty
        ``qa_pairs`` yields the score(history_0) baseline. Thin wrapper over
        :meth:`score` that builds the ``user`` string via :func:`history_user`."""
        return self.score(history_user(qa_pairs, instruction=instruction),
                          answer, system=system, scaffold=scaffold)
