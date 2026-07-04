"""Problem-diversity measurement + optional creator penalty (Sprint 9).

The literature's most convergent warning for self-play problem generation is
curriculum collapse into near-duplicate problems: R-Zero ships a BLEU-based
within-batch repetition penalty, Vocabulary Dropout forces lexical variety,
and PopuLoRA reports self-calibration-to-easy (RELATED_WORK.md §2/§5).
twin-models had NO diversity mechanism — this module adds the measurement
(always-on telemetry) and a config-gated reward penalty (``rewards.
w_diversity``, default 0 = off, run as a measured ablation).

Similarity is Jaccard over word bigrams of the problem *statement* — cheap,
tokenizer-free, and robust to small numeric edits (swapping constants in a
template leaves most bigrams intact, which is exactly the collapse case we
want to catch).
"""

import re

_WORD_RE = re.compile(r"[a-z0-9]+")


def _ngrams(text: str, n: int = 2) -> frozenset:
    words = _WORD_RE.findall((text or "").lower())
    if len(words) < n:
        return frozenset([tuple(words)]) if words else frozenset()
    return frozenset(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def jaccard(a: str, b: str, *, n: int = 2) -> float:
    """Jaccard similarity of word-``n``-gram sets, in [0, 1]."""
    ga, gb = _ngrams(a, n), _ngrams(b, n)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def mean_pairwise_similarity(statements: list[str]) -> float:
    """Mean Jaccard over all statement pairs (0 when fewer than 2)."""
    n = len(statements)
    if n < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += jaccard(statements[i], statements[j])
            pairs += 1
    return total / pairs


def cross_suite_penalties(suite_statements: list[list[str]]) -> list[float]:
    """Per-suite repetition penalty, parallel to ``suite_statements``.

    A suite's penalty is the mean, over its problems, of each problem's
    **nearest-neighbour similarity** to any problem in a *different* suite of
    the same iteration. Nearest-neighbour (not mean) so one verbatim
    duplicate is not diluted by unrelated siblings; cross-suite (not within)
    so a coherent themed ramp isn't punished for sharing vocabulary with
    itself. All suites share the iteration's theme, so a baseline similarity
    is normal — GRPO only feels the *differences* between suites, i.e. the
    suite that copied another's problems earns less than the one that
    explored. Empty/singleton cases -> 0.0."""
    penalties: list[float] = []
    for gi, mine in enumerate(suite_statements):
        others = [s for gj, other in enumerate(suite_statements)
                  if gj != gi for s in other]
        if not mine or not others:
            penalties.append(0.0)
            continue
        nn = [max(jaccard(p, o) for o in others) for p in mine]
        penalties.append(sum(nn) / len(nn))
    return penalties
