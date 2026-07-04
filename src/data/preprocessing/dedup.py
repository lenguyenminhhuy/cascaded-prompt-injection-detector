"""Near-duplicate dedup over the rendered pool (PLAN.md Chunk 5.3).

Two-level: exact dedup on normalized rendered_input, then near-dup dedup on a
token-shingle signature so trivially-reworded variants collapse. First
occurrence wins; order is stable.
"""

from __future__ import annotations

import re

from src.data.schema import Sample

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")
ZWSP = "​"


def _normalize(text: str) -> str:
    text = text.replace(ZWSP, "").lower()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def _shingles(norm: str, k: int = 5) -> frozenset[str]:
    toks = norm.split()
    if len(toks) < k:
        return frozenset([" ".join(toks)])
    return frozenset(" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1))


def dedup(samples: list[Sample], jaccard_threshold: float = 0.9) -> list[Sample]:
    """Near-dup dedup, bucketed by (channel, label) to bound the O(n^2) cost."""
    seen_exact: set[str] = set()
    kept: list[Sample] = []
    # per-bucket list of kept signatures
    bucket_sigs: dict[tuple[str, str], list[frozenset[str]]] = {}
    for s in samples:
        norm = _normalize(s.rendered_input)
        if norm in seen_exact:
            continue
        sig = _shingles(norm)
        bucket = (s.channel, s.label)
        dup = False
        for other in bucket_sigs.get(bucket, ()):  # only compare like-with-like
            inter = len(sig & other)
            if inter == 0:
                continue
            union = len(sig | other)
            if union and inter / union >= jaccard_threshold:
                dup = True
                break
        if dup:
            continue
        seen_exact.add(norm)
        bucket_sigs.setdefault(bucket, []).append(sig)
        kept.append(s)
    return kept
