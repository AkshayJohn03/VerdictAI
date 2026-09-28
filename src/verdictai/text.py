"""Deterministic text utilities shared by heuristic judges and decontamination.

All functions are pure, offline and unicode-aware (``\\w`` in Python ``re``
matches unicode word characters, so CJK / accented / RTL tokens survive
tokenization).
"""

from __future__ import annotations

import re
from collections import Counter

_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)

STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are",
        "was", "were", "be", "been", "it", "this", "that", "as", "at", "by", "from", "you",
        "your", "we", "our", "i", "they", "their", "he", "she", "his", "her", "its", "not",
        "no", "do", "does", "did", "but", "if", "then", "than", "so", "such", "can", "could",
        "should", "would", "will", "shall", "may", "might", "must", "have", "has", "had",
        "about", "into", "over", "under", "out", "up", "down", "what", "which", "who", "whom",
        "how", "when", "where", "why", "all", "any", "both", "each", "few", "more", "most",
        "other", "some", "only", "own", "same", "too", "very", "just", "there", "here",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercased unicode word tokens."""
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens with stopwords and very short tokens removed."""
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 2]


def keyword_coverage(keywords: list[str], target_text: str) -> float:
    """Fraction of ``keywords`` present in ``target_text`` (0..1)."""
    if not keywords:
        return 0.0
    target = set(tokenize(target_text))
    hits = sum(1 for k in keywords if k in target)
    return hits / len(keywords)


def f1_overlap(text_a: str, text_b: str) -> float:
    """Multiset token F1 between two texts (bag-of-words overlap, 0..1)."""
    counts_a = Counter(tokenize(text_a))
    counts_b = Counter(tokenize(text_b))
    if not counts_a or not counts_b:
        return 0.0
    overlap = sum((counts_a & counts_b).values())
    precision = overlap / sum(counts_a.values())
    recall = overlap / sum(counts_b.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace (used before n-gram matching)."""
    return " ".join(text.lower().split())


def ngrams(text: str, n: int) -> list[str]:
    """Word n-grams of ``text`` joined by single spaces."""
    tokens = tokenize(text)
    if n <= 0 or len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
