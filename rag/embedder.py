"""Lightweight embedding helpers with optional sentence-transformers support."""

from __future__ import annotations

from collections import Counter
from math import sqrt


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alpha tokens."""
    tokens = []
    current = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens


def simple_embedding(text: str) -> dict[str, float]:
    """Create a normalized bag-of-words embedding."""
    counts = Counter(tokenize(text))
    norm = sqrt(sum(value * value for value in counts.values())) or 1.0
    return {token: value / norm for token, value in counts.items()}


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    """Compute cosine similarity between sparse embeddings."""
    if not left or not right:
        return 0.0
    common = set(left).intersection(right)
    return sum(left[token] * right[token] for token in common)

