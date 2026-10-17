"""Retrieval and answer metrics. Answer normalization follows the official SQuAD v1.1 script."""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence

import numpy as np


def recall_at_k(ranked: Sequence[str], gold: str, k: int) -> float:
    return float(gold in ranked[:k])


def reciprocal_rank(ranked: Sequence[str], gold: str, k: int = 10) -> float:
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id == gold:
            return 1.0 / rank
    return 0.0


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, golds: Sequence[str]) -> float:
    return float(any(normalize_answer(prediction) == normalize_answer(g) for g in golds))


def f1(prediction: str, golds: Sequence[str]) -> float:
    def score(gold: str) -> float:
        pred_tokens = normalize_answer(prediction).split()
        gold_tokens = normalize_answer(gold).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        same = sum(common.values())
        if same == 0:
            return 0.0
        precision, recall = same / len(pred_tokens), same / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    return max(score(g) for g in golds)


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q)) if values else 0.0
