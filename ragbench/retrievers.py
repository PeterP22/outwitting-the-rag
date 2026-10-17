"""Retrieval setups behind one interface: search(query, k) -> ranked doc ids."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import bm25s
import numpy as np

from ragbench.data import Doc

Embed = Callable[[Sequence[str]], np.ndarray]
Score = Callable[[str, Sequence[str]], Sequence[float]]


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[str]: ...


def _doc_text(doc: Doc) -> str:
    return f"{doc.title}. {doc.text}"


class BM25:
    name = "bm25"

    def __init__(self, docs: Sequence[Doc]) -> None:
        self.ids = [d.id for d in docs]
        self.index = bm25s.BM25()
        self.index.index(bm25s.tokenize([_doc_text(d) for d in docs], stopwords="en",
                                        show_progress=False), show_progress=False)

    def search(self, query: str, k: int) -> list[str]:
        tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        if not tokens.vocab:
            return []
        hits, _ = self.index.retrieve(tokens, k=min(k, len(self.ids)), show_progress=False)
        return [self.ids[i] for i in hits[0]]


class Dense:
    name = "dense"

    def __init__(self, docs: Sequence[Doc], embed_docs: Embed, embed_query: Embed) -> None:
        self.ids = [d.id for d in docs]
        self.matrix = _unit(embed_docs([_doc_text(d) for d in docs]))
        self.embed_query = embed_query

    def search(self, query: str, k: int) -> list[str]:
        scores = self.matrix @ _unit(self.embed_query([query]))[0]
        top = np.argsort(-scores)[:k]
        return [self.ids[i] for i in top]


class HybridRRF:
    """Reciprocal rank fusion (Cormack et al., 2009) over several retrievers."""

    def __init__(self, retrievers: Sequence[Retriever], depth: int = 50, k0: int = 60) -> None:
        self.retrievers = retrievers
        self.depth, self.k0 = depth, k0
        self.name = "hybrid_rrf"

    def search(self, query: str, k: int) -> list[str]:
        fused: dict[str, float] = {}
        for retriever in self.retrievers:
            for rank, doc_id in enumerate(retriever.search(query, self.depth), start=1):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (self.k0 + rank)
        return sorted(fused, key=lambda d: -fused[d])[:k]


class Rerank:
    """Cross-encoder rerank of the base retriever's top `depth` candidates."""

    def __init__(self, base: Retriever, docs: Sequence[Doc], score: Score,
                 depth: int = 20) -> None:
        self.base, self.score, self.depth = base, score, depth
        self.text = {d.id: _doc_text(d) for d in docs}
        self.name = f"{base.name}+rerank"

    def search(self, query: str, k: int) -> list[str]:
        candidates = self.base.search(query, self.depth)
        scores = self.score(query, [self.text[c] for c in candidates])
        order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
        return [candidates[i] for i in order[:k]]


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def fastembed_encoders(model: str = "BAAI/bge-small-en-v1.5") -> tuple[Embed, Embed]:
    from fastembed import TextEmbedding

    encoder = TextEmbedding(model)
    return (lambda texts: np.stack(list(encoder.passage_embed(list(texts), batch_size=64))),
            lambda texts: np.stack(list(encoder.query_embed(list(texts)))))


def fastembed_reranker(model: str = "Xenova/ms-marco-MiniLM-L-6-v2") -> Score:
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    encoder = TextCrossEncoder(model)
    return lambda query, texts: list(encoder.rerank(query, list(texts)))
