import numpy as np

from ragbench.data import Doc
from ragbench.retrievers import BM25, Dense, HybridRRF, Rerank

DOCS = [
    Doc("d1", "Denver", "The Denver Broncos won Super Bowl 50 in Santa Clara."),
    Doc("d2", "Rivers", "The Rhine river flows through Germany and the Netherlands."),
    Doc("d3", "Physics", "Oxygen is a chemical element with symbol O."),
]
VOCAB = ["broncos", "super", "bowl", "rhine", "river", "germany", "oxygen", "element"]


def bag(texts):
    return np.array([[t.lower().count(w) for w in VOCAB] for t in texts], dtype=float) + 1e-3


class Fixed:
    def __init__(self, name, ranking):
        self.name, self.ranking = name, ranking

    def search(self, query, k):
        return self.ranking[:k]


def test_bm25_finds_keyword_match():
    assert BM25(DOCS).search("Which river flows through Germany?", 1) == ["d2"]


def test_bm25_handles_stopword_only_query():
    assert BM25(DOCS).search("the of and", 3) == []


def test_dense_uses_cosine_similarity():
    dense = Dense(DOCS, bag, bag)
    assert dense.search("oxygen element", 2)[0] == "d3"


def test_rrf_rewards_agreement():
    a = Fixed("a", ["d1", "d2", "d3"])
    b = Fixed("b", ["d2", "d3", "d1"])
    assert HybridRRF([a, b]).search("q", 3)[0] == "d2"


def test_rerank_reorders_candidates():
    base = Fixed("base", ["d1", "d2", "d3"])
    scores = {"d1": 0.1, "d2": 0.2, "d3": 0.9}
    text_to_id = {f"{d.title}. {d.text}": d.id for d in DOCS}
    reranker = Rerank(base, DOCS, lambda q, texts: [scores[text_to_id[t]] for t in texts])
    assert reranker.search("q", 2) == ["d3", "d2"]
    assert reranker.name == "base+rerank"
