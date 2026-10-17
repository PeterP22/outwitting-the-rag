from ragbench import metrics


def test_recall_and_mrr():
    ranked = ["a", "b", "c"]
    assert metrics.recall_at_k(ranked, "b", 1) == 0.0
    assert metrics.recall_at_k(ranked, "b", 2) == 1.0
    assert metrics.reciprocal_rank(ranked, "c") == 1 / 3
    assert metrics.reciprocal_rank(ranked, "z") == 0.0


def test_squad_normalization_em_f1():
    assert metrics.exact_match("The Denver Broncos!", ["Denver Broncos"]) == 1.0
    assert metrics.exact_match("Broncos", ["Denver Broncos"]) == 0.0
    assert metrics.f1("Broncos", ["Denver Broncos"]) == 2 * (1 * 0.5) / 1.5
    assert metrics.f1("unknown", ["Denver Broncos", "Broncos"]) == 0.0


def test_percentile():
    assert metrics.percentile([1, 2, 3], 50) == 2.0
    assert metrics.percentile([], 50) == 0.0
