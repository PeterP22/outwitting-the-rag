"""End-to-end generation scoring with a mocked LLM: no network, deterministic cost math."""

from ragbench.data import Doc, Query, load_squad
from ragbench.llm import JUDGE_INSTRUCTIONS, Completion, Price, is_yes
from ragbench.run import evaluate_generation, evaluate_retrieval

DOCS = {
    "d1": Doc("d1", "Denver", "The Denver Broncos won Super Bowl 50."),
    "d2": Doc("d2", "Rhine", "The Rhine flows through Germany."),
}
QUERIES = [
    Query("q1", "Who won Super Bowl 50?", "d1", ("Denver Broncos",)),
    Query("q2", "Where does the Rhine flow?", "d2", ("Germany",)),
    Query("q3", "What is the capital of Mars?", "d2", ("none",)),
]


class Fixed:
    name = "fixed"

    def search(self, query, k):
        return ["d1", "d2"][:k]


class MockLLM:
    model = "mock"

    def __init__(self):
        self.calls = []

    def complete(self, instructions, prompt):
        self.calls.append(instructions)
        if instructions == JUDGE_INSTRUCTIONS:
            text = "yes" if "Denver Broncos" in prompt.split("Answer:")[1] else "no"
            return Completion(text, 50, 1, 0.1)
        if "Super Bowl" in prompt.split("Question:")[1]:
            return Completion("Denver Broncos", 100, 4, 0.2)
        if "Rhine" in prompt.split("Question:")[1]:
            return Completion("the Netherlands", 100, 4, 0.2)
        return Completion("unknown", 100, 1, 0.2)


def test_generation_metrics_and_cost():
    llm = MockLLM()
    out = evaluate_generation(Fixed(), QUERIES, DOCS, llm, Price(1.0, 10.0), workers=1)
    assert out["exact_match"] == 1 / 3
    assert abs(out["abstain_rate"] - 1 / 3) < 1e-12
    assert out["faithfulness"] == 0.5  # one supported, one hallucinated
    assert llm.calls.count(JUDGE_INSTRUCTIONS) == 2  # abstentions are not judged
    assert out["avg_input_tokens"] == 100
    expected = (300 * 1.0 + 9 * 10.0) / 1e6
    assert abs(out["generation_usd"] - expected) < 1e-12
    assert abs(out["usd_per_1k_queries"] - 1000 * expected / 3) < 1e-9


def test_retrieval_metrics():
    out = evaluate_retrieval(Fixed(), QUERIES[:2])
    assert out["recall@1"] == 0.5
    assert out["mrr@10"] == (1 + 0.5) / 2


def test_is_yes():
    assert is_yes("Yes.") and not is_yes("no")


def test_load_squad_shape(tmp_path):
    path = tmp_path / "squad.json"
    path.write_text(
        '{"data": [{"title": "Super_Bowl", "paragraphs": [{"context": "ctx", "qas": '
        '[{"id": "x", "question": "q?", "answers": [{"text": "a"}, {"text": "a"}]}]}]}]}'
    )
    docs, queries = load_squad(path)
    assert docs[0].title == "Super Bowl" and queries[0].doc_id == "Super_Bowl#0"
    assert queries[0].answers == ("a",)
