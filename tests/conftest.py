import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Tests never call a real LLM."""
    import openai

    def refuse(*args, **kwargs):
        raise AssertionError("tests must not construct a real OpenAI client")

    monkeypatch.setattr(openai, "OpenAI", refuse)
