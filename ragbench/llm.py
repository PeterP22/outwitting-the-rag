"""LLM calls behind a tiny protocol so tests can mock them. Token usage drives the cost math."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

ANSWER_INSTRUCTIONS = (
    "Answer the question using only the context. Reply with the shortest exact phrase from "
    "the context that answers it, not a sentence. If the context does not contain the answer, "
    "reply exactly: unknown"
)
JUDGE_INSTRUCTIONS = (
    "You verify whether an answer is fully supported by the context. "
    "Reply with exactly one word: yes or no."
)


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    seconds: float
    cached: bool = False


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens."""

    input: float
    output: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input + output_tokens * self.output) / 1e6


# https://developers.openai.com/api/docs/pricing (checked 2026-10-17)
PRICES = {"gpt-5.4-nano": Price(0.20, 1.25), "gpt-5-nano": Price(0.05, 0.40)}


class LLM(Protocol):
    model: str

    def complete(self, instructions: str, prompt: str) -> Completion: ...


class OpenAIResponses:
    def __init__(self, model: str = "gpt-5.4-nano", effort: str = "none",
                 cache_path: Path | None = Path(".cache/llm.jsonl")) -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model, self.effort = model, effort
        self.cache_path = cache_path
        self.cache: dict[str, dict] = {}
        if cache_path and cache_path.exists():
            for line in cache_path.read_text().splitlines():
                row = json.loads(line)
                self.cache[row["key"]] = row

    def complete(self, instructions: str, prompt: str) -> Completion:
        import time

        key = hashlib.sha256(f"{self.model}|{self.effort}|{instructions}|{prompt}".encode())
        key_hex = key.hexdigest()
        if key_hex in self.cache:
            row = self.cache[key_hex]
            return Completion(row["text"], row["in"], row["out"], row["s"], cached=True)
        start = time.perf_counter()
        response = self.client.responses.create(
            model=self.model, instructions=instructions, input=prompt,
            reasoning={"effort": self.effort}, max_output_tokens=200,
        )
        seconds = time.perf_counter() - start
        usage = response.usage
        out = Completion(response.output_text.strip(), usage.input_tokens, usage.output_tokens,
                         seconds)
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            row = {"key": key_hex, "text": out.text, "in": out.input_tokens,
                   "out": out.output_tokens, "s": seconds}
            with self.cache_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            self.cache[key_hex] = row
        return out


def answer_prompt(question: str, contexts: list[str]) -> str:
    joined = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return f"Context:\n{joined}\n\nQuestion: {question}"


def judge_prompt(answer: str, contexts: list[str]) -> str:
    joined = "\n\n".join(contexts)
    return f"Context:\n{joined}\n\nAnswer: {answer}\n\nIs the answer supported by the context?"


def is_yes(text: str) -> bool:
    return text.strip().lower().startswith("yes")
