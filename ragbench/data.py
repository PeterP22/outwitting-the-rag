"""SQuAD v1.1 dev as a retrieval corpus: 2,067 unique paragraphs, each question has one gold one."""

from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
SQUAD_SHA256 = "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9"


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Query:
    id: str
    question: str
    doc_id: str
    answers: tuple[str, ...]


def download(cache_dir: Path = Path(".cache")) -> Path:
    path = cache_dir / "squad-dev-v1.1.json"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SQUAD_URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SQUAD_SHA256:
        raise ValueError(f"unexpected SQuAD checksum {digest}")
    return path


def load_squad(path: Path) -> tuple[list[Doc], list[Query]]:
    raw = json.loads(path.read_text())
    docs: list[Doc] = []
    queries: list[Query] = []
    for article in raw["data"]:
        title = article["title"].replace("_", " ")
        for i, para in enumerate(article["paragraphs"]):
            doc_id = f"{article['title']}#{i}"
            docs.append(Doc(doc_id, title, para["context"]))
            for qa in para["qas"]:
                answers = tuple(dict.fromkeys(a["text"] for a in qa["answers"]))
                queries.append(Query(qa["id"], qa["question"], doc_id, answers))
    return docs, queries


def sample(queries: list[Query], n: int, seed: int = 0) -> list[Query]:
    return random.Random(seed).sample(queries, min(n, len(queries)))
