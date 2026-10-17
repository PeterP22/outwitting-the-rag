"""Run the benchmark: python -m ragbench.run [--retrieval-queries N] [--gen-queries M]."""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ragbench import metrics
from ragbench.data import Doc, Query, download, load_squad, sample
from ragbench.llm import (
    ANSWER_INSTRUCTIONS,
    JUDGE_INSTRUCTIONS,
    LLM,
    PRICES,
    Completion,
    OpenAIResponses,
    Price,
    answer_prompt,
    is_yes,
    judge_prompt,
)
from ragbench.retrievers import (
    BM25,
    Dense,
    HybridRRF,
    Rerank,
    Retriever,
    fastembed_encoders,
    fastembed_reranker,
)


def evaluate_retrieval(retriever: Retriever, queries: Sequence[Query], k: int = 10) -> dict:
    rows, latencies = [], []
    for q in queries:
        start = time.perf_counter()
        ranked = retriever.search(q.question, k)
        latencies.append((time.perf_counter() - start) * 1000)
        rows.append(ranked)
    n = len(queries)
    pairs = list(zip(rows, queries, strict=True))
    return {
        "recall@1": sum(metrics.recall_at_k(r, q.doc_id, 1) for r, q in pairs) / n,
        "recall@5": sum(metrics.recall_at_k(r, q.doc_id, 5) for r, q in pairs) / n,
        "recall@10": sum(metrics.recall_at_k(r, q.doc_id, 10) for r, q in pairs) / n,
        "mrr@10": sum(metrics.reciprocal_rank(r, q.doc_id) for r, q in pairs) / n,
        "latency_ms_p50": metrics.percentile(latencies, 50),
        "latency_ms_p95": metrics.percentile(latencies, 95),
    }


def evaluate_generation(retriever: Retriever, queries: Sequence[Query], docs: dict[str, Doc],
                        llm: LLM, price: Price, top_k: int = 3, workers: int = 8) -> dict:
    retrieved = {q.id: [docs[d].text for d in retriever.search(q.question, top_k)]
                 for q in queries}

    def one(q: Query) -> tuple[Completion, Completion | None]:
        answer = llm.complete(ANSWER_INSTRUCTIONS, answer_prompt(q.question, retrieved[q.id]))
        if answer.text.lower().strip(" .") == "unknown":
            return answer, None
        judge = llm.complete(JUDGE_INSTRUCTIONS, judge_prompt(answer.text, retrieved[q.id]))
        return answer, judge

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(one, queries))

    n = len(queries)
    pairs = list(zip(queries, results, strict=True))
    answers = [a for a, _ in results]
    judges = [j for _, j in results if j is not None]
    gen_in = sum(a.input_tokens for a in answers)
    gen_out = sum(a.output_tokens for a in answers)
    judge_in = sum(j.input_tokens for j in judges)
    judge_out = sum(j.output_tokens for j in judges)
    fresh = [c for c in [*answers, *judges] if not c.cached]
    return {
        "queries": n,
        "exact_match": sum(metrics.exact_match(a.text, q.answers) for q, (a, _) in pairs) / n,
        "f1": sum(metrics.f1(a.text, q.answers) for q, (a, _) in pairs) / n,
        "abstain_rate": 1 - len(judges) / n,
        "faithfulness": sum(is_yes(j.text) for j in judges) / max(len(judges), 1),
        "llm_latency_s_p50": metrics.percentile([a.seconds for a in answers], 50),
        "avg_input_tokens": gen_in / n,
        "avg_output_tokens": gen_out / n,
        "usd_per_1k_queries": 1000 * price.cost(gen_in, gen_out) / n,
        "generation_usd": price.cost(gen_in, gen_out),
        "judge_usd": price.cost(judge_in, judge_out),
        "fresh_usd": price.cost(sum(c.input_tokens for c in fresh),
                                sum(c.output_tokens for c in fresh)),
    }


def chart(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results["setups"])
    recall = [results["setups"][n]["retrieval"]["recall@5"] for n in names]
    f1 = [results["setups"][n].get("generation", {}).get("f1", 0) for n in names]
    x = list(range(len(names)))
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=150)
    ax.bar([i - 0.2 for i in x], recall, 0.4, label="recall@5", color="#9ca3af")
    ax.bar([i + 0.2 for i in x], f1, 0.4, label="answer F1", color="#4f46e5")
    for i, (r, f) in enumerate(zip(recall, f1, strict=True)):
        ax.text(i - 0.2, r + 0.01, f"{r:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.2, f + 0.01, f"{f:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.08)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def build_retrievers(docs: Sequence[Doc]) -> list[Retriever]:
    bm25 = BM25(docs)
    embed_docs, embed_query = fastembed_encoders()
    dense = Dense(docs, embed_docs, embed_query)
    hybrid = HybridRRF([bm25, dense])
    return [bm25, dense, hybrid, Rerank(hybrid, docs, fastembed_reranker())]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-queries", type=int, default=1000)
    parser.add_argument("--gen-queries", type=int, default=200)
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument("--effort", default="none")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/results.json"))
    parser.add_argument("--chart", type=Path, default=Path("results/chart.png"))
    args = parser.parse_args(argv)

    docs, queries = load_squad(download())
    by_id = {d.id: d for d in docs}
    eval_queries = sample(queries, args.retrieval_queries, args.seed)
    gen_queries = eval_queries[: args.gen_queries]
    llm = OpenAIResponses(args.model, args.effort) if args.gen_queries else None
    price = PRICES[args.model]

    start = time.perf_counter()
    retrievers = build_retrievers(docs)
    results: dict = {
        "dataset": "SQuAD v1.1 dev", "corpus_docs": len(docs),
        "retrieval_queries": len(eval_queries), "generation_queries": len(gen_queries),
        "model": args.model, "reasoning_effort": args.effort, "top_k_contexts": 3,
        "price_usd_per_1m": {"input": price.input, "output": price.output},
        "index_build_s": round(time.perf_counter() - start, 1),
        "machine": platform.machine(), "setups": {},
    }
    for retriever in retrievers:
        entry = {"retrieval": evaluate_retrieval(retriever, eval_queries)}
        if llm is not None:
            entry["generation"] = evaluate_generation(retriever, gen_queries, by_id, llm, price)
        results["setups"][retriever.name] = entry
        print(retriever.name, json.dumps(entry), flush=True)
    results["total_fresh_usd"] = sum(s.get("generation", {}).get("fresh_usd", 0)
                                     for s in results["setups"].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    chart(results, args.chart)


if __name__ == "__main__":
    main()
