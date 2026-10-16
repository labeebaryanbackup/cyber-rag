"""Benchmark retrieval configs against standard IR metrics.

The honest "does the fancy technique actually help?" harness. Measures hit@k /
recall@k / MRR / nDCG@k and per-query latency for: baseline hybrid, cross-encoder
rerank, LLM-as-reranker, HyDE query rewriting, and multi-query expansion.

Usage:  python eval/bench_retrieval.py            # all configs
        python eval/bench_retrieval.py --limit 3  # quick smoke (note: cross-encoder
                                                  # downloads ~1-2GB weights on first use)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.hybrid import hybrid_retrieve
from eval.retrieval_metrics import evaluate_retrieval

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_configs(top_k):
    return {
        "baseline": lambda q: hybrid_retrieve(q, top_k=top_k, rerank=False),
        "cross-enc": lambda q: hybrid_retrieve(q, top_k=top_k, rerank=True, rerank_method="cross"),
        "llm-rerank": lambda q: hybrid_retrieve(q, top_k=top_k, rerank=True, rerank_method="llm"),
        "hyde": lambda q: hybrid_retrieve(q, top_k=top_k, rerank=False, rewrite="hyde"),
        "multi": lambda q: hybrid_retrieve(q, top_k=top_k, rerank=False, rewrite="multi"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(os.path.join(ROOT, "eval", "questions.json"), encoding="utf-8") as f:
        questions = json.load(f)
    if args.limit:
        questions = questions[: args.limit]

    configs = build_configs(args.top_k)
    header = f"{'config':<12}" + "".join(f"{m:>10}" for m in ["hit@k", "recall@k", "MRR", "nDCG@k", "latency_s"])
    print(header)
    print("-" * len(header))

    results = {}
    for name, fn in configs.items():
        t0 = time.time()
        r = evaluate_retrieval(questions, top_k=args.top_k, retrieve_fn=fn)
        dt = round(time.time() - t0, 1)
        results[name] = {**r, "latency_s": dt}
        print(f"{name:<12}" + f"{r['hit@k']:>10}{r['recall@k']:>10}{r['MRR']:>10}{r['nDCG@k']:>10}{dt:>10}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
