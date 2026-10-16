"""Standard, rank-aware retrieval metrics for CyberRAG.

The existing eval reports a single binary ``context_hit`` (did ANY expected doc
appear in the top-5). That's a start but it can't tell "found it at rank 1" from
"found it at rank 5", nor measure multi-relevant queries. These are the standard
IR measures that let you compare retrieval configs rigorously:

- ``hit@k``    — fraction of queries with >=1 relevant doc in the top-k
- ``recall@k`` — fraction of expected docs surfaced (approximate: labels are
                 substring-based, so multiple labels can name the same doc)
- ``MRR``      — mean reciprocal rank of the FIRST relevant doc (rewards
                 finding it early)
- ``nDCG@k``   — rank-aware gain, normalized against the ideal ranking

All metrics are config-agnostic: pass any ``retrieve_fn(query) -> list[chunk]``
and they measure whatever retrieval pipeline you point them at (baseline,
+query-rewriting, +cross-encoder, ...).
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.hybrid import hybrid_retrieve


def is_relevant(meta: dict, expected_docs: list[str]) -> bool:
    """A retrieved chunk is relevant if its doc name matches an expected label.

    Labels are substrings (e.g. ``T1059``, ``exploiting-sql-injection``); match
    case-insensitively against the chunk's ``doc`` metadata field, either
    direction (label-in-doc or doc-in-label).
    """
    doc = str(meta.get("doc", "")).lower()
    return any(e.lower() in doc or doc in e.lower() for e in expected_docs)


def _labels(retrieved: list[dict], expected_docs: list[str]) -> list[int]:
    return [1 if is_relevant(m["meta"], expected_docs) else 0 for m in retrieved]


def _dcg(rels: list[int], k: int) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def hit_at_k(retrieved: list[dict], expected_docs: list[str], k: int) -> int:
    return int(any(_labels(retrieved, expected_docs)[:k]))


def recall_at_k(retrieved: list[dict], expected_docs: list[str], k: int) -> float:
    """Fraction of DISTINCT expected docs matched by the top-k retrieved set.

    A retrieved chunk counts toward the label only once, even if it matches
    several expected substrings (labels are approximate — multiple substrings
    can name the same underlying doc).
    """
    retrieved_docs = [str(m["meta"].get("doc", "")).lower() for m in retrieved[:k]]
    matched = sum(
        1 for e in expected_docs
        if any(e.lower() in d or d in e.lower() for d in retrieved_docs)
    )
    return round(matched / max(1, len(expected_docs)), 4)


def mrr(retrieved: list[dict], expected_docs: list[str]) -> float:
    for i, m in enumerate(retrieved):
        if is_relevant(m["meta"], expected_docs):
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[dict], expected_docs: list[str], k: int) -> float:
    rels = _labels(retrieved, expected_docs)
    idcg = _dcg(sorted(rels, reverse=True), k)
    return round(_dcg(rels, k) / idcg, 4) if idcg > 0 else 0.0


def evaluate_retrieval(questions: list[dict], top_k: int = 5, retrieve_fn=None) -> dict:
    """Run hit@k / recall@k / MRR / nDCG@k over a labelled question set.

    ``retrieve_fn(query) -> list[chunk]`` defaults to the current hybrid
    retrieval with reranking off (matching how the eval measures raw retrieval).
    """
    retrieve_fn = retrieve_fn or (lambda q: hybrid_retrieve(q, top_k=top_k, rerank=False))

    hits, recalls, mrrs, ndcgs = [], [], [], []
    for q in questions:
        expected = q.get("expected_docs", [])
        if not expected:
            continue
        ret = retrieve_fn(q["question"])
        hits.append(hit_at_k(ret, expected, top_k))
        recalls.append(recall_at_k(ret, expected, top_k))
        mrrs.append(mrr(ret, expected))
        ndcgs.append(ndcg_at_k(ret, expected, top_k))

    n = len(hits)
    return {
        "n": n,
        "k": top_k,
        "hit@k": round(sum(hits) / n, 4) if n else None,
        "recall@k": round(sum(recalls) / n, 4) if n else None,
        "MRR": round(sum(mrrs) / n, 4) if n else None,
        "nDCG@k": round(sum(ndcgs) / n, 4) if n else None,
    }


if __name__ == "__main__":
    import json

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "eval", "questions.json"), encoding="utf-8") as f:
        questions = json.load(f)
    result = evaluate_retrieval(questions, top_k=5)
    for k, v in result.items():
        print(f"{k:<12} {v}")
