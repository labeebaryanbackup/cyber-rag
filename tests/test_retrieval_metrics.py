"""Unit tests for the IR retrieval metrics (hit@k / recall@k / MRR / nDCG@k)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.retrieval_metrics import hit_at_k, mrr, ndcg_at_k, recall_at_k


def _chunk(doc):
    return {"doc": "...", "meta": {"doc": doc}}


def test_hit_and_mrr():
    retrieved = [_chunk("T1003.md"), _chunk("other.md"), _chunk("T1059.md")]
    assert hit_at_k(retrieved, ["T1059"], 5) == 1
    assert mrr(retrieved, ["T1059"]) == 1 / 3  # first relevant at rank 3


def test_miss():
    retrieved = [_chunk("a.md"), _chunk("b.md")]
    assert hit_at_k(retrieved, ["zzz"], 5) == 0
    assert mrr(retrieved, ["zzz"]) == 0.0


def test_recall_counts_distinct_docs():
    retrieved = [_chunk("T1003.md"), _chunk("T1059.md")]
    assert recall_at_k(retrieved, ["T1003", "T1059"], 5) == 1.0  # both found
    assert recall_at_k([_chunk("T1003.md")], ["T1003", "T1059"], 5) == 0.5  # one of two


def test_ndcg_rewards_early_relevant_rank():
    good = [_chunk("T1059.md"), _chunk("x.md"), _chunk("y.md"), _chunk("z.md"), _chunk("w.md")]
    bad = [_chunk("x.md"), _chunk("y.md"), _chunk("z.md"), _chunk("w.md"), _chunk("T1059.md")]
    assert ndcg_at_k(good, ["T1059"], 5) > ndcg_at_k(bad, ["T1059"], 5)
    assert ndcg_at_k(good, ["T1059"], 5) == 1.0  # ideal ranking


def test_substring_match_is_bidirectional():
    # label "exploiting-sql-injection" should match doc "exploiting-sql-injection-vulnerabilities"
    assert hit_at_k([_chunk("exploiting-sql-injection-vulnerabilities.md")], ["exploiting-sql-injection"], 5) == 1
    assert hit_at_k([_chunk("sqli.md")], ["sqli"], 5) == 1
