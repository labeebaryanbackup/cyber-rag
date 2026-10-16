#!/usr/bin/env python3
"""
CyberRAG unified engine — the production query path. 100% local.

Combines:
  1. Knowledge-graph multi-hop facts (for entity/relationship questions about
     APT groups, malware, techniques, mitigations) — rag/kg.py
  2. Hybrid retrieval (vector + BM25 + RRF + cross-encoder rerank) over the full corpus
     (ATT&CK, CWE/CAPEC, CISA KEV, Sigma, bug-hunting playbooks) — rag/hybrid.py
  3. Grounded generation with a local LLM (qwythos-ctf:64k) — citations required.

A query is routed to the KG when it mentions a known APT/malware/technique AND
asks a relational question ("which", "what techniques", "used by", "mitigate").
Otherwise it's pure hybrid RAG. KG facts + retrieved passages are BOTH given to
the generator so answers are grounded and traceable.

Usage:
  python rag/engine.py "Which techniques does APT29 use and how do I detect them?"
  python rag/engine.py --no-rag "..."     # baseline: local model, no retrieval/KG
  python rag/engine.py --source attack,kev "..."   # scope retrieval
"""
import os
import sys
import argparse
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import local_backend as ollama
from rag.hybrid import hybrid_retrieve
from rag import kg

GEN_MODEL = os.getenv("CYBERRAG_GEN_MODEL", "qwythos-ctf:64k")

SYSTEM = (
    "You are a senior cybersecurity threat-intelligence analyst. Answer using ONLY "
    "the provided KNOWLEDGE GRAPH FACTS and CONTEXT PASSAGES. Cite sources inline "
    "as [doc] for passages and [ATT&CK] for graph facts. If the material does not "
    "contain the answer, say 'The knowledge base does not cover this' — never invent "
    "CVE IDs, technique IDs, payloads, or attributions. Be precise and technical."
)

REL_HINTS = ("which", "what techniques", "what malware", "used by", "uses ",
             "mitigate", "detect", "associated with", "linked to", "group",
             "apt", "actor", "campaign", "hops")


def route_uses_kg(query):
    ql = query.lower()
    if not any(h in ql for h in REL_HINTS):
        return False
    node = kg.find_node(query)
    return node is not None


def answer(query, use_rag=True, source_filter=None, rerank=True, use_kg=True):
    t0 = time.time()
    kg_facts, kg_node = [], None
    passages = []

    if use_rag:
        if use_kg and route_uses_kg(query):
            kg_node, kg_facts = kg.multihop_context(query)
        passages = hybrid_retrieve(query, top_k=5, source_filter=source_filter, rerank=rerank)

    if use_rag:
        parts = []
        if kg_facts:
            parts.append("KNOWLEDGE GRAPH FACTS (MITRE ATT&CK):\n" + "\n".join(kg_facts))
        if passages:
            ctx = "\n\n".join(f"[{p['meta'].get('doc','?')}] (rel {p['score']})\n{p['doc']}"
                              for p in passages)
            parts.append("CONTEXT PASSAGES:\n" + ctx)
        user = "\n\n".join(parts) + f"\n\nQuestion: {query}\n\nAnswer (cite [doc]/[ATT&CK] inline):"
        system = SYSTEM
    else:
        user = query
        system = "You are a senior cybersecurity threat-intelligence analyst. Answer precisely and technically."

    resp = ollama.chat(model=GEN_MODEL, messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], options={"temperature": 0.1})

    return {
        "answer": resp["message"]["content"],
        "kg_node": kg_node,
        "kg_fact_count": len(kg_facts),
        "sources": [{"doc": p["meta"].get("doc"), "source": p["meta"].get("source"),
                     "attack_ids": p["meta"].get("attack_ids"), "score": p["score"]}
                    for p in passages],
        "used_rag": use_rag,
        "latency_s": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-kg", action="store_true")
    ap.add_argument("--source", default="", help="comma list: anthropic-cyber,bughunter,mitre-attack,cisa-kev,capec,sigma,vault-notes")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    sf = [s.strip() for s in args.source.split(",")] if args.source else None
    out = answer(args.query, use_rag=not args.no_rag, source_filter=sf,
                 rerank=not args.no_rerank, use_kg=not args.no_kg)
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print("\n=== ANSWER ===\n" + out["answer"])
    if out["kg_node"]:
        print(f"\n[KG] resolved entity: {out['kg_node']} ({out['kg_fact_count']} graph facts used)")
    if out["sources"]:
        print("\n=== SOURCES ===")
        for s in out["sources"]:
            aid = f" ATT&CK:{s['attack_ids']}" if s.get("attack_ids") else ""
            print(f"  [{s['score']}] {s['doc']} ({s['source']}){aid}")
    print(f"\n[{out['latency_s']}s, local-only]")


if __name__ == "__main__":
    main()
