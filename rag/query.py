#!/usr/bin/env python3
"""
CyberRAG query pipeline — local retrieval-augmented generation.

Production path is 100% local: embed query (Ollama nomic-embed-text) -> retrieve
from Chroma -> grounded prompt -> generate (Ollama qwen2.5-coder:7b-64k).
No network egress.

Usage:
    python rag/query.py "How do I exploit SSTI in Jinja2?"
    python rag/query.py --no-rag "..."     # local model WITHOUT retrieval (baseline)
"""
import os
import sys
import argparse
import json

import chromadb
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import local_backend as ollama

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(ROOT, "data", "chroma")
COLLECTION = "cyber_kb"
EMBED_MODEL = os.getenv("CYBERRAG_EMBED_MODEL", "nomic-embed-text")
GEN_MODEL = os.getenv("CYBERRAG_GEN_MODEL", "qwen2.5-coder:7b")
TOP_K = 5

SYSTEM = (
    "You are a senior cybersecurity threat-intelligence analyst. Answer the "
    "question using ONLY the provided context passages. Cite the source of each "
    "claim inline as [doc]. If the context does not contain the answer, say "
    "'The knowledge base does not cover this' — do NOT invent details. Be precise "
    "and technical: exact techniques, payloads, CVE/ATT&CK IDs where present."
)


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION)


def retrieve(query, k=TOP_K):
    col = get_collection()
    qemb = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
    res = col.query(query_embeddings=[qemb], n_results=k,
                    include=["documents", "metadatas", "distances"])
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"doc": doc, "meta": meta, "score": round(1 - dist, 3)})
    return hits


def build_prompt(query, hits):
    ctx = "\n\n".join(
        f"[{h['meta'].get('doc','?')}] (relevance {h['score']})\n{h['doc']}"
        for h in hits
    )
    return f"Context passages:\n{ctx}\n\nQuestion: {query}\n\nAnswer (cite [doc] inline):"


def generate(query, use_rag=True, k=TOP_K):
    hits = retrieve(query, k) if use_rag else []
    if use_rag:
        user = build_prompt(query, hits)
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
        "sources": [{"doc": h["meta"].get("doc"), "source": h["meta"].get("source"),
                     "attack_ids": h["meta"].get("attack_ids"), "score": h["score"]}
                    for h in hits],
        "used_rag": use_rag,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--no-rag", action="store_true", help="baseline: local model, no retrieval")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("-k", type=int, default=TOP_K)
    args = ap.parse_args()

    out = generate(args.query, use_rag=not args.no_rag, k=args.k)
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print("\n=== ANSWER ===\n" + out["answer"])
    if out["sources"]:
        print("\n=== SOURCES ===")
        for s in out["sources"]:
            aid = f" ATT&CK:{s['attack_ids']}" if s.get("attack_ids") else ""
            print(f"  [{s['score']}] {s['doc']} ({s['source']}){aid}")


if __name__ == "__main__":
    main()
