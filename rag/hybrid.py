#!/usr/bin/env python3
"""
Hybrid retrieval for CyberRAG — vector + BM25 + reciprocal-rank fusion + LLM rerank.

Why hybrid: dense vectors catch semantic matches; BM25 catches exact terms (CVE
IDs, technique IDs, tool names) that embeddings often miss. RRF fuses both rank
lists; an optional LLM rerank promotes the truly relevant chunks to the top.
All local. Supports source_filter (e.g. ['mitre-attack','cisa-kev']).
"""
import os
import re
import pickle

import chromadb
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import local_backend as ollama
from rank_bm25 import BM25Okapi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(ROOT, "data", "chroma")
BM25_PATH = os.path.join(ROOT, "data", "bm25.pkl")
COLLECTION = "cyber_kb"
EMBED_MODEL = os.getenv("CYBERRAG_EMBED_MODEL", "nomic-embed-text")
GEN_MODEL = os.getenv("CYBERRAG_GEN_MODEL", "qwythos-ctf:64k")

_tok = re.compile(r"[A-Za-z0-9_.\-]+")

# Common English stopwords. Stripped from QUERIES only (not from the indexed
# corpus) so that rare, high-signal tokens — CVE IDs, ATT&CK technique IDs, tool
# and malware names — dominate BM25 scoring instead of being drowned out by
# matches on "what/is/the/how" across thousands of documents.
_STOP = frozenset("""a an and are as at be been but by can could do does for from
had has have how i if in into is it its more most no not of on or our should so
such that the their then there these they this to use using was what when where
which who why will with would you your about above after again against""".split())


def tokenize(text):
    return [t.lower() for t in _tok.findall(text)]


def tokenize_query(text):
    """Tokenize a query and drop stopwords so rare identifiers dominate BM25."""
    toks = [t for t in tokenize(text) if t not in _STOP and len(t) > 1]
    return toks or tokenize(text)  # never return empty


def build_bm25():
    """One-time: pull all docs from Chroma, build a BM25 index, pickle it."""
    col = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION)
    got = col.get(include=["documents", "metadatas"])
    docs = got["documents"]
    ids = got["ids"]
    metas = got["metadatas"]
    bm25 = BM25Okapi([tokenize(d) for d in docs])
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids, "docs": docs, "metas": metas}, f)
    print(f"[bm25] built over {len(docs)} chunks -> {BM25_PATH}")


_cache = {}


def _load_bm25():
    if "bm25" not in _cache:
        with open(BM25_PATH, "rb") as f:
            _cache["bm25"] = pickle.load(f)
    return _cache["bm25"]


def _col():
    if "col" not in _cache:
        _cache["col"] = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION)
    return _cache["col"]


def _generate(prompt, num_predict=140):
    """One local LLM call (temperature 0); returns stripped text or '' on failure."""
    try:
        r = ollama.generate(model=GEN_MODEL, prompt=prompt,
                            options={"temperature": 0, "num_predict": num_predict})
        return (r.get("response") or "").strip()
    except Exception:
        return ""


def hyde_document(query):
    """HyDE (Hypothetical Document Embeddings): generate the answer a real report
    would contain, then embed *that* for vector search.

    A question ("how do I ...") and the relevant doc ("the attacker ...") use
    different words, so embedding the bare query often misses. The fake answer
    borrows the document's vocabulary (technique IDs, tool names, procedures),
    bridging that gap. BM25 still gets the original query (exact IDs)."""
    prompt = (
        "Write a short, dense, technical paragraph (4-6 sentences) that a senior "
        "cybersecurity analyst would write to answer this question, including the "
        "specific ATT&CK technique IDs (T####), CVE numbers, tool names, and "
        "procedures a real report would mention.\n\nQuestion: " + query + "\n\nAnswer:"
    )
    return _generate(prompt)


def multi_queries(query, n=2):
    """Query expansion: generate `n` alternative phrasings of the same intent."""
    prompt = (
        f"Rewrite the following search query {n} different ways, each on its own "
        f"line, keeping the same intent but different wording. Output only the "
        f"rewrites, one per line, no numbering.\n\nQuery: {query}\n\nRewrites:"
    )
    lines = [l.strip(" -•\t") for l in _generate(prompt).splitlines() if l.strip()]
    return [query] + [l for l in lines if l][:n]


def vector_search(query, k, source_filter=None, qtext=None):
    """Dense retrieval. ``qtext`` overrides what gets embedded (e.g. query+HyDE)."""
    qemb = ollama.embeddings(model=EMBED_MODEL, prompt=qtext or query)["embedding"]
    where = {"source": {"$in": source_filter}} if source_filter else None
    res = _col().query(query_embeddings=[qemb], n_results=k, where=where,
                       include=["documents", "metadatas", "distances"])
    out = []
    for i, (doc, meta, dist) in enumerate(zip(res["documents"][0], res["metadatas"][0], res["distances"][0])):
        out.append({"id": res["ids"][0][i], "doc": doc, "meta": meta, "vscore": 1 - dist})
    return out


def bm25_search(query, k, source_filter=None):
    store = _load_bm25()
    scores = store["bm25"].get_scores(tokenize_query(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out = []
    for i in ranked:
        meta = store["metas"][i]
        if source_filter and meta.get("source") not in source_filter:
            continue
        out.append({"id": store["ids"][i], "doc": store["docs"][i], "meta": meta, "bscore": scores[i]})
        if len(out) >= k:
            break
    return out


_ID_RE = re.compile(r"\b(?:CVE-\d{4}-\d{4,7}|T\d{4}(?:\.\d{3})?|CAPEC-\d+|CWE-\d+)\b", re.I)


def _query_ids(query):
    return set(m.upper() for m in _ID_RE.findall(query))


def rrf_fuse(vec, bm, k=60, query=None):
    """Reciprocal Rank Fusion of two ranked lists, with an exact-identifier boost.

    If the query names a hard identifier (CVE / ATT&CK / CAPEC / CWE), any
    candidate whose text or metadata contains that exact ID is promoted — an
    exact ID match is a near-certain relevance signal that plain BM25 can let
    slip when common query words inflate unrelated docs.
    """
    ids = _query_ids(query) if query else set()
    rank = {}
    for lst in (vec, bm):
        for r, item in enumerate(lst):
            rank.setdefault(item["id"], {"item": item, "score": 0.0})
            rank[item["id"]]["score"] += 1.0 / (k + r + 1)
    if ids:
        for entry in rank.values():
            it = entry["item"]
            hay = (it.get("doc", "") + " " + str(it.get("meta", {}).get("attack_ids", "")) +
                   " " + str(it.get("meta", {}).get("cve_ids", "")) +
                   " " + str(it.get("meta", {}).get("doc", ""))).upper()
            if any(i in hay for i in ids):
                entry["score"] += 1.0  # dominant boost: exact ID match wins
    fused = sorted(rank.values(), key=lambda x: x["score"], reverse=True)
    return [f["item"] for f in fused]


def llm_rerank(query, candidates, top_n):
    """Lightweight local rerank: ask the LLM to score each candidate 0-10.
    Keeps the fused RRF order as a tiebreaker so rerank only ever helps."""
    for idx, c in enumerate(candidates):
        prompt = (f"On a scale of 0 to 10, how relevant is the passage to the question? "
                  f"Answer with ONLY a single integer 0-10, nothing else.\n\n"
                  f"Question: {query}\n\nPassage:\n{c['doc'][:700]}\n\nScore (0-10):")
        try:
            r = ollama.generate(model=GEN_MODEL, prompt=prompt,
                                options={"temperature": 0, "num_predict": 6, "stop": ["\n"]})
            m = re.search(r"\b(10|[0-9])\b", r["response"])
            c["rerank"] = int(m.group()) if m else None
        except Exception:
            c["rerank"] = None
        # fused-order tiebreaker: earlier RRF rank = small bonus, keeps stable order
        c["_fused_rank"] = idx
    # if the LLM gave usable scores, sort by them; else fall back to fused order
    if any(c.get("rerank") is not None for c in candidates):
        candidates.sort(key=lambda x: (x.get("rerank") if x.get("rerank") is not None else -1,
                                       -x["_fused_rank"]), reverse=True)
    return candidates[:top_n]


_reranker = None


def _get_reranker():
    """Lazy-load the cross-encoder (downloads ~1GB of weights on first use)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return _reranker


def cross_encoder_rerank(query, candidates, top_n):
    """Proper cross-encoder rerank (bge-reranker-v2-m3).

    Unlike the LLM-as-reranker, a cross-encoder scores query+passage JOINTLY in
    one forward pass — faster, deterministic, and the industry-standard 2nd
    stage. Runs on CPU (no GPU needed to rerank a dozen candidates)."""
    model = _get_reranker()
    pairs = [(query, c["doc"][:700]) for c in candidates]
    try:
        scores = model.predict(pairs)
    except Exception:
        return llm_rerank(query, candidates, top_n)  # graceful fallback
    for c, s in zip(candidates, scores):
        c["cross"] = float(s)
    candidates.sort(key=lambda x: x.get("cross", 0.0), reverse=True)
    return candidates[:top_n]


def id_lookup(ids, source_filter=None):
    """Directly fetch the canonical doc for an exact identifier (CVE/ATT&CK).
    The authoritative corpus names these docs `<ID>.md` (e.g. CVE-2021-44228.md,
    T1059.md), so we match on doc name — guarantees the canonical doc is a
    candidate even if vector+BM25 both miss it (common for opaque ID tokens)."""
    if not ids:
        return []
    col = _col()
    out, seen = [], set()
    for ident in ids:
        candidates = [f"{ident}.md", f"{ident.upper()}.md"]
        for docname in set(candidates):
            try:
                got = col.get(where={"doc": {"$eq": docname}},
                              include=["documents", "metadatas"], limit=4)
            except Exception:
                continue
            for i, cid in enumerate(got.get("ids", [])):
                if cid in seen:
                    continue
                meta = got["metadatas"][i]
                if source_filter and meta.get("source") not in source_filter:
                    continue
                seen.add(cid)
                out.append({"id": cid, "doc": got["documents"][i], "meta": meta, "idmatch": True})
    return out


def _final_score(f: dict) -> float:
    """Best available score for display (rerank > cross > vector > bm25)."""
    for key in ("rerank", "cross", "vscore", "bscore"):
        v = f.get(key)
        if v is not None:
            return round(float(v), 3)
    return 0.0


def hybrid_retrieve(query, top_k=5, pool=12, source_filter=None, rerank=True,
                    rerank_method="cross", rewrite=None):
    """Main entry: vector + BM25 (+ exact-ID lookup) -> RRF -> rerank -> top_k.

    ``rerank_method``: ``"llm"`` (LLM-as-reranker) or ``"cross"`` (cross-encoder).

    ``rewrite`` selects query rewriting (applied to the VECTOR side only — BM25
    always gets the original query for exact identifiers):

    - ``None``    — current behaviour (embed the bare query)
    - ``"hyde"``  — embed query + a generated hypothetical answer document
    - ``"multi"`` — generate alternative phrasings, vector-search each, dedupe
    """
    if rewrite == "hyde":
        hyde = hyde_document(query)
        vec = vector_search(query, pool, source_filter, qtext=(query + "\n\n" + hyde) if hyde else query)
    elif rewrite == "multi":
        vec, seen = [], set()
        for q in multi_queries(query):
            for item in vector_search(q, pool, source_filter):
                if item["id"] not in seen:
                    seen.add(item["id"])
                    vec.append(item)
    else:
        vec = vector_search(query, pool, source_filter)
    bm = bm25_search(query, pool, source_filter)
    ids = _query_ids(query)
    # guarantee canonical ID docs are in the candidate pool
    direct = id_lookup(ids, source_filter)
    if direct:
        # prepend so RRF sees them at rank 0 in a third list
        vec = direct + vec
    fused = rrf_fuse(vec, bm, query=query)[:pool]
    if rerank and fused:
        fused = cross_encoder_rerank(query, fused, top_k) if rerank_method == "cross" else llm_rerank(query, fused, top_k)
    else:
        fused = fused[:top_k]
    return [{"doc": f["doc"], "meta": f["meta"], "score": _final_score(f)} for f in fused]


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_bm25()
    else:
        q = sys.argv[1] if len(sys.argv) > 1 else "What is T1059 and how to detect it?"
        for h in hybrid_retrieve(q, source_filter=None):
            print(f"[{h['score']}] {h['meta'].get('doc')} ({h['meta'].get('source')})")
