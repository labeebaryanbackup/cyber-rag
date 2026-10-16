#!/usr/bin/env python3
"""
CyberRAG knowledge graph — multi-hop CTI reasoning over MITRE ATT&CK.

Builds a local NetworkX graph from ATT&CK STIX (no Neo4j needed — self-contained,
fits the cheap-local thesis). Nodes: techniques, groups (APTs), software/malware,
mitigations, tactics. Edges from STIX relationships: uses, mitigates, subtechnique-of.

Enables queries pure vector RAG CAN'T answer:
  - "Which groups use T1059?"            (technique <- group)
  - "What techniques does APT29 use?"    (group -> techniques)
  - "What malware is linked to Sandworm and what does it do?"  (2-hop)
  - "How do I mitigate the techniques used by FIN7?"           (group -> tech -> mitigation)

Build:  python rag/kg.py build
Query:  python rag/kg.py "techniques used by APT29"
"""
import os
import json
import pickle
import urllib.request
import re

import networkx as nx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KG_PATH = os.path.join(ROOT, "data", "attack_kg.pkl")
STIX_URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"


def _ext_id(obj, source="mitre-attack"):
    for r in obj.get("external_references", []):
        if r.get("source_name") == source:
            return r.get("external_id", "")
    return ""


def build():
    print("[kg] downloading ATT&CK STIX...")
    req = urllib.request.Request(STIX_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())

    G = nx.DiGraph()
    stix_id_to_node = {}   # STIX id -> our node key
    objs = data["objects"]

    # First pass: nodes
    for o in objs:
        t = o.get("type")
        if t == "attack-pattern":
            tid = _ext_id(o)
            if not tid:
                continue
            key = tid
            G.add_node(key, kind="technique", name=o.get("name", ""),
                       desc=(o.get("description", "") or "")[:500])
            stix_id_to_node[o["id"]] = key
        elif t == "intrusion-set":  # APT group
            gid = _ext_id(o)
            key = f"{gid}:{o.get('name','')}" if gid else o.get("name", "")
            G.add_node(key, kind="group", name=o.get("name", ""),
                       aliases=",".join(o.get("aliases", [])),
                       desc=(o.get("description", "") or "")[:500])
            stix_id_to_node[o["id"]] = key
        elif t in ("malware", "tool"):
            sid = _ext_id(o)
            key = f"{sid}:{o.get('name','')}" if sid else o.get("name", "")
            G.add_node(key, kind="software", swtype=t, name=o.get("name", ""),
                       desc=(o.get("description", "") or "")[:500])
            stix_id_to_node[o["id"]] = key
        elif t == "course-of-action":  # mitigation
            mid = _ext_id(o)
            key = f"{mid}:{o.get('name','')}" if mid else o.get("name", "")
            G.add_node(key, kind="mitigation", name=o.get("name", ""),
                       desc=(o.get("description", "") or "")[:500])
            stix_id_to_node[o["id"]] = key

    # Second pass: edges from relationships
    n_edges = 0
    for o in objs:
        if o.get("type") != "relationship":
            continue
        src = stix_id_to_node.get(o.get("source_ref"))
        dst = stix_id_to_node.get(o.get("target_ref"))
        if not src or not dst:
            continue
        rel = o.get("relationship_type", "related")
        G.add_edge(src, dst, rel=rel)
        n_edges += 1

    with open(KG_PATH, "wb") as f:
        pickle.dump(G, f)
    kinds = {}
    for _, d in G.nodes(data=True):
        kinds[d.get("kind")] = kinds.get(d.get("kind"), 0) + 1
    print(f"[kg] built: {G.number_of_nodes()} nodes ({kinds}), {n_edges} edges -> {KG_PATH}")


_G = None


def _load():
    global _G
    if _G is None:
        with open(KG_PATH, "rb") as f:
            _G = pickle.load(f)
    return _G


def find_node(query):
    """Fuzzy-match a node by name/alias/id from a free-text query.
    Priority: technique ID > exact group/alias word match > longest name contained."""
    G = _load()
    ql = query.lower()
    qwords = set(re.findall(r"[a-z0-9]+", ql))
    # 1. explicit technique ID
    m = re.search(r"\bT\d{4}(?:\.\d{3})?\b", query, re.I)
    if m and m.group().upper() in G:
        return m.group().upper()
    # 2 & 3: score candidates; require name length >= 3 to avoid matching "at","os"
    best = None
    best_len = 0
    for key, d in G.nodes(data=True):
        kind = d.get("kind")
        name = (d.get("name", "") or "").lower().strip()
        # group/software names: match as a whole phrase OR all name-words present
        candidates = [name]
        if d.get("aliases"):
            candidates += [a.strip().lower() for a in d["aliases"].split(",") if a.strip()]
        for cand in candidates:
            if len(cand) < 3:
                continue
            cwords = set(re.findall(r"[a-z0-9]+", cand))
            # whole phrase appears, or every word of the candidate name is in the query
            if (cand in ql or (cwords and cwords <= qwords)):
                # prefer groups/software over generic technique names; prefer longer matches
                weight = len(cand) + (5 if kind in ("group", "software") else 0)
                if weight > best_len:
                    best_len = weight
                    best = key
    return best


def neighbors(node, rel=None, direction="out"):
    G = _load()
    out = []
    edges = G.out_edges(node, data=True) if direction == "out" else G.in_edges(node, data=True)
    for u, v, d in edges:
        if rel and d.get("rel") != rel:
            continue
        other = v if direction == "out" else u
        nd = G.nodes[other]
        out.append({"node": other, "kind": nd.get("kind"), "name": nd.get("name"), "rel": d.get("rel")})
    return out


def multihop_context(query):
    """Resolve a CTI question into graph facts to feed the RAG generator."""
    G = _load()
    node = find_node(query)
    if not node:
        return None, []
    d = G.nodes[node]
    facts = [f"Entity: {node} ({d.get('kind')}) — {d.get('name')}"]
    if d.get("desc"):
        facts.append(f"Description: {d['desc']}")
    # outgoing: what this entity uses/mitigates
    for n in neighbors(node, direction="out")[:25]:
        facts.append(f"{d.get('name')} --{n['rel']}--> {n['name']} ({n['kind']})")
    # incoming: who uses this technique / what mitigates it
    for n in neighbors(node, direction="in")[:25]:
        facts.append(f"{n['name']} ({n['kind']}) --{n['rel']}--> {d.get('name')}")
    return node, facts


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build()
    else:
        q = sys.argv[1] if len(sys.argv) > 1 else "techniques used by APT29"
        node, facts = multihop_context(q)
        print(f"resolved node: {node}\n")
        for f in facts:
            print(" ", f)
