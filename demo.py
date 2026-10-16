#!/usr/bin/env python3
"""
CyberRAG demo CLI — interactive local cybersecurity analyst.

A clean front-end over the unified engine for live demos. Shows the answer,
which knowledge-graph entity (if any) was resolved, and the cited sources —
proving every claim is grounded and that nothing left the local network.

Run:  python demo.py
      python demo.py "Which techniques does APT29 use?"
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag.engine import answer

BANNER = r"""
  ____      _               ____      _    ____
 / ___|   _| |__   ___ _ __|  _ \    / \  / ___|
| |  | | | | '_ \ / _ \ '__| |_) |  / _ \| |  _
| |__| |_| | |_) |  __/ |  |  _ <  / ___ \ |_| |
 \____\__, |_.__/ \___|_|  |_| \_\/_/   \_\____|
      |___/   Local Cybersecurity Threat-Intel RAG
   100% on-prem · no data egress · cited & graph-aware
"""


def show(out):
    print("\n" + "─" * 64)
    print(out["answer"])
    print("─" * 64)
    if out.get("kg_node"):
        print(f"🔗 Knowledge graph: {out['kg_node']}  ({out['kg_fact_count']} facts)")
    if out.get("sources"):
        print("📚 Sources:")
        for s in out["sources"]:
            aid = f"  ATT&CK:{s['attack_ids']}" if s.get("attack_ids") else ""
            print(f"   [{s['score']}] {s['doc']} · {s['source']}{aid}")
    print(f"⏱  {out['latency_s']}s · 🔒 local-only · 💲 $0")


def main():
    if len(sys.argv) > 1:
        show(answer(" ".join(sys.argv[1:])))
        return
    print(BANNER)
    print("Ask a cybersecurity question. Commands: /source <list>, /norag, /quit\n")
    source_filter = None
    use_rag = True
    while True:
        try:
            q = input("cyber-rag> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("/quit", "/exit", "/q"):
            break
        if q.startswith("/source"):
            parts = q.split(None, 1)
            source_filter = [s.strip() for s in parts[1].split(",")] if len(parts) > 1 else None
            print(f"  source filter = {source_filter}")
            continue
        if q == "/norag":
            use_rag = not use_rag
            print(f"  RAG {'ON' if use_rag else 'OFF (raw local model)'}")
            continue
        show(answer(q, use_rag=use_rag, source_filter=source_filter))


if __name__ == "__main__":
    main()
