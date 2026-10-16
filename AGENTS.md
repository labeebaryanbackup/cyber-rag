# CyberRAG repository guide

## Architecture

- `ingest/` fetches or imports documents and builds the local Chroma index.
- `rag/` contains vector/BM25 hybrid retrieval, reciprocal-rank fusion, Ollama reranking/generation, and the NetworkX ATT&CK knowledge graph.
- `demo.py` is the user-facing CLI. `eval/` contains the fixed questions, runner, and dated benchmark results. `docs/ARCHITECTURE.md` is the detailed design reference.
- Query-time behavior should remain local by default: Ollama embeddings/generation, on-disk Chroma, BM25, and the knowledge graph.

## Working rules

- Preserve citation grounding and the anti-hallucination rule for CVE and ATT&CK identifiers.
- Keep vector, lexical, and graph retrieval concerns separate; when retrieval changes, verify exact-ID queries as well as semantic questions.
- Do not commit private incident reports or other ingested internal intelligence. Treat `corpus/` and `data/` as potentially sensitive or large, and inspect Git status before rebuilding them.
- `ingest/fetch_authoritative.py` downloads external sources. `eval/run_eval.py --cloud` sends prompts to a cloud model. Run either only when the task explicitly allows network use and, for cloud evaluation, data egress/cost.
- Do not rebuild indexes merely for a documentation or presentation edit.

## Commands and verification

Prefer the checked-in local environment interpreter when present:

```powershell
$python = '.venv/Scripts/python.exe'
& $python demo.py 'Which techniques does APT29 use and how can they be detected?'
& $python eval/run_eval.py --limit 1
```

For a full local rebuild, only after confirming that generated data changes are intended:

```powershell
& $python ingest/build_index.py
& $python rag/hybrid.py build
& $python rag/kg.py build
```

- Use `python -m pytest` for any added unit tests; the current repository has an evaluation harness rather than a dedicated test tree.
- For retrieval changes, compare against the committed benchmark in `eval/results_2026-06-22_1901.json` and explain any metric or latency regression.
- Never describe a run as zero-egress if a cloud flag, remote model, or network fetch was used.
