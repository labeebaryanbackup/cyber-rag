# CyberRAG — Executive Brief

**Status:** independent prototype built in response to a CyberSecurity Malaysia challenge brief. It is not an official deployment or endorsed product.

## The problem

Small local language models preserve privacy and avoid per-query cloud cost, but they often miss specialist threat-intelligence facts or invent CVE and ATT&CK identifiers. Sending internal incident material to a hosted model can also conflict with an organisation's data-handling requirements.

## The prototype

CyberRAG keeps the normal query path on one machine and grounds answers with:

- local vector retrieval;
- BM25 exact-term retrieval;
- reciprocal-rank fusion and exact identifier boosts;
- a MITRE ATT&CK relationship graph;
- a local Ollama generator that must cite retrieved evidence.

Public sources can be rebuilt from MITRE ATT&CK, CISA KEV, CAPEC, and Sigma. Local PDF, DOCX, Markdown, text, and HTML documents can be added without committing them to the repository.

## Evidence

A committed 15-question pilot snapshot records:

| Metric | Local model only | Local model + CyberRAG |
|---|---:|---:|
| Keyword coverage | 0.627 | **0.843** |
| Context hit rate | — | **0.933** |
| Mean latency | 11.95 s | 14.65 s |

The result supports the narrower claim that retrieval improved this local model on this fixed question set. It does not establish cloud parity, production reliability, or broad cybersecurity accuracy.

## Appropriate next validation

1. Expand the benchmark and separate development from held-out questions.
2. Record evaluator name, version, prompt, and backend for every result.
3. Test permission boundaries and prompt-injection behaviour with synthetic documents.
4. Run a network-egress check during query-only operation.
5. Evaluate with domain reviewers before operational use.
