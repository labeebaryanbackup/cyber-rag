import pytest

from eval.run_eval import keyword_coverage
from ingest.build_index import parse_extra_sources
from rag.hybrid import _query_ids, rrf_fuse, tokenize, tokenize_query


def test_tokenize_normalizes_security_identifiers():
    assert tokenize("Check CVE-2021-44228 and T1059.003") == [
        "check",
        "cve-2021-44228",
        "and",
        "t1059.003",
    ]


def test_query_tokenizer_removes_low_signal_stopwords():
    assert tokenize_query("What is the mitigation for T1059?") == [
        "mitigation",
        "t1059",
    ]


def test_query_identifier_extraction_is_case_insensitive():
    assert _query_ids("Compare cve-2021-44228, T1059.003 and capec-66") == {
        "CVE-2021-44228",
        "T1059.003",
        "CAPEC-66",
    }


def test_rrf_promotes_an_exact_identifier_match():
    unrelated = {"id": "a", "doc": "General Java logging guidance", "meta": {}}
    exact = {
        "id": "b",
        "doc": "CVE-2021-44228 affects Log4j",
        "meta": {"cve_ids": "CVE-2021-44228"},
    }

    fused = rrf_fuse([unrelated, exact], [], query="Explain CVE-2021-44228")

    assert fused[0]["id"] == "b"


def test_keyword_coverage_is_deterministic():
    assert keyword_coverage("APT29 uses PowerShell", ["APT29", "PowerShell", "T1059"]) == 0.667
    assert keyword_coverage("anything", []) is None


def test_extra_sources_are_explicit_and_portable():
    assert parse_extra_sources(["notes=./notes/*.md"]) == [
        ("./notes/*.md", "notes")
    ]

    with pytest.raises(ValueError):
        parse_extra_sources(["missing-separator"])
