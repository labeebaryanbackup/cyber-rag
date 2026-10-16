#!/usr/bin/env python3
"""
Fetch authoritative cybersecurity corpora (one-time download, then ingest offline).

Sources (all free, license-friendly):
  - MITRE ATT&CK Enterprise   (STIX 2.1 JSON, github mitre-attack/attack-stix-data)
  - CWE                       (Common Weakness Enumeration, XML/CSV from MITRE)
  - CAPEC                     (Common Attack Pattern Enum, CSV from MITRE)
  - CISA KEV                  (Known Exploited Vulnerabilities, JSON)
  - Sigma rules               (detection rules, github SigmaHQ/sigma)

Writes normalized .md/.json docs into corpus/<source>/ for the ingest step.
This is the ONLY step that touches the network; the RAG runtime stays offline.

Run:  python ingest/fetch_authoritative.py
"""
import os
import json
import urllib.request
import zipfile
import io
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "corpus")

SOURCES = {
    "attack": "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json",
    "kev": "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
    "capec": "https://raw.githubusercontent.com/mitre/cti/master/capec/2.1/stix-capec.json",
}
SIGMA_ZIP = "https://github.com/SigmaHQ/sigma/archive/refs/heads/master.zip"


def get(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 CyberRAG"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def ingest_attack():
    """MITRE ATT&CK STIX -> one .md per technique."""
    print("[attack] downloading STIX...")
    data = json.loads(get(SOURCES["attack"]))
    out = os.path.join(CORPUS, "attack")
    n = 0
    for obj in data.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        ext = next((r for r in obj.get("external_references", [])
                    if r.get("source_name") == "mitre-attack"), {})
        tid = ext.get("external_id", "")
        if not tid:
            continue
        name = obj.get("name", "")
        desc = obj.get("description", "")
        tactics = ",".join(p.get("phase_name", "") for p in obj.get("kill_chain_phases", []))
        platforms = ",".join(obj.get("x_mitre_platforms", []))
        det = obj.get("x_mitre_detection", "")
        md = (f"# {tid}: {name}\n\n**Tactics:** {tactics}\n**Platforms:** {platforms}\n\n"
              f"## Description\n{desc}\n\n## Detection\n{det}\n")
        write(os.path.join(out, f"{tid}.md"), md)
        n += 1
    print(f"[attack] wrote {n} technique docs")


def ingest_kev():
    """CISA Known Exploited Vulnerabilities -> one .md per CVE."""
    print("[kev] downloading...")
    data = json.loads(get(SOURCES["kev"]))
    out = os.path.join(CORPUS, "kev")
    n = 0
    for v in data.get("vulnerabilities", []):
        cve = v.get("cveID", "")
        md = (f"# {cve}: {v.get('vulnerabilityName','')}\n\n"
              f"**Vendor:** {v.get('vendorProject','')}  **Product:** {v.get('product','')}\n"
              f"**Date added:** {v.get('dateAdded','')}  **Due:** {v.get('dueDate','')}\n"
              f"**Known ransomware use:** {v.get('knownRansomwareCampaignUse','')}\n\n"
              f"## Required action\n{v.get('requiredAction','')}\n\n"
              f"## Description\n{v.get('shortDescription','')}\n")
        write(os.path.join(out, f"{cve}.md"), md)
        n += 1
    print(f"[kev] wrote {n} CVE docs")


def ingest_capec():
    """CAPEC STIX -> one .md per attack pattern."""
    print("[capec] downloading...")
    try:
        data = json.loads(get(SOURCES["capec"]))
    except Exception as e:
        print(f"[capec] skip ({e})")
        return
    out = os.path.join(CORPUS, "capec")
    n = 0
    for obj in data.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        ext = next((r for r in obj.get("external_references", [])
                    if r.get("source_name") == "capec"), {})
        cid = ext.get("external_id", "")
        if not cid:
            continue
        md = (f"# {cid}: {obj.get('name','')}\n\n## Description\n{obj.get('description','')}\n")
        write(os.path.join(out, f"{cid.replace('/','-')}.md"), md)
        n += 1
    print(f"[capec] wrote {n} attack-pattern docs")


def ingest_sigma(limit=1500):
    """Sigma detection rules (YAML) -> .md, capped to keep corpus balanced."""
    print("[sigma] downloading repo zip (large)...")
    try:
        z = zipfile.ZipFile(io.BytesIO(get(SIGMA_ZIP, timeout=300)))
    except Exception as e:
        print(f"[sigma] skip ({e})")
        return
    out = os.path.join(CORPUS, "sigma")
    n = 0
    for name in z.namelist():
        if not re.search(r"/rules/.*\.yml$", name):
            continue
        if n >= limit:
            break
        try:
            y = z.read(name).decode("utf-8", errors="replace")
        except Exception:
            continue
        title = re.search(r"^title:\s*(.+)$", y, re.M)
        base = os.path.basename(name).replace(".yml", "")
        md = f"# Sigma rule: {title.group(1) if title else base}\n\n```yaml\n{y}\n```\n"
        write(os.path.join(out, f"{base}.md"), md)
        n += 1
    print(f"[sigma] wrote {n} detection-rule docs")


if __name__ == "__main__":
    ingest_attack()
    ingest_kev()
    ingest_capec()
    ingest_sigma()
    print("\n[done] authoritative corpus fetched into corpus/")
