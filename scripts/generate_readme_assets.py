"""Render CyberRAG's README benchmark graphic from the checked-in pilot result."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import xml.etree.ElementTree as ET
from html import escape
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "eval" / "results_2026-06-22_1901.json"
OUTPUT = ROOT / "docs" / "assets" / "cyberrag-pilot-coverage.svg"


def _load_result() -> dict[str, Any]:
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    summary = data.get("summary")
    rows = data.get("rows")
    if not isinstance(summary, dict) or not isinstance(rows, list) or not rows:
        raise ValueError("pilot result is missing its summary or paired rows")

    required = ("keyword_coverage", "latency_s")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError(f"invalid pilot row {index}")
        for arm in ("local_only", "cyberrag"):
            record = row.get(arm)
            if not isinstance(record, dict):
                raise ValueError(f"row {index} is missing {arm}")
            for metric in required:
                value = record.get(metric)
                if not isinstance(value, (int, float)):
                    raise ValueError(f"row {index} has no numeric {arm}.{metric}")
            coverage = float(record["keyword_coverage"])
            if not 0.0 <= coverage <= 1.0:
                raise ValueError("keyword coverage must remain within [0, 1]")
        hit = row["cyberrag"].get("context_hit")
        if hit not in (0, 1):
            raise ValueError(f"row {index} has no binary context_hit")

    if summary.get("n") != len(rows):
        raise ValueError("pilot summary count does not match its rows")
    for arm in ("local_only", "cyberrag"):
        recorded = summary.get(arm)
        if not isinstance(recorded, dict):
            raise ValueError(f"pilot summary is missing {arm}")
        for metric in required:
            recomputed = round(statistics.fmean(row[arm][metric] for row in rows), 3)
            if recorded.get(metric) != recomputed:
                raise ValueError(
                    f"pilot summary {arm}.{metric} does not match its rows"
                )
    context_hit_rate = round(
        statistics.fmean(row["cyberrag"]["context_hit"] for row in rows), 3
    )
    if summary["cyberrag"].get("context_hit_rate") != context_hit_rate:
        raise ValueError("pilot summary context_hit_rate does not match its rows")
    return data


def _svg(data: dict[str, Any]) -> str:
    summary = data["summary"]
    rows = data["rows"]
    local_coverage = float(summary["local_only"]["keyword_coverage"])
    rag_coverage = float(summary["cyberrag"]["keyword_coverage"])
    local_latency = float(summary["local_only"]["latency_s"])
    rag_latency = float(summary["cyberrag"]["latency_s"])
    coverage_delta_pp = (rag_coverage - local_coverage) * 100
    latency_delta = rag_latency - local_latency
    context_hits = sum(int(row["cyberrag"]["context_hit"]) for row in rows)
    improved = sum(
        row["cyberrag"]["keyword_coverage"] > row["local_only"]["keyword_coverage"]
        for row in rows
    )
    tied = sum(
        row["cyberrag"]["keyword_coverage"] == row["local_only"]["keyword_coverage"]
        for row in rows
    )
    lower = len(rows) - improved - tied

    width = 1200
    height = 790
    plot_left = 595
    plot_right = 1124
    plot_width = plot_right - plot_left
    row_start = 260
    row_step = 30

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">CyberRAG paired pilot keyword coverage</title>',
        (
            f'<desc id="desc">A paired dot plot for {len(rows)} cybersecurity questions. '
            f"Mean keyword coverage rose from {local_coverage:.3f} without retrieval to "
            f"{rag_coverage:.3f} with CyberRAG; context retrieval hit {context_hits} of "
            f"{len(rows)} expected documents and mean latency increased by {latency_delta:.3f} seconds.</desc>"
        ),
        '<rect width="1200" height="790" rx="28" fill="#071923"/>',
        '<path d="M0 0H1200V16H0Z" fill="#35d2cf"/>',
        '<line x1="472" y1="78" x2="472" y2="704" stroke="#1e3c47"/>',
        '<g font-family="Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif">',
        '<text x="48" y="64" fill="#89f2ed" font-size="15" font-weight="700" letter-spacing="2.2">CYBERRAG / PAIRED LOCAL PILOT</text>',
        '<text x="48" y="112" fill="#f2fbfc" font-size="36" font-weight="760">Did retrieval help?</text>',
        '<text x="48" y="143" fill="#9fc1c8" font-size="16">Question-level evidence, including ties and regressions.</text>',
        f'<text x="48" y="228" fill="#35d2cf" font-size="55" font-weight="780">+{coverage_delta_pp:.1f} pp</text>',
        '<text x="48" y="255" fill="#b7d0d5" font-size="15">mean keyword coverage</text>',
        f'<text x="48" y="316" fill="#f2fbfc" font-size="30" font-weight="730">{context_hits}/{len(rows)}</text>',
        '<text x="48" y="340" fill="#b7d0d5" font-size="15">expected-document hits</text>',
        f'<text x="48" y="393" fill="#f4aa5b" font-size="30" font-weight="730">+{latency_delta:.2f}s</text>',
        '<text x="48" y="417" fill="#b7d0d5" font-size="15">mean latency trade-off</text>',
        '<rect x="48" y="462" width="400" height="88" rx="12" fill="#0d2834" stroke="#1f4653"/>',
        f'<text x="72" y="499" fill="#35d2cf" font-size="25" font-weight="740">{improved} better</text>',
        f'<text x="213" y="499" fill="#d4e2e5" font-size="25" font-weight="740">{tied} tied</text>',
        f'<text x="328" y="499" fill="#f47e72" font-size="25" font-weight="740">{lower} lower</text>',
        '<text x="72" y="528" fill="#8eafb6" font-size="13">per-question deterministic keyword score</text>',
        '<text x="497" y="143" fill="#f2fbfc" font-size="25" font-weight="720">Coverage by question</text>',
        '<circle cx="767" cy="137" r="6" fill="#071923" stroke="#f4aa5b" stroke-width="2.5"/>',
        '<text x="780" y="142" fill="#b7d0d5" font-size="13">local only</text>',
        '<circle cx="895" cy="137" r="6" fill="#35d2cf"/>',
        '<text x="908" y="142" fill="#b7d0d5" font-size="13">CyberRAG</text>',
        '<text x="1153" y="180" fill="#789aa2" font-size="11" font-weight="700" text-anchor="middle" letter-spacing="0.8">HIT</text>',
    ]

    for tick in range(0, 6):
        value = tick / 5
        x = plot_left + value * plot_width
        lines.extend(
            [
                f'<line x1="{x:.1f}" y1="190" x2="{x:.1f}" y2="{row_start + (len(rows) - 1) * row_step + 15}" stroke="#1e3c47"/>',
                f'<text x="{x:.1f}" y="180" fill="#789aa2" font-size="12" text-anchor="middle">{value:.1f}</text>',
            ]
        )

    for index, row in enumerate(rows):
        y = row_start + index * row_step
        local = float(row["local_only"]["keyword_coverage"])
        rag = float(row["cyberrag"]["keyword_coverage"])
        local_x = plot_left + local * plot_width
        rag_x = plot_left + rag * plot_width
        direction = (
            "#35d2cf" if rag > local else "#f47e72" if rag < local else "#6e8d94"
        )
        lines.extend(
            [
                f'<text x="568" y="{y + 5}" fill="#c9dcdf" font-size="13" text-anchor="end">{escape(row["id"])}</text>',
                f'<line x1="{local_x:.1f}" y1="{y}" x2="{rag_x:.1f}" y2="{y}" stroke="{direction}" stroke-width="3" opacity="0.8"/>',
                f'<circle cx="{local_x:.1f}" cy="{y}" r="6" fill="#071923" stroke="#f4aa5b" stroke-width="2.5"/>',
                f'<circle cx="{rag_x:.1f}" cy="{y}" r="6.3" fill="#35d2cf"/>',
            ]
        )
        if row["cyberrag"]["context_hit"] == 0:
            lines.append(
                f'<text x="1150" y="{y + 5}" fill="#f47e72" font-size="15" font-weight="700">×</text>'
            )
        else:
            lines.append(
                f'<circle cx="1153" cy="{y}" r="4" fill="#35d2cf" opacity="0.75"/>'
            )

    footer_y = row_start + len(rows) * row_step + 28
    lines.extend(
        [
            f'<line x1="497" y1="{footer_y - 22}" x2="1155" y2="{footer_y - 22}" stroke="#1e3c47"/>',
            f'<text x="497" y="{footer_y}" fill="#789aa2" font-size="13">● expected document retrieved in top 5   × miss</text>',
            '<text x="48" y="752" fill="#789aa2" font-size="13">15-question pilot • deterministic substring coverage • model-judge score excluded here because the legacy artifact did not record its backend</text>',
            '<text x="1152" y="752" fill="#789aa2" font-size="13" text-anchor="end">Not production validation</text>',
            "</g>",
            "</svg>",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_svg(content: str) -> None:
    try:
        ET.fromstring(content)
    except ET.ParseError as exc:
        raise ValueError(f"generated invalid SVG: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the checked-in SVG differs from the pilot-derived output",
    )
    args = parser.parse_args()

    content = _svg(_load_result())
    _validate_svg(content)
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != content:
            print(f"stale: {OUTPUT.relative_to(ROOT)}", file=sys.stderr)
            return 1
        print("CyberRAG README benchmark asset is current and valid XML.")
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(content, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    print(f"wrote {OUTPUT.relative_to(ROOT)} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
