"""Persist benchmark results as JSON and a human-readable text summary."""

import json
import os
import time
from pathlib import Path

_REPORTS_DIR = Path("reports")
_VERSION     = "1.1.1"


def save_report(report: dict) -> Path:
    _REPORTS_DIR.mkdir(exist_ok=True)
    ts    = time.strftime("%Y-%m-%d_%H-%M-%S")
    base  = _REPORTS_DIR / f"chronosbenchx_{ts}"
    json_path = base.with_suffix(".json")
    txt_path  = base.with_suffix(".txt")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write(_format(report))

    print(f"Report saved → {json_path}")
    return json_path


def _format(report: dict) -> str:
    meta    = report.get("meta", {})
    scores  = report.get("scores", {})
    results = report.get("results", {})

    lines = [
        f"ChronosBench X v{_VERSION}",
        f"Platform : {meta.get('platform', 'unknown')}",
        f"Timestamp: {meta.get('timestamp', 'unknown')}",
        "",
        "Scores",
        "------",
    ]

    component_scores = scores.get("scores", {})
    for name, value in component_scores.items():
        lines.append(f"  {name:<12} {value:>5} / 2000")

    lines += [
        "",
        f"  {'composite':<12} {scores.get('composite', 0):>5} / 2000",
        "",
        "Raw Results",
        "-----------",
    ]

    for section, data in results.items():
        lines.append(f"\n[{section}]")
        if isinstance(data, dict):
            for k, v in data.items():
                lines.append(f"  {k}: {v}")
        else:
            lines.append(f"  {data}")

    return "\n".join(lines) + "\n"
