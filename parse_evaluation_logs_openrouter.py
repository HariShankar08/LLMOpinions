#!/usr/bin/env python3
"""
Parse OpenRouter evaluation logs and produce a CSV with average representativeness
per country-language key (e.g., hk-en) across all runs.
"""

import os
import re
import glob
import csv
from collections import defaultdict
from datetime import datetime

LOGS_DIR = "evaluation_logs_openrouter"


def parse_filename(path: str):
    base = os.path.basename(path)
    if not base.endswith('.log'):
        return None
    name = base[:-4]
    parts = name.split('_')
    if len(parts) < 4:
        return None
    region = parts[0]
    country = parts[1]
    language = parts[2]
    model_short = parts[3]
    return region, country, language, model_short


def extract_avg_repr(content: str):
    m = re.search(r"Average Representativeness:\s*([0-9.]+)", content)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def main():
    if not os.path.isdir(LOGS_DIR):
        print(f"Logs dir not found: {LOGS_DIR}")
        return

    entries = []
    for path in glob.glob(os.path.join(LOGS_DIR, "*.log")):
        meta = parse_filename(path)
        if not meta:
            continue
        region, country, language, model_short = meta
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue
        avg = extract_avg_repr(content)
        key = f"{country}-{language}"
        # Include entry even if avg is None (failed runs), to keep EA keys visible
        entries.append((key, avg, region, model_short, os.path.basename(path)))

    if not entries:
        print("No averages found in logs.")
        return

    # Aggregate by key
    agg = defaultdict(lambda: {"sum": 0.0, "n": 0, "regions": set(), "models": set(), "logs": [], "failed": 0})
    for key, avg, region, model_short, logf in entries:
        a = agg[key]
        if isinstance(avg, (int, float)):
            a["sum"] += float(avg)
            a["n"] += 1
        else:
            a["failed"] += 1
        a["regions"].add(region)
        a["models"].add(model_short)
        a["logs"].append(logf)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(LOGS_DIR, f"averages_openrouter_{ts}.csv")

    # Write CSV with minimal required columns plus context
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "avg_repr", "num_success", "num_failed", "regions", "models", "log_files"])
        for key in sorted(agg.keys()):
            a = agg[key]
            avg_repr = (a["sum"] / a["n"]) if a["n"] else ""
            w.writerow([
                key,
                f"{avg_repr}",
                a["n"],
                a["failed"],
                ",".join(sorted(a["regions"])),
                ",".join(sorted(a["models"])),
                ",".join(sorted(a["logs"]))
            ])

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()


