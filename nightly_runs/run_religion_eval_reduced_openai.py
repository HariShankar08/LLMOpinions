#!/usr/bin/env python3
"""
Reduced nightly automation for religion-focused representativeness evaluation using OpenAI.

Mirrors nightly_runs/run_religion_eval_reduced.py but calls evaluate_openai_logprobs.py.
Produces identical outputs in nightly_runs/output/<timestamp>.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


EVAL_SCRIPT = os.path.abspath(os.path.join(os.getcwd(), "evaluate_openai_logprobs.py"))
EVAL_LOGS_DIR = os.path.abspath(os.path.join(os.getcwd(), "evaluation_logs"))


@dataclass
class Variant:
    name: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repeats: int = 1
    cot: bool = False
    per_request_delay: float = 2.5
    retry_max_attempts: int = 10
    retry_max_backoff: int = 60
    logprobs: int = 1


@dataclass
class Locale:
    region: str
    country: str
    language: str


def make_output_dir(base: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, timestamp)
    os.makedirs(path, exist_ok=True)
    return path


def run_eval(locale: Locale, variant: Variant, model_id: str, stdout_path: str) -> Tuple[str, str]:
    cmd = [
        sys.executable,
        EVAL_SCRIPT,
        "--region", locale.region,
        "--country", locale.country,
        "--language", locale.language,
        "--model_id", model_id,
        "--logprobs", str(variant.logprobs),
        "--repeats", str(variant.repeats),
        "--per_request_delay", str(variant.per_request_delay),
        "--retry_max_attempts", str(variant.retry_max_attempts),
        "--retry_max_backoff", str(variant.retry_max_backoff),
    ]
    if variant.temperature is not None:
        cmd.extend(["--temperature", str(variant.temperature)])
    if variant.top_p is not None:
        cmd.extend(["--top_p", str(variant.top_p)])
    if variant.cot:
        cmd.append("--cot")

    with open(stdout_path, "w", encoding="utf-8") as f:
        f.write(f"Command: \n{' '.join(cmd)}\n\n")
        f.flush()
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        if result.returncode != 0:
            print(f"Warning: {locale.country}-{locale.language} / {variant.name} exited with code {result.returncode}")

    csv_file = os.path.join(EVAL_LOGS_DIR, f"{locale.country}-{locale.language}.csv")
    avg_file = os.path.join(EVAL_LOGS_DIR, f"{locale.country}-{locale.language}_averages.csv")
    return csv_file, avg_file


def copy_variant_outputs(csv_src: str, avg_src: str, out_dir: str, locale: Locale, variant_name: str) -> Tuple[str, str]:
    base_name = f"{locale.country}-{locale.language}_{variant_name}"
    csv_dst = os.path.join(out_dir, f"{base_name}.csv")
    avg_dst = os.path.join(out_dir, f"{base_name}_averages.csv")
    if os.path.exists(csv_src):
        shutil.copy2(csv_src, csv_dst)
    if os.path.exists(avg_src):
        shutil.copy2(avg_src, avg_dst)
    return csv_dst, avg_dst


RELIGION_KEYS_CONTAINS = (
    "QCURREL",
    "QCHREL",
    "QATTEND",
    "CHURCHEDU",
    "QGOD",
)


def is_religion_question(question: str) -> bool:
    return any(key in question for key in RELIGION_KEYS_CONTAINS)


def read_csv_safely(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        print(f"Warning: failed to read {path}: {e}")
        return []


def write_religion_summary(out_dir: str, results: List[Tuple[Locale, Variant, str]]) -> None:
    summary_path = os.path.join(out_dir, "religion_summary.csv")
    all_results: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}
    for locale, variant, csv_path in results:
        rows = read_csv_safely(csv_path)
        locale_key = f"{locale.country}-{locale.language}"
        for row in rows:
            question = row.get("question", "")
            if is_religion_question(question):
                key = (locale_key, question)
                if key not in all_results:
                    all_results[key] = {}
                all_results[key][variant.name] = {
                    "wd": float(row.get("wd", 0)),
                    "jsd": float(row.get("jsd", 0)),
                    "hell": float(row.get("hell", 0)),
                }

    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "locale", "question",
            "baseline_wd", "baseline_jsd", "baseline_hell",
            "top_p_wd", "top_p_jsd", "top_p_hell",
            "delta_wd", "delta_jsd", "delta_hell"
        ])
        for (locale_key, question), variants in all_results.items():
            baseline = variants.get("baseline", {"wd": 0, "jsd": 0, "hell": 0})
            top_p = variants.get("top_p0.9_rep3", {"wd": 0, "jsd": 0, "hell": 0})
            delta_wd = top_p["wd"] - baseline["wd"]
            delta_jsd = top_p["jsd"] - baseline["jsd"]
            delta_hell = top_p["hell"] - baseline["hell"]
            writer.writerow([
                locale_key, question,
                f"{baseline['wd']:.6f}", f"{baseline['jsd']:.6f}", f"{baseline['hell']:.6f}",
                f"{top_p['wd']:.6f}", f"{top_p['jsd']:.6f}", f"{top_p['hell']:.6f}",
                f"{delta_wd:.6f}", f"{delta_jsd:.6f}", f"{delta_hell:.6f}"
            ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Reduced nightly religion evaluation (OpenAI)")
    parser.add_argument("--model_id", default="gpt-4o-mini", help="OpenAI model to evaluate")
    parser.add_argument("--base_output", default="nightly_runs/output", help="Base output directory")
    parser.add_argument("--sleep_between_runs", default="5.0", help="Sleep between runs (seconds)")
    args = parser.parse_args()

    os.makedirs(args.base_output, exist_ok=True)
    out_dir = make_output_dir(args.base_output)

    locales: List[Locale] = [
        Locale(region="EA", country="hk", language="en"),
        Locale(region="EA", country="hk", language="zh"),
        Locale(region="EA", country="jp", language="ja"),
        Locale(region="EA", country="ko", language="ko"),
        Locale(region="EA", country="vi", language="vi"),

        Locale(region="SEA", country="id", language="id"),
        Locale(region="SEA", country="sg", language="en"),
        Locale(region="SEA", country="th", language="th"),
        Locale(region="SEA", country="ms", language="ma"),

        Locale(region="IND", country="ind", language="en"),
        Locale(region="IND", country="ind", language="hi"),
    ]

    variants: List[Variant] = [
        Variant(name="baseline", temperature=0.0, top_p=None, repeats=1, cot=False),
        Variant(name="top_p0.9_rep3", temperature=0.0, top_p=0.9, repeats=3, cot=False),
    ]

    results_for_summary: List[Tuple[Locale, Variant, str]] = []

    print(f"REDUCED RUN PLAN (OpenAI):")
    print(f"- {len(locales)} locales")
    print(f"- {len(variants)} variants")
    print(f"- Total: {len(locales) * len(variants)} runs")
    print(f"- Estimated time: ~{len(locales) * len(variants) * 15} minutes")
    print()

    for loc in locales:
        for var in variants:
            variant_log = os.path.join(out_dir, f"{loc.country}-{loc.language}_{var.name}.log")
            if loc.country == "hk" and (loc.language == "en" or loc.language == "zh"):
                print(f"Skipping {loc.country}-{loc.language} / {var.name} (already completed)")
                continue
            print(f"Running {loc.country}-{loc.language} / {var.name}…")
            csv_src, avg_src = run_eval(
                locale=loc,
                variant=var,
                model_id=args.model_id,
                stdout_path=variant_log,
            )
            csv_dst, _ = copy_variant_outputs(csv_src, avg_src, out_dir, loc, var.name)
            results_for_summary.append((loc, var, csv_dst))
            time.sleep(max(0.0, float(args.sleep_between_runs)))

    write_religion_summary(out_dir, results_for_summary)

    with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write("REDUCED RELIGION EVALUATION RESULTS (OpenAI)\n")
        f.write("===========================================\n\n")
        f.write("This folder contains results from a strategically reduced evaluation using OpenAI.\n\n")
        f.write("FILES:\n")
        f.write("- *_baseline.csv: deterministic temperature=0 results\n")
        f.write("- *_top_p0.9_rep3.csv: stochastic top_p=0.9, 3 repeats\n")
        f.write("- religion_summary.csv: religion-focused metrics and deltas\n")

    print(f"All done. Outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


