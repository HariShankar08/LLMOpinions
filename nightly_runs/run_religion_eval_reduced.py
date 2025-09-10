#!/usr/bin/env python3
"""
Reduced nightly automation for religion-focused representativeness evaluation.

This is a streamlined version that maintains diversity while reducing computational load.
Focuses on key representative locales and essential variants.
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


EVAL_SCRIPT = os.path.abspath(os.path.join(os.getcwd(), "evaluate_gemini_logprobs.py"))
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
    """Create timestamped output directory."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, timestamp)
    os.makedirs(path, exist_ok=True)
    return path


def run_eval(locale: Locale, variant: Variant, project_id: str, location: str, model_id: str, stdout_path: str) -> Tuple[str, str]:
    """Run the evaluation script with given parameters."""
    cmd = [
        sys.executable,
        EVAL_SCRIPT,
        "--project_id", project_id,
        "--location", location,
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
    
    # Expected output files
    csv_file = os.path.join(EVAL_LOGS_DIR, f"{locale.country}-{locale.language}.csv")
    avg_file = os.path.join(EVAL_LOGS_DIR, f"{locale.country}-{locale.language}_averages.csv")
    
    return csv_file, avg_file


def copy_variant_outputs(csv_src: str, avg_src: str, out_dir: str, locale: Locale, variant_name: str) -> Tuple[str, str]:
    """Copy CSV and averages files to timestamped directory with variant names."""
    base_name = f"{locale.country}-{locale.language}_{variant_name}"
    
    csv_dst = os.path.join(out_dir, f"{base_name}.csv")
    avg_dst = os.path.join(out_dir, f"{base_name}_averages.csv")
    
    if os.path.exists(csv_src):
        shutil.copy2(csv_src, csv_dst)
    if os.path.exists(avg_src):
        shutil.copy2(avg_src, avg_dst)
    
    return csv_dst, avg_dst


# Religion question identification
RELIGION_KEYS_CONTAINS = (
    "QCURREL",
    "QCHREL", 
    "QATTEND",
    "CHURCHEDU",
    "QGOD",
)


def is_religion_question(question: str) -> bool:
    """Check if a question relates to religion."""
    return any(key in question for key in RELIGION_KEYS_CONTAINS)


def read_csv_safely(path: str) -> List[Dict[str, str]]:
    """Read CSV with error handling."""
    if not os.path.exists(path):
        return []
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        print(f"Warning: failed to read {path}: {e}")
        return []


def write_religion_summary(out_dir: str, results: List[Tuple[Locale, Variant, str]]) -> None:
    """Generate religion-focused summary with deltas vs baseline."""
    summary_path = os.path.join(out_dir, "religion_summary.csv")
    
    # Collect all religion results
    all_results = {}  # (locale, question) -> {variant -> metrics}
    
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
    
    # Write summary
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
    parser = argparse.ArgumentParser(description="Reduced nightly religion evaluation automation")
    parser.add_argument("--project_id", help="Google Cloud project ID")
    parser.add_argument("--location", default="us-central1", help="Vertex AI location")
    parser.add_argument("--model_id", default="gemini-2.5-flash", help="Model to evaluate")
    parser.add_argument("--base_output", default="nightly_runs/output", help="Base output directory")
    parser.add_argument("--sleep_between_runs", default="5.0", help="Sleep between runs (seconds)")
    parser.add_argument("--resume_output_dir", default=None, help="Existing timestamped output dir to resume into (skips completed)")
    
    args = parser.parse_args()
    
    if not args.project_id:
        project_id = os.environ.get("PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project_id:
            args.project_id = project_id
    
    if not args.project_id:
        print("Error: --project_id not provided and neither PROJECT_ID nor GOOGLE_CLOUD_PROJECT are set.", file=sys.stderr)
        return 2
    
    os.makedirs(args.base_output, exist_ok=True)
    if args.resume_output_dir:
        out_dir = args.resume_output_dir
        if not os.path.isdir(out_dir):
            print(f"Error: --resume_output_dir not found: {out_dir}", file=sys.stderr)
            return 2
        print(f"Resuming into existing output dir: {out_dir}")
    else:
        out_dir = make_output_dir(args.base_output)
    
    # REDUCED DIVERSE COVERAGE - strategically selected for maximum diversity
    locales: List[Locale] = [
        # EA - representative coverage across major countries and languages
        Locale(region="EA", country="hk", language="en"),     # Already complete
        Locale(region="EA", country="hk", language="zh"),     # Already complete  
        Locale(region="EA", country="jp", language="ja"),     # Major East Asian country, native language
        Locale(region="EA", country="ko", language="ko"),     # Another major EA country, native language
        Locale(region="EA", country="vi", language="vi"),     # Southeast/East Asia border, native language
        
        # SEA - diverse religious and linguistic landscape
        Locale(region="SEA", country="id", language="id"),    # Largest Muslim country, native language
        Locale(region="SEA", country="sg", language="en"),    # Multi-religious hub, English
        Locale(region="SEA", country="th", language="th"),    # Buddhist majority, native language
        Locale(region="SEA", country="ms", language="ma"),    # Muslim majority, Malay language
        
        # IND - critical for religious diversity
        Locale(region="IND", country="ind", language="en"),   # English for comparison
        Locale(region="IND", country="ind", language="hi"),   # Native Hindi
    ]
    
    # REDUCED VARIANTS - keep most informative ones
    variants: List[Variant] = [
        Variant(name="baseline", temperature=0.0, top_p=None, repeats=1, cot=False),
        Variant(name="top_p0.9_rep3", temperature=0.0, top_p=0.9, repeats=3, cot=False),  # Best performing variant
    ]
    
    results_for_summary: List[Tuple[Locale, Variant, str]] = []
    
    print(f"REDUCED RUN PLAN:")
    print(f"- {len(locales)} locales (reduced from 18)")
    print(f"- {len(variants)} variants (reduced from 4)")  
    print(f"- Total: {len(locales) * len(variants)} runs (reduced from 72)")
    print(f"- Estimated time: ~{len(locales) * len(variants) * 15} minutes")
    print()
    
    # Orchestrate runs
    for loc in locales:
        for var in variants:
            variant_log = os.path.join(out_dir, f"{loc.country}-{loc.language}_{var.name}.log")
            
            # Skip if already completed
            if loc.country == "hk" and (loc.language == "en" or loc.language == "zh"):
                print(f"Skipping {loc.country}-{loc.language} / {var.name} (already completed)")
                continue
                
            # If resuming, skip already completed outputs
            completed_csv = os.path.join(out_dir, f"{loc.country}-{loc.language}_{var.name}.csv")
            if args.resume_output_dir and os.path.exists(completed_csv) and os.path.getsize(completed_csv) > 0:
                print(f"Skipping {loc.country}-{loc.language} / {var.name} (already present in resume dir)")
                results_for_summary.append((loc, var, completed_csv))
                continue

            print(f"Running {loc.country}-{loc.language} / {var.name}…")
            csv_src, avg_src = run_eval(
                locale=loc,
                variant=var,
                project_id=args.project_id,
                location=args.location,
                model_id=args.model_id,
                stdout_path=variant_log,
            )
            # Copy to timestamped folder
            csv_dst, _ = copy_variant_outputs(csv_src, avg_src, out_dir, loc, var.name)
            results_for_summary.append((loc, var, csv_dst))
            # Gentle sleep between runs to reduce 429s
            time.sleep(max(0.0, float(args.sleep_between_runs)))
    
    # Build religion summaries and deltas vs baseline
    write_religion_summary(out_dir, results_for_summary)
    
    # Write a README explaining the reduced scope
    with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write("REDUCED RELIGION EVALUATION RESULTS\n")
        f.write("===================================\n\n")
        f.write("This folder contains results from a strategically reduced evaluation\n")
        f.write("designed to maintain diversity while reducing computational load.\n\n")
        f.write("COVERAGE:\n")
        f.write(f"- {len(locales)} locales across 3 regions (reduced from 18)\n")
        f.write(f"- {len(variants)} variants (reduced from 4)\n")
        f.write(f"- Focus on representative countries, languages, and religious contexts\n\n")
        f.write("SELECTION CRITERIA:\n")
        f.write("- Geographic diversity: EA, SEA, IND regions\n")
        f.write("- Religious diversity: Christian, Muslim, Buddhist, Hindu contexts\n") 
        f.write("- Linguistic diversity: English + 6 native languages\n")
        f.write("- Variant diversity: deterministic baseline + stochastic sampling\n\n")
        f.write("FILES:\n")
        f.write("- *_baseline.csv: deterministic temperature=0 results\n")
        f.write("- *_top_p0.9_rep3.csv: stochastic top_p=0.9, 3 repeats\n")
        f.write("- religion_summary.csv: religion-focused metrics and deltas\n")
    
    print(f"All done. Outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
