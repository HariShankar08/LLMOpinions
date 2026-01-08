#!/usr/bin/env python3
"""
Serial OpenAI logprobs runner across all locales.

Runs evaluate_openai_logprobs.py one (country,language) at a time to avoid rate limits.
Copies each CSV into a separate timestamped output folder.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List


EVAL_SCRIPT = os.path.abspath(os.path.join(os.getcwd(), "evaluate_openai_logprobs.py"))
EVAL_LOGS_DIR = os.path.abspath(os.path.join(os.getcwd(), "evaluation_logs"))


@dataclass
class Locale:
    region: str
    country: str
    language: str


def build_all_locales() -> List[Locale]:
    locales: List[Locale] = []
    # EA
    locales += [
        Locale("EA", "hk", "en"), Locale("EA", "hk", "zh"),
        Locale("EA", "jp", "en"), Locale("EA", "jp", "ja"),
        Locale("EA", "ko", "en"), Locale("EA", "ko", "ko"),
        Locale("EA", "tw", "en"), Locale("EA", "tw", "zh"),
        Locale("EA", "vi", "en"), Locale("EA", "vi", "vi"),
    ]
    # SEA
    locales += [
        Locale("SEA", "ca", "en"), Locale("SEA", "ca", "km"),
        Locale("SEA", "id", "en"), Locale("SEA", "id", "id"),
        Locale("SEA", "ms", "en"), Locale("SEA", "ms", "ma"), Locale("SEA", "ms", "zh"),
        Locale("SEA", "sg", "en"), Locale("SEA", "sg", "ma"), Locale("SEA", "sg", "ta"), Locale("SEA", "sg", "zh"),
        Locale("SEA", "sl", "en"), Locale("SEA", "sl", "si"), Locale("SEA", "sl", "ta"),
        Locale("SEA", "th", "en"), Locale("SEA", "th", "th"),
    ]
    # IND
    locales += [Locale("IND", "ind", "en"), Locale("IND", "ind", "hi")]
    return locales


def make_output_dir(base_dir: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, ts, "openai")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def run_one(locale: Locale, model_id: str, logprobs: int, repeats: int, temperature: float, top_p: float | None,
            per_request_delay: float, retry_max_attempts: int, retry_max_backoff: float,
            openai_key: str, steer_file: str | None, steer_text: str | None, disable_auto_persona: bool,
            stdout_path: str) -> str:
    """Run one locale. Returns path to per-locale CSV written by the evaluator."""
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = openai_key

    cmd = [
        sys.executable,
        EVAL_SCRIPT,
        "--region", locale.region,
        "--country", locale.country,
        "--language", locale.language,
        "--model_id", model_id,
        "--logprobs", str(max(1, min(19, logprobs))),
        "--repeats", str(max(1, repeats)),
        "--temperature", str(temperature),
        "--per_request_delay", str(per_request_delay),
        "--retry_max_attempts", str(retry_max_attempts),
        "--retry_max_backoff", str(retry_max_backoff),
    ]
    if top_p is not None:
        cmd += ["--top_p", str(top_p)]
    if steer_file:
        cmd += ["--steer_file", steer_file]
    elif steer_text:
        cmd += ["--steer_text", steer_text]
    if disable_auto_persona:
        cmd += ["--disable_auto_persona"]

    with open(stdout_path, "w", encoding="utf-8") as logf:
        logf.write("Command:\n" + " ".join(cmd) + "\n\n")
        logf.flush()
        subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=False, env=env)

    return os.path.join(EVAL_LOGS_DIR, f"{locale.country}-{locale.language}.csv")


def copy_into_folder(csv_src: str, out_dir: str, locale: Locale) -> str:
    base = f"{locale.region}_{locale.country}-{locale.language}.csv"
    dst = os.path.join(out_dir, base)
    if os.path.exists(csv_src):
        shutil.copy2(csv_src, dst)
    return dst


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serial OpenAI logprobs across locales")
    p.add_argument("--openai_key", type=str, required=True)
    p.add_argument("--model_id", type=str, default="gpt-4o-mini")
    p.add_argument("--base_output", type=str, default=os.path.join(os.getcwd(), "nightly_runs", "output_serial"))
    p.add_argument("--sleep_between_runs", type=float, default=5.0)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--logprobs", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--per_request_delay", type=float, default=1.5)
    p.add_argument("--retry_max_attempts", type=int, default=6)
    p.add_argument("--retry_max_backoff", type=float, default=30.0)
    # Steering
    p.add_argument("--steer_file", type=str, default=None, help="Path to steering text file")
    p.add_argument("--steer_text", type=str, default=None, help="Inline steering text (ignored if --steer_file provided)")
    p.add_argument("--disable_auto_persona", action="store_true", help="Disable automatic country persona")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.base_output, exist_ok=True)
    out_dir = make_output_dir(args.base_output)

    locales = build_all_locales()
    print(f"Will run {len(locales)} locales serially (OpenAI). Output -> {out_dir}")

    for loc in locales:
        tag = f"{loc.region}_{loc.country}-{loc.language}"
        stdout_path = os.path.join(out_dir, f"{tag}.log")
        print(f"Running OpenAI: {tag}…")
        csv_src = run_one(
            locale=loc,
            model_id=args.model_id,
            logprobs=args.logprobs,
            repeats=args.repeats,
            temperature=args.temperature,
            top_p=(None if args.top_p is None else float(args.top_p)),
            per_request_delay=args.per_request_delay,
            retry_max_attempts=args.retry_max_attempts,
            retry_max_backoff=args.retry_max_backoff,
            openai_key=args.openai_key,
            steer_file=args.steer_file,
            steer_text=args.steer_text,
            disable_auto_persona=bool(args.disable_auto_persona),
            stdout_path=stdout_path,
        )
        copied = copy_into_folder(csv_src, out_dir, loc)
        print(f"Copied -> {copied}")
        time.sleep(max(0.0, float(args.sleep_between_runs)))

    print("All OpenAI runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


