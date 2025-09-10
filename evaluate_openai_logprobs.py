"""
Evaluate OpenAI models with token-level logprobs, mirroring the Gemini evaluator.

This script provides two modes:
- Full evaluation flow (Translate region/country/language) producing CSVs identical to Gemini version
- Single-prompt/classification JSONL mode for ad-hoc testing

Environment variables:
  - OPENAI_API_KEY must be set

Requires:
  - python -m pip install -U openai pandas scipy tqdm
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import time
import random

import pandas as pd
from scipy.stats import wasserstein_distance
from tqdm import tqdm


def _import_openai_or_exit():
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        message = (
            "Missing dependency: openai.\n\n"
            "Install with:\n  python -m pip install -U openai\n\n"
            f"Original error: {exc}"
        )
        print(message, file=sys.stderr)
        sys.exit(2)
    return OpenAI


def _require_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("Error: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(2)
    return key


def read_prompts(input_path: str) -> List[str]:
    path_lower = input_path.lower()
    if path_lower.endswith(".txt"):
        with open(input_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    if path_lower.endswith(".csv"):
        prompts: List[str] = []
        with open(input_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "prompt" not in (reader.fieldnames or []):
                print("Error: CSV must include a 'prompt' column.", file=sys.stderr)
                sys.exit(2)
            for row in reader:
                value = (row.get("prompt") or "").strip()
                if value:
                    prompts.append(value)
        return prompts
    print("Error: input must be a .txt or .csv file.", file=sys.stderr)
    sys.exit(2)


def normalize_choices(choices_arg: Optional[str], choices_file: Optional[str]) -> Optional[List[str]]:
    choices: Optional[List[str]] = None
    if choices_file:
        with open(choices_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
                print("Error: choices_file must contain a JSON array of strings.", file=sys.stderr)
                sys.exit(2)
            choices = [x for x in data if x.strip()]
    elif choices_arg:
        parts = [p.strip() for p in choices_arg.split(",")]
        choices = [p for p in parts if p]
    return choices if choices else None


@dataclass
class TokenLogprob:
    token: str
    logprob: float
    alternatives: List[Tuple[str, float]]


def _extract_text_from_chat(response: Any) -> str:
    try:
        choice0 = response.choices[0]
        content = getattr(choice0, "message", None)
        if content and hasattr(content, "content"):
            return str(content.content or "").strip()
        if hasattr(choice0, "message") and isinstance(choice0.message, dict):
            return str(choice0.message.get("content", "")).strip()
    except Exception:
        pass
    return ""


def _extract_logprobs_from_chat(response: Any) -> List[TokenLogprob]:
    tokens: List[TokenLogprob] = []
    try:
        lp = response.choices[0].logprobs
        if not lp:
            return tokens
        content_list = getattr(lp, "content", None)
        if not content_list:
            return tokens
        # Iterate over token positions
        for idx, item in enumerate(content_list):
            chosen_token = getattr(item, "token", None)
            chosen_logprob = getattr(item, "logprob", None)
            top_list = getattr(item, "top_logprobs", None) or []
            alt: List[Tuple[str, float]] = []
            for cand in top_list:
                tok = getattr(cand, "token", None)
                lpv = getattr(cand, "logprob", None)
                if tok is not None and lpv is not None and tok != chosen_token:
                    alt.append((str(tok), float(lpv)))
            if chosen_token is not None and chosen_logprob is not None:
                tokens.append(TokenLogprob(token=str(chosen_token), logprob=float(chosen_logprob), alternatives=alt))
    except Exception:
        return []
    return tokens


def aggregate_logprobs(tokens: List[TokenLogprob]) -> Dict[str, float]:
    if not tokens:
        return {"avg_logprob": float("nan"), "sum_logprob": float("nan"), "sequence_probability": float("nan")}
    logps = [t.logprob for t in tokens]
    avg = float(sum(logps) / len(logps))
    total = float(sum(logps))
    seq_prob = float(math.exp(total))
    return {"avg_logprob": avg, "sum_logprob": total, "sequence_probability": seq_prob}


def normalize_option_code(token_text: str) -> str:
    text = token_text.strip().strip('"').strip("'")
    i = 0
    while i < len(text) and text[i].isdigit():
        i += 1
    return text[:i] if i > 0 else text


def build_option_distribution_from_step(tokens: List[TokenLogprob], option_codes: List[str]) -> Optional[Dict[str, float]]:
    codes_set = set(option_codes)
    for idx, chosen in enumerate(tokens):
        chosen_code = normalize_option_code(chosen.token)
        if chosen_code in codes_set:
            probs: Dict[str, float] = {}
            probs[chosen_code] = math.exp(chosen.logprob) if not math.isnan(chosen.logprob) else 0.0
            for alt_token, alt_logp in chosen.alternatives:
                alt_code = normalize_option_code(alt_token)
                if alt_code in codes_set:
                    probs[alt_code] = max(probs.get(alt_code, 0.0), math.exp(alt_logp) if not math.isnan(alt_logp) else 0.0)
            total = sum(probs.values())
            if total > 0:
                for k in list(probs.keys()):
                    probs[k] = probs[k] / total
            else:
                n = float(len(option_codes))
                probs = {c: 1.0 / n for c in option_codes}
            return probs
    return None


def get_question_distribution(df: pd.DataFrame, question: str) -> pd.Series:
    question_data = df[question]
    question_data = question_data.astype(str)
    question_data = question_data[question_data.notna() & (question_data != "") & (question_data.str.strip() != "")]
    return question_data.value_counts(normalize=True)


def compare_distributions(d1: pd.Series, d2: pd.Series, num_options: int) -> float:
    if not d1.index.equals(d2.index):
        d1 = d1.reindex(d2.index, fill_value=0)
    wd = wasserstein_distance(d1, d2)
    if num_options == 1:
        return 1.0
    return 1.0 - (wd / (num_options - 1))


def _series_to_probs(series: pd.Series, order: List[str]) -> List[float]:
    arr = [float(series.get(k, 0.0)) for k in order]
    total = sum(arr)
    if total <= 0:
        n = float(len(order))
        return [1.0 / n for _ in order]
    return [v / total for v in arr]


def jensen_shannon_divergence(p_series: pd.Series, q_series: pd.Series, order: List[str]) -> float:
    p = _series_to_probs(p_series, order)
    q = _series_to_probs(q_series, order)
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    def _kl(a, b) -> float:
        s = 0.0
        for ai, bi in zip(a, b):
            if ai > 0.0 and bi > 0.0:
                s += ai * (math.log(ai / bi, 2))
        return s
    jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(jsd)


def hellinger_distance(p_series: pd.Series, q_series: pd.Series, order: List[str]) -> float:
    p = _series_to_probs(p_series, order)
    q = _series_to_probs(q_series, order)
    s = 0.0
    for pi, qi in zip(p, q):
        s += (math.sqrt(pi) - math.sqrt(qi)) ** 2
    return math.sqrt(s) / math.sqrt(2.0)


def bootstrap_ci(values: List[float], samples: int = 500, alpha: float = 0.05) -> Tuple[float, float]:
    if not values:
        return (float('nan'), float('nan'))
    n = len(values)
    rng = random.Random(1337)
    estimates: List[float] = []
    for _ in range(int(samples)):
        sample = [values[rng.randrange(0, n)] for _ in range(n)]
        estimates.append(float(sum(sample) / len(sample)))
    estimates.sort()
    lo_idx = int(alpha / 2 * len(estimates))
    hi_idx = int((1 - alpha / 2) * len(estimates)) - 1
    lo = estimates[max(0, min(lo_idx, len(estimates) - 1))]
    hi = estimates[max(0, min(hi_idx, len(estimates) - 1))]
    return (lo, hi)


def generate_with_logprobs_chat(
    client: Any,
    model_id: str,
    prompt: str,
    top_alternatives: int,
    max_output_tokens: Optional[int],
    temperature: Optional[float],
    top_p: Optional[float],
):
    kwargs: Dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "logprobs": True,
        "top_logprobs": int(top_alternatives),
    }
    if max_output_tokens is not None:
        kwargs["max_tokens"] = int(max_output_tokens)
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    if top_p is not None:
        kwargs["top_p"] = float(top_p)
    return client.chat.completions.create(**kwargs)


def write_jsonl(records: Iterable[Dict[str, Any]], output_path: Optional[str]) -> None:
    if output_path is None or output_path == "-":
        out = sys.stdout
        close_after = False
    else:
        out = open(output_path, "w", encoding="utf-8")
        close_after = True
    try:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
    finally:
        if close_after:
            out.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate OpenAI with logprobs (Chat Completions)")
    # Accept Gemini-like args for compatibility; project/location are unused
    parser.add_argument("--project_id", type=str, default=None, help="Unused for OpenAI; accepted for CLI parity")
    parser.add_argument("--location", type=str, default="global", help="Unused for OpenAI; accepted for CLI parity")
    parser.add_argument("--model_id", type=str, default="gpt-4o-mini", help="OpenAI model ID (default: gpt-4o-mini)")
    parser.add_argument("--input", dest="input_path", type=str, required=False, help="Input file (.txt lines or .csv with 'prompt' column)")
    parser.add_argument("--prompt", dest="single_prompt", type=str, required=False, help="Single prompt string (overrides --input if provided)")
    parser.add_argument("--output", dest="output_path", type=str, default="-", help="Output JSONL file path or '-' for stdout")
    parser.add_argument("--logprobs", dest="top_alternatives", type=int, default=3, help="Number of top alternative tokens to return (1-20)")
    parser.add_argument("--classification_choices", type=str, default=None, help="Comma-separated enum choices for classification mode")
    parser.add_argument("--choices_file", type=str, default=None, help="Path to JSON array of strings for classification choices")
    parser.add_argument("--max_output_tokens", type=int, default=None, help="Optional max output tokens")
    parser.add_argument("--temperature", type=float, default=None, help="Optional sampling temperature")
    parser.add_argument("--top_p", type=float, default=None, help="Optional nucleus sampling top_p")
    # Full evaluation flow (Translate parity)
    parser.add_argument("--region", type=str, default=None, help="Region directory under Translate/ (e.g., IND, EA, SEA)")
    parser.add_argument("--country", type=str, default=None, help="Country code used in JSON filename (e.g., ind, hk, id)")
    parser.add_argument("--language", type=str, default=None, help="Language code used in JSON filename (e.g., en, hi, ja)")
    parser.add_argument("--secondary-filter-var", dest="secondary_filter_var", type=str, default=None)
    parser.add_argument("--secondary-filter-value", dest="secondary_filter_value", type=str, default=None)
    parser.add_argument("--cot", action="store_true", help="Enable chain-of-thought (two-step) before final answer")
    parser.add_argument("--repeats", type=int, default=1, help="Repeat generations per question and average")
    parser.add_argument("--per_request_delay", type=float, default=1.0, help="Sleep seconds between requests")
    parser.add_argument("--retry_max_attempts", type=int, default=6, help="Max retry attempts per request")
    parser.add_argument("--retry_base", type=float, default=2.0, help="Exponential backoff base seconds")
    parser.add_argument("--retry_max_backoff", type=float, default=30.0, help="Max backoff seconds")

    args = parser.parse_args(argv)

    if args.top_alternatives < 1 or args.top_alternatives > 20:
        print("Error: --logprobs must be between 1 and 20.", file=sys.stderr)
        return 2

    OpenAI = _import_openai_or_exit()
    api_key = _require_openai_key()
    client = OpenAI(api_key=api_key)

    # If region/country/language are provided, run full evaluation flow
    if args.region and args.country and args.language:
        base_dir = os.path.join(os.getcwd(), "Translate", args.region)
        responses_path = os.path.join(base_dir, "responses.csv")
        questions_path = os.path.join(base_dir, f"{args.country}_{args.language}.json")

        responses_df = pd.read_csv(responses_path)
        with open(questions_path, 'r', encoding='utf-8') as f:
            questions_map = json.load(f)

        if args.secondary_filter_var is not None and args.secondary_filter_value is not None:
            if args.secondary_filter_var in responses_df.columns:
                responses_df = responses_df[responses_df[args.secondary_filter_var] == args.secondary_filter_value]

        scores_wd: List[float] = []
        scores_jsd: List[float] = []
        scores_hell: List[float] = []
        weights: List[float] = []
        results_rows: List[Tuple[str, float, float, float]] = []

        for question_key in tqdm(questions_map, desc="OpenAI logprobs (representativeness)"):
            if question_key in ['COUNTRY', 'QRID', 'weight', 'QMLangRec']:
                continue
            entry = questions_map[question_key]
            if not (isinstance(entry, dict) and 'question' in entry and 'options' in entry and isinstance(entry['options'], dict)):
                continue

            options_dict: Dict[str, str] = entry['options']
            try:
                option_codes = sorted(options_dict.keys(), key=lambda x: int(x))
            except Exception:
                option_codes = list(options_dict.keys())

            qd1 = get_question_distribution(responses_df, question_key)

            lines = [f"Question: {entry['question']}"]
            for code in option_codes:
                lines.append(f"{code}: {options_dict[code]}")
            lines.append(f"\nReply with only one option code from {{{', '.join(option_codes)}}}.")
            prompt = "\n".join(lines)

            reasoning_text: Optional[str] = None
            if args.cot:
                attempts = 0
                while True:
                    try:
                        cot_resp = client.chat.completions.create(
                            model=args.model_id,
                            messages=[{"role": "user", "content": f"{prompt}\n\nLet's think step by step."}],
                            temperature=(args.temperature if args.temperature is not None else 0.0),
                            max_tokens=256,
                            top_p=float(args.top_p) if args.top_p is not None else None,
                        )
                        reasoning_text = _extract_text_from_chat(cot_resp).strip()
                        break
                    except Exception as exc:
                        message = str(exc)
                        retryable = ("429" in message) or ("quota" in message.lower())
                        attempts += 1
                        if not retryable or attempts >= args.retry_max_attempts:
                            raise
                        sleep_secs = min(args.retry_max_backoff, args.retry_base * (2 ** (attempts - 1)))
                        jitter = 0.2 * sleep_secs
                        time.sleep(sleep_secs + random.uniform(-jitter, jitter))
                time.sleep(max(0.0, args.per_request_delay))

            final_prompt = prompt if not reasoning_text else f"{prompt}\n\n{reasoning_text}\n\nAnswer with only one code."

            desired_logprobs = max(1, min(19, len(option_codes)))
            agg: Optional[Dict[str, float]] = None
            for _ in range(max(1, int(args.repeats))):
                attempts = 0
                while True:
                    try:
                        resp = generate_with_logprobs_chat(
                            client=client,
                            model_id=args.model_id,
                            prompt=final_prompt,
                            top_alternatives=desired_logprobs,
                            max_output_tokens=2,
                            temperature=(args.temperature if args.temperature is not None else 0.0),
                            top_p=args.top_p,
                        )
                        break
                    except Exception as exc:
                        message = str(exc)
                        retryable = ("429" in message) or ("quota" in message.lower())
                        attempts += 1
                        if not retryable or attempts >= args.retry_max_attempts:
                            raise
                        sleep_secs = min(args.retry_max_backoff, args.retry_base * (2 ** (attempts - 1)))
                        jitter = 0.2 * sleep_secs
                        time.sleep(sleep_secs + random.uniform(-jitter, jitter))
                time.sleep(max(0.0, args.per_request_delay))

                tokens = _extract_logprobs_from_chat(resp)
                dist_dict = build_option_distribution_from_step(tokens, option_codes)
                if dist_dict is None:
                    n = float(len(option_codes))
                    dist_dict = {c: 1.0 / n for c in option_codes}
                if agg is None:
                    agg = dict(dist_dict)
                else:
                    for k, v in dist_dict.items():
                        agg[k] = agg.get(k, 0.0) + v

            if agg is None:
                n = float(len(option_codes))
                agg = {c: 1.0 / n for c in option_codes}
            else:
                r = float(max(1, int(args.repeats)))
                for k in list(agg.keys()):
                    agg[k] = agg[k] / r

            qd2 = pd.Series(agg, dtype=float)

            wd = compare_distributions(qd1, qd2, num_options=len(option_codes))
            jsd = jensen_shannon_divergence(qd1.reindex(option_codes, fill_value=0.0), qd2.reindex(option_codes, fill_value=0.0), option_codes)
            hell = hellinger_distance(qd1.reindex(option_codes, fill_value=0.0), qd2.reindex(option_codes, fill_value=0.0), option_codes)

            weight_n = float(responses_df[question_key].notna().sum())
            print(f"Question {question_key} WD={wd:.4f} JSD={jsd:.4f} HELL={hell:.4f}")
            scores_wd.append(wd)
            scores_jsd.append(jsd)
            scores_hell.append(hell)
            weights.append(weight_n)
            results_rows.append((question_key, wd, jsd, hell))

        def _weighted_mean(vals: List[float], w: List[float]) -> float:
            if not vals:
                return float('nan')
            if not w or sum(w) <= 0:
                return float(sum(vals) / len(vals))
            return float(sum(v * wi for v, wi in zip(vals, w)) / sum(w))

        avg_wd = float(sum(scores_wd) / len(scores_wd)) if scores_wd else 0.0
        wavg_wd = _weighted_mean(scores_wd, weights)
        ci_lo_wd, ci_hi_wd = bootstrap_ci(scores_wd)

        avg_jsd = float(sum(scores_jsd) / len(scores_jsd)) if scores_jsd else 0.0
        wavg_jsd = _weighted_mean(scores_jsd, weights)
        ci_lo_jsd, ci_hi_jsd = bootstrap_ci(scores_jsd)

        avg_hell = float(sum(scores_hell) / len(scores_hell)) if scores_hell else 0.0
        wavg_hell = _weighted_mean(scores_hell, weights)
        ci_lo_hell, ci_hi_hell = bootstrap_ci(scores_hell)

        try:
            logs_dir = os.path.join(os.getcwd(), "evaluation_logs")
            os.makedirs(logs_dir, exist_ok=True)
            csv_path = os.path.join(logs_dir, f"{args.country}-{args.language}.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow(["question", "wd", "jsd", "hell"])
                for q, wd_v, jsd_v, hell_v in results_rows:
                    writer.writerow([q, f"{wd_v:.6f}", f"{jsd_v:.6f}", f"{hell_v:.6f}"])
                writer.writerow(["WD_mean", f"{avg_wd:.6f}"])
                writer.writerow(["WD_weighted_mean", f"{wavg_wd:.6f}"])
                writer.writerow(["WD_CI95", f"{ci_lo_wd:.6f}-{ci_hi_wd:.6f}"])
                writer.writerow(["JSD_mean", f"{avg_jsd:.6f}"])
                writer.writerow(["JSD_weighted_mean", f"{wavg_jsd:.6f}"])
                writer.writerow(["JSD_CI95", f"{ci_lo_jsd:.6f}-{ci_hi_jsd:.6f}"])
                writer.writerow(["HELL_mean", f"{avg_hell:.6f}"])
                writer.writerow(["HELL_weighted_mean", f"{wavg_hell:.6f}"])
                writer.writerow(["HELL_CI95", f"{ci_lo_hell:.6f}-{ci_hi_hell:.6f}"])
            ts = time.strftime("%Y%m%d_%H%M%S")
            averages_path = os.path.join(logs_dir, f"averages_openai_{ts}.csv")
            with open(averages_path, "w", encoding="utf-8", newline="") as f_avg:
                writer = csv.writer(f_avg)
                writer.writerow(["region","country","language","model_id","temperature","top_p","logprobs","cot","repeats",
                                 "wd_mean","wd_weighted_mean","wd_ci95","jsd_mean","jsd_weighted_mean","jsd_ci95","hell_mean","hell_weighted_mean","hell_ci95","num_questions"])
                writer.writerow([args.region, args.country, args.language, args.model_id,
                                 args.temperature if args.temperature is not None else 0.0,
                                 args.top_p if args.top_p is not None else "",
                                 max(1, min(19, len(option_codes))), int(args.cot), int(args.repeats),
                                 f"{avg_wd:.6f}", f"{wavg_wd:.6f}", f"{ci_lo_wd:.6f}-{ci_hi_wd:.6f}",
                                 f"{avg_jsd:.6f}", f"{wavg_jsd:.6f}", f"{ci_lo_jsd:.6f}-{ci_hi_jsd:.6f}",
                                 f"{avg_hell:.6f}", f"{wavg_hell:.6f}", f"{ci_lo_hell:.6f}-{ci_hi_hell:.6f}",
                                 len(results_rows)])
            print(f"Saved cleaned results to: {csv_path}")
        except Exception as exc:
            print(f"Warning: failed to write cleaned CSV results: {exc}", file=sys.stderr)

        print("=" * 20)
        print("Average Representativeness:", f"{avg_wd:.6f}")
        return 0

    # Else: single-prompt/classification JSONL mode
    choices = normalize_choices(args.classification_choices, args.choices_file)
    records: List[Dict[str, Any]] = []
    if args.single_prompt:
        prompts = [args.single_prompt]
    elif args.input_path:
        prompts = read_prompts(args.input_path)
    else:
        prompts = [
            "I am not sure if I really like this restaurant a lot.",
        ]
        if not choices:
            prompts = ["Explain why the sky appears blue at midday in one sentence."]

    for prompt in prompts:
        try:
            resp = generate_with_logprobs_chat(
                client=client,
                model_id=args.model_id,
                prompt=prompt,
                top_alternatives=args.top_alternatives,
                max_output_tokens=args.max_output_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            text = _extract_text_from_chat(resp)
            toks = _extract_logprobs_from_chat(resp)
            aggregates = aggregate_logprobs(toks)

            record: Dict[str, Any] = {
                "prompt": prompt,
                "model_id": args.model_id,
                "text": text,
                "tokens": [
                    {
                        "token": t.token,
                        "logprob": t.logprob,
                        "alternatives": [
                            {"token": alt_tok, "logprob": alt_lp} for alt_tok, alt_lp in t.alternatives
                        ],
                    }
                    for t in toks
                ],
                "avg_logprob": aggregates["avg_logprob"],
                "sum_logprob": aggregates["sum_logprob"],
                "sequence_probability": aggregates["sequence_probability"],
            }

            if choices and toks:
                first_token_lp = toks[0].logprob
                record["classification"] = text.strip() if text else None
                record["classification_confidence"] = float(math.exp(first_token_lp))

            records.append(record)
        except Exception as exc:
            records.append({"prompt": prompt, "model_id": args.model_id, "error": str(exc)})

    write_jsonl(records, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


