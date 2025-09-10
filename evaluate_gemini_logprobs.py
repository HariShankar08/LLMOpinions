"""
Evaluate Gemini models with logprobs on Vertex AI.

This script demonstrates generating responses with token-level log probabilities
using the Google GenAI SDK (Vertex AI), inspired by Google's developer blog
"Unlock Gemini’s reasoning: A step-by-step guide to logprobs on Vertex AI".

Usage examples:

  - Minimal (reads prompts from a text file, one per line):
      python evaluate_gemini_logprobs.py \
        --project_id YOUR_GCP_PROJECT \
        --input prompts.txt \
        --output results.jsonl

  - Classification with a fixed set of choices (schema enum):
      python evaluate_gemini_logprobs.py \
        --project_id YOUR_GCP_PROJECT \
        --input prompts.txt \
        --classification_choices "Positive,Negative,Neutral" \
        --output classification.jsonl

Environment variables:
  - PROJECT_ID or GOOGLE_CLOUD_PROJECT can be used instead of --project_id

Requires:
  - python -m pip install -U google-genai

Notes:
  - This script is a self-contained utility and does not modify existing pipelines.
  - Output is JSON Lines (JSONL), one record per prompt, with token logprobs.
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

# Optional deps for full evaluation flow parity with noCoT
import pandas as pd
from scipy.stats import wasserstein_distance
from tqdm import tqdm


def _import_genai_or_exit() -> Tuple[Any, Any]:
    """Import google-genai SDK and exit with a helpful message if unavailable."""
    try:
        from google import genai  # type: ignore
        from google.genai.types import GenerateContentConfig  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        message = (
            "Missing dependency: google-genai.\n\n"
            "Install with:\n  python -m pip install -U google-genai\n\n"
            f"Original error: {exc}"
        )
        print(message, file=sys.stderr)
        sys.exit(2)
    return genai, GenerateContentConfig


def resolve_project_id(cli_project_id: Optional[str]) -> str:
    """Resolve project id from CLI or environment variables."""
    project_id = (
        cli_project_id
        or os.getenv("PROJECT_ID")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
    )
    if not project_id:
        print(
            "Error: --project_id not provided and neither PROJECT_ID nor GOOGLE_CLOUD_PROJECT are set.",
            file=sys.stderr,
        )
        sys.exit(2)
    return project_id


def read_prompts(input_path: str) -> List[str]:
    """Read prompts from a .txt (one per line) or .csv (column named 'prompt')."""
    path_lower = input_path.lower()
    if path_lower.endswith(".txt"):
        with open(input_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    if path_lower.endswith(".csv"):
        prompts: List[str] = []
        with open(input_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "prompt" not in reader.fieldnames if reader.fieldnames else True:
                print(
                    "Error: CSV must include a 'prompt' column.",
                    file=sys.stderr,
                )
                sys.exit(2)
            for row in reader:
                value = (row.get("prompt") or "").strip()
                if value:
                    prompts.append(value)
        return prompts
    print("Error: input must be a .txt or .csv file.", file=sys.stderr)
    sys.exit(2)


def normalize_choices(choices_arg: Optional[str], choices_file: Optional[str]) -> Optional[List[str]]:
    """Return a list of enum choices for classification, or None for free text."""
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


def extract_text(response: Any) -> str:
    """Best-effort extraction of response text from google-genai Response."""
    # Preferred attribute on google-genai responses
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    # Fallback: inspect candidates/parts
    candidates = getattr(response, "candidates", None)
    if candidates and isinstance(candidates, list) and candidates:
        content = getattr(candidates[0], "content", None)
        if content:
            parts = getattr(content, "parts", None)
            if parts and isinstance(parts, list) and parts:
                maybe_text = getattr(parts[0], "text", None)
                if isinstance(maybe_text, str):
                    return maybe_text
    return ""


def extract_logprobs(response: Any) -> List[TokenLogprob]:
    """Extract chosen tokens, their logprobs, and alternatives from the response."""
    tokens: List[TokenLogprob] = []
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return tokens

    first = candidates[0]
    logprobs_result = getattr(first, "logprobs_result", None)
    if not logprobs_result:
        return tokens

    chosen = getattr(logprobs_result, "chosen_candidates", [])
    top = getattr(logprobs_result, "top_candidates", [])

    for i, chosen_candidate in enumerate(chosen):
        token_value = getattr(chosen_candidate, "token", "")
        token_logp = float(getattr(chosen_candidate, "log_probability", float("nan")))
        alt_list: List[Tuple[str, float]] = []
        if i < len(top):
            top_alts = getattr(top[i], "candidates", [])
            for alt in top_alts:
                alt_token = getattr(alt, "token", "")
                alt_logp = float(getattr(alt, "log_probability", float("nan")))
                if alt_token != token_value:
                    alt_list.append((alt_token, alt_logp))
        tokens.append(TokenLogprob(token=token_value, logprob=token_logp, alternatives=alt_list))
    return tokens


def aggregate_logprobs(tokens: List[TokenLogprob]) -> Dict[str, float]:
    """Compute simple aggregates over token logprobs for convenience."""
    if not tokens:
        return {"avg_logprob": float("nan"), "sum_logprob": float("nan"), "sequence_probability": float("nan")}
    logps = [t.logprob for t in tokens]
    avg = float(sum(logps) / len(logps))
    total = float(sum(logps))
    seq_prob = float(math.exp(total))  # P(sequence) under independence assumption across steps
    return {"avg_logprob": avg, "sum_logprob": total, "sequence_probability": seq_prob}


def normalize_option_code(token_text: str) -> str:
    """Normalize a token to match option code forms (strip quotes/whitespace)."""
    text = token_text.strip().strip('"').strip("'")
    # Keep only leading signless numeric part if present
    # e.g., '1,' -> '1'
    i = 0
    while i < len(text) and text[i].isdigit():
        i += 1
    return text[:i] if i > 0 else text


def build_option_distribution_from_step(tokens: List[TokenLogprob], option_codes: List[str]) -> Optional[Dict[str, float]]:
    """Find the first step where the chosen token is one of the option codes and
    build a probability distribution over codes from that step's alternatives.

    Returns None if no suitable step found.
    """
    codes_set = set(option_codes)
    for idx, chosen in enumerate(tokens):
        chosen_code = normalize_option_code(chosen.token)
        if chosen_code in codes_set:
            # Gather probabilities from this step
            probs: Dict[str, float] = {}
            # Include chosen token
            probs[chosen_code] = math.exp(chosen.logprob) if not math.isnan(chosen.logprob) else 0.0
            # Include alternatives
            for alt_token, alt_logp in chosen.alternatives:
                alt_code = normalize_option_code(alt_token)
                if alt_code in codes_set:
                    probs[alt_code] = max(probs.get(alt_code, 0.0), math.exp(alt_logp) if not math.isnan(alt_logp) else 0.0)
            # Normalize
            total = sum(probs.values())
            if total > 0:
                for k in list(probs.keys()):
                    probs[k] = probs[k] / total
            else:
                # Degenerate distribution
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
        # uniform fallback
        n = float(len(order))
        return [1.0 / n for _ in order]
    return [v / total for v in arr]


def jensen_shannon_divergence(p_series: pd.Series, q_series: pd.Series, order: List[str]) -> float:
    """JSD in [0,1] using log base 2.
    Returns divergence (lower is better)."""
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
    # Already in [0,1] with base-2 logs
    return float(jsd)


def hellinger_distance(p_series: pd.Series, q_series: pd.Series, order: List[str]) -> float:
    """Hellinger distance in [0,1]. Returns distance (lower is better)."""
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


def generate_with_logprobs(
    client: Any,
    model_id: str,
    prompt: str,
    response_mime_type: str,
    response_schema: Optional[Dict[str, Any]],
    top_alternatives: int,
    max_output_tokens: Optional[int],
    temperature: Optional[float],
    top_p: Optional[float],
    GenerateContentConfig: Any,
) -> Any:
    config_kwargs: Dict[str, Any] = {
        "response_mime_type": response_mime_type,
        "response_logprobs": True,
        "logprobs": int(top_alternatives),
    }
    if response_schema is not None:
        config_kwargs["response_schema"] = response_schema
    if max_output_tokens is not None:
        config_kwargs["max_output_tokens"] = int(max_output_tokens)
    if temperature is not None:
        config_kwargs["temperature"] = float(temperature)
    if top_p is not None:
        config_kwargs["top_p"] = float(top_p)
    # Try config= style first (works across a range of SDK versions)
    try:
        return client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=GenerateContentConfig(**config_kwargs),
        )
    except TypeError as err:
        # Fall back to generation_config if config is unsupported
        try:
            return client.models.generate_content(
                model=model_id,
                contents=prompt,
                generation_config=GenerateContentConfig(**config_kwargs),
            )
        except TypeError as err_gen:
            err = err_gen
        if "generation_config" in str(err):
            # Older SDK: pass config fields at top-level
            flattened_kwargs = dict(config_kwargs)
            try:
                return client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    **flattened_kwargs,
                )
            except TypeError as err2:
                # Even older SDK: remove structured output keys unsupported by older versions
                if "response_mime_type" in str(err2) or "response_schema" in str(err2):
                    flattened_kwargs.pop("response_mime_type", None)
                    flattened_kwargs.pop("response_schema", None)
                    try:
                        return client.models.generate_content(
                            model=model_id,
                            contents=prompt,
                            **flattened_kwargs,
                        )
                    except TypeError as err3:
                        if "response_logprobs" in str(err3) or "logprobs" in str(err3):
                            flattened_kwargs.pop("response_logprobs", None)
                            flattened_kwargs.pop("logprobs", None)
                            return client.models.generate_content(
                                model=model_id,
                                contents=prompt,
                                **flattened_kwargs,
                            )
                        raise
                if "response_logprobs" in str(err2) or "logprobs" in str(err2):
                    flattened_kwargs.pop("response_logprobs", None)
                    flattened_kwargs.pop("logprobs", None)
                    return client.models.generate_content(
                        model=model_id,
                        contents=prompt,
                        **flattened_kwargs,
                    )
                raise
        # If the error was not about generation_config, try removing structured keys directly
        if "response_mime_type" in str(err) or "response_schema" in str(err):
            flattened_kwargs = dict(config_kwargs)
            flattened_kwargs.pop("response_mime_type", None)
            flattened_kwargs.pop("response_schema", None)
            return client.models.generate_content(
                model=model_id,
                contents=prompt,
                **flattened_kwargs,
            )
        if "response_logprobs" in str(err) or "logprobs" in str(err):
            flattened_kwargs = dict(config_kwargs)
            flattened_kwargs.pop("response_logprobs", None)
            flattened_kwargs.pop("logprobs", None)
            return client.models.generate_content(
                model=model_id,
                contents=prompt,
                **flattened_kwargs,
            )
        raise


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
    parser = argparse.ArgumentParser(description="Evaluate Gemini with logprobs (Vertex AI)")
    parser.add_argument("--project_id", type=str, default=None, help="GCP project ID (or set PROJECT_ID/GOOGLE_CLOUD_PROJECT)")
    parser.add_argument("--location", type=str, default="global", help="Vertex AI location (default: global)")
    parser.add_argument("--model_id", type=str, default="gemini-2.5-flash", help="Gemini model ID (default: gemini-2.5-flash)")
    parser.add_argument("--input", dest="input_path", type=str, required=False, help="Input file (.txt lines or .csv with 'prompt' column)")
    parser.add_argument("--prompt", dest="single_prompt", type=str, required=False, help="Single prompt string (overrides --input if provided)")
    parser.add_argument("--output", dest="output_path", type=str, default="-", help="Output JSONL file path or '-' for stdout")
    parser.add_argument("--logprobs", dest="top_alternatives", type=int, default=3, help="Number of top alternative tokens to return (1-20)")
    parser.add_argument("--classification_choices", type=str, default=None, help="Comma-separated enum choices for classification mode")
    parser.add_argument("--choices_file", type=str, default=None, help="Path to JSON array of strings for classification choices")
    parser.add_argument("--max_output_tokens", type=int, default=None, help="Optional max output tokens")
    parser.add_argument("--temperature", type=float, default=None, help="Optional sampling temperature")
    parser.add_argument("--top_p", type=float, default=None, help="Optional nucleus sampling top_p")
    # Full evaluation flow (noCoT parity)
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

    # If region/country/language are provided, run full evaluation flow (noCoT parity)
    if args.region and args.country and args.language:
        genai, GenerateContentConfig = _import_genai_or_exit()
        project_id = resolve_project_id(args.project_id)
        client = genai.Client(vertexai=True, project=project_id, location=args.location)

        base_dir = os.path.join(os.getcwd(), "Translate", args.region)
        responses_path = os.path.join(base_dir, "responses.csv")
        questions_path = os.path.join(base_dir, f"{args.country}_{args.language}.json")

        # Load data
        responses_df = pd.read_csv(responses_path)
        with open(questions_path, 'r', encoding='utf-8') as f:
            questions_map = json.load(f)

        # Secondary filter (optional)
        if args.secondary_filter_var is not None and args.secondary_filter_value is not None:
            if args.secondary_filter_var in responses_df.columns:
                responses_df = responses_df[responses_df[args.secondary_filter_var] == args.secondary_filter_value]

        scores_wd: List[float] = []
        scores_jsd: List[float] = []
        scores_hell: List[float] = []
        weights: List[float] = []
        results_rows: List[Tuple[str, float, float, float]] = []
        for question_key in tqdm(questions_map, desc="Gemini logprobs (representativeness)"):
            if question_key in ['COUNTRY', 'QRID', 'weight', 'QMLangRec']:
                continue
            entry = questions_map[question_key]
            if not (isinstance(entry, dict) and 'question' in entry and 'options' in entry and isinstance(entry['options'], dict)):
                continue

            # Human-readable label mapping; but we classify by numeric codes for reliability
            options_dict: Dict[str, str] = entry['options']
            try:
                option_codes = sorted(options_dict.keys(), key=lambda x: int(x))
            except Exception:
                option_codes = list(options_dict.keys())

            # qd1: observed distribution from responses; ensure full support
            qd1 = get_question_distribution(responses_df, question_key)

            # Build concise prompt: require only the option code as output
            lines = [f"Question: {entry['question']}"]
            for code in option_codes:
                lines.append(f"{code}: {options_dict[code]}")
            lines.append(f"\nReply with only one option code from {{{', '.join(option_codes)}}}.")
            prompt = "\n".join(lines)

            # Optional CoT: generate brief reasoning text first
            reasoning_text: Optional[str] = None
            if args.cot:
                attempts = 0
                while True:
                    try:
                        reasoning_resp = client.models.generate_content(
                            model=args.model_id,
                            contents=f"{prompt}\n\nLet's think step by step.",
                            config=GenerateContentConfig(
                                temperature=(args.temperature if args.temperature is not None else 0.0),
                                **({"top_p": float(args.top_p)} if args.top_p is not None else {}),
                                response_mime_type="text/plain",
                                max_output_tokens=256,
                            ),
                        )
                        reasoning_text = extract_text(reasoning_resp).strip()
                        break
                    except Exception as exc:
                        message = str(exc)
                        retryable = ("429" in message) or ("RESOURCE_EXHAUSTED" in message) or ("quota" in message.lower())
                        attempts += 1
                        if not retryable or attempts >= args.retry_max_attempts:
                            raise
                        sleep_secs = min(args.retry_max_backoff, args.retry_base * (2 ** (attempts - 1)))
                        jitter = 0.2 * sleep_secs
                        time.sleep(sleep_secs + random.uniform(-jitter, jitter))
                time.sleep(max(0.0, args.per_request_delay))

            # Build final prompt possibly including reasoning
            final_prompt = prompt if not reasoning_text else f"{prompt}\n\n{reasoning_text}\n\nAnswer with only one code."

            # Repeats: average distributions across multiple runs
            desired_logprobs = max(1, min(19, len(option_codes)))
            agg: Optional[Dict[str, float]] = None
            for _ in range(max(1, int(args.repeats))):
                attempts = 0
                while True:
                    try:
                        response = generate_with_logprobs(
                            client=client,
                            model_id=args.model_id,
                            prompt=final_prompt,
                            response_mime_type="text/plain",
                            response_schema=None,
                            top_alternatives=desired_logprobs,
                            max_output_tokens=2,
                            temperature=(args.temperature if args.temperature is not None else 0.0),
                            top_p=args.top_p,
                            GenerateContentConfig=GenerateContentConfig,
                        )
                        break
                    except Exception as exc:
                        message = str(exc)
                        retryable = (
                            "429" in message
                            or "RESOURCE_EXHAUSTED" in message
                            or "quota" in message.lower()
                        )
                        attempts += 1
                        if not retryable or attempts >= args.retry_max_attempts:
                            raise
                        sleep_secs = min(args.retry_max_backoff, args.retry_base * (2 ** (attempts - 1)))
                        jitter = 0.2 * sleep_secs
                        time.sleep(sleep_secs + random.uniform(-jitter, jitter))
                time.sleep(max(0.0, args.per_request_delay))

                tokens = extract_logprobs(response)
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

            # qd2 averaged
            qd2 = pd.Series(agg, dtype=float)

            # Compute metrics
            wd = compare_distributions(qd1, qd2, num_options=len(option_codes))
            jsd = jensen_shannon_divergence(qd1.reindex(option_codes, fill_value=0.0), qd2.reindex(option_codes, fill_value=0.0), option_codes)
            hell = hellinger_distance(qd1.reindex(option_codes, fill_value=0.0), qd2.reindex(option_codes, fill_value=0.0), option_codes)

            # Weight: respondent count for the question (non-null rows)
            weight_n = float(responses_df[question_key].notna().sum())
            print(f"Question {question_key} WD={wd:.4f} JSD={jsd:.4f} HELL={hell:.4f}")
            scores_wd.append(wd)
            scores_jsd.append(jsd)
            scores_hell.append(hell)
            weights.append(weight_n)
            results_rows.append((question_key, wd, jsd, hell))

        # Aggregations
        def _weighted_mean(vals: List[float], w: List[float]) -> float:
            if not vals:
                return float('nan')
            if not w or sum(w) <= 0:
                return float(sum(vals) / len(vals))
            return float(sum(v * wi for v, wi in zip(vals, w)) / sum(w))

        avg_wd = float(sum(scores_wd) / len(scores_wd)) if scores_wd else 0.0
        med_wd = float(sorted(scores_wd)[len(scores_wd)//2]) if scores_wd else 0.0
        wavg_wd = _weighted_mean(scores_wd, weights)
        ci_lo_wd, ci_hi_wd = bootstrap_ci(scores_wd)

        avg_jsd = float(sum(scores_jsd) / len(scores_jsd)) if scores_jsd else 0.0
        med_jsd = float(sorted(scores_jsd)[len(scores_jsd)//2]) if scores_jsd else 0.0
        wavg_jsd = _weighted_mean(scores_jsd, weights)
        ci_lo_jsd, ci_hi_jsd = bootstrap_ci(scores_jsd)

        avg_hell = float(sum(scores_hell) / len(scores_hell)) if scores_hell else 0.0
        med_hell = float(sorted(scores_hell)[len(scores_hell)//2]) if scores_hell else 0.0
        wavg_hell = _weighted_mean(scores_hell, weights)
        ci_lo_hell, ci_hi_hell = bootstrap_ci(scores_hell)

        # Write cleaned CSV: question,repre followed by a final Average row; also save a run-level summary
        try:
            logs_dir = os.path.join(os.getcwd(), "evaluation_logs")
            os.makedirs(logs_dir, exist_ok=True)
            csv_path = os.path.join(logs_dir, f"{args.country}-{args.language}.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow(["question", "wd", "jsd", "hell"])
                for q, wd, jsd, hell in results_rows:
                    writer.writerow([q, f"{wd:.6f}", f"{jsd:.6f}", f"{hell:.6f}"])
                writer.writerow(["WD_mean", f"{avg_wd:.6f}"])
                writer.writerow(["WD_median", f"{med_wd:.6f}"])
                writer.writerow(["WD_weighted_mean", f"{wavg_wd:.6f}"])
                writer.writerow(["WD_CI95", f"{ci_lo_wd:.6f}-{ci_hi_wd:.6f}"])
                writer.writerow(["JSD_mean", f"{avg_jsd:.6f}"])
                writer.writerow(["JSD_median", f"{med_jsd:.6f}"])
                writer.writerow(["JSD_weighted_mean", f"{wavg_jsd:.6f}"])
                writer.writerow(["JSD_CI95", f"{ci_lo_jsd:.6f}-{ci_hi_jsd:.6f}"])
                writer.writerow(["HELL_mean", f"{avg_hell:.6f}"])
                writer.writerow(["HELL_median", f"{med_hell:.6f}"])
                writer.writerow(["HELL_weighted_mean", f"{wavg_hell:.6f}"])
                writer.writerow(["HELL_CI95", f"{ci_lo_hell:.6f}-{ci_hi_hell:.6f}"])
            # Save run-level averages row
            ts = time.strftime("%Y%m%d_%H%M%S")
            averages_path = os.path.join(logs_dir, f"averages_gemini_{ts}.csv")
            with open(averages_path, "w", encoding="utf-8", newline="") as f_avg:
                writer = csv.writer(f_avg)
                writer.writerow(["region","country","language","model_id","location","temperature","top_p","logprobs","cot","repeats",
                                 "wd_mean","wd_weighted_mean","wd_ci95","jsd_mean","jsd_weighted_mean","jsd_ci95","hell_mean","hell_weighted_mean","hell_ci95","num_questions"]) 
                writer.writerow([args.region, args.country, args.language, args.model_id, args.location,
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

        print('=' * 20)
        print('Average Representativeness:', avg_score)
        return 0

    # Else: single-prompt/classification JSONL mode (existing behavior)
    genai, GenerateContentConfig = _import_genai_or_exit()
    project_id = resolve_project_id(args.project_id)
    client = genai.Client(vertexai=True, project=project_id, location=args.location)

    choices = normalize_choices(args.classification_choices, args.choices_file)
    if choices:
        response_mime_type = "application/json"
        response_schema = {"type": "STRING", "enum": choices}
    else:
        response_mime_type = "text/plain"
        response_schema = None

    prompts: List[str]
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

    records: List[Dict[str, Any]] = []
    for prompt in prompts:
        try:
            response = generate_with_logprobs(
                client=client,
                model_id=args.model_id,
                prompt=prompt,
                response_mime_type=response_mime_type,
                response_schema=response_schema,
                top_alternatives=args.top_alternatives,
                max_output_tokens=args.max_output_tokens,
                temperature=args.temperature,
                GenerateContentConfig=GenerateContentConfig,
            )
            text = extract_text(response)
            tokens = extract_logprobs(response)
            aggregates = aggregate_logprobs(tokens)

            record: Dict[str, Any] = {
                "prompt": prompt,
                "model_id": args.model_id,
                "location": args.location,
                "text": text,
                "tokens": [
                    {
                        "token": t.token,
                        "logprob": t.logprob,
                        "alternatives": [
                            {"token": alt_tok, "logprob": alt_lp} for alt_tok, alt_lp in t.alternatives
                        ],
                    }
                    for t in tokens
                ],
                "avg_logprob": aggregates["avg_logprob"],
                "sum_logprob": aggregates["sum_logprob"],
                "sequence_probability": aggregates["sequence_probability"],
            }

            # For classification mode, add a simple confidence metric for the 1st token
            if choices and tokens:
                first_token_lp = tokens[0].logprob
                record["classification"] = text.strip() if text else None
                record["classification_confidence"] = float(math.exp(first_token_lp))

            records.append(record)
        except Exception as exc:  # Robust per-prompt error capture
            records.append(
                {
                    "prompt": prompt,
                    "model_id": args.model_id,
                    "location": args.location,
                    "error": str(exc),
                }
            )

    write_jsonl(records, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


