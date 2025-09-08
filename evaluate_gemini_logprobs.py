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


def generate_with_logprobs(
    client: Any,
    model_id: str,
    prompt: str,
    response_mime_type: str,
    response_schema: Optional[Dict[str, Any]],
    top_alternatives: int,
    max_output_tokens: Optional[int],
    temperature: Optional[float],
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
    # Prefer new API style using generation_config; fall back to flattened kwargs
    try:
        return client.models.generate_content(
            model=model_id,
            contents=prompt,
            generation_config=GenerateContentConfig(**config_kwargs),
        )
    except TypeError as err:
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
    parser.add_argument("--model_id", type=str, default="gemini-2.0-flash", help="Gemini model ID (default: gemini-2.5-flash)")
    parser.add_argument("--input", dest="input_path", type=str, required=False, help="Input file (.txt lines or .csv with 'prompt' column)")
    parser.add_argument("--prompt", dest="single_prompt", type=str, required=False, help="Single prompt string (overrides --input if provided)")
    parser.add_argument("--output", dest="output_path", type=str, default="-", help="Output JSONL file path or '-' for stdout")
    parser.add_argument("--logprobs", dest="top_alternatives", type=int, default=3, help="Number of top alternative tokens to return (1-20)")
    parser.add_argument("--classification_choices", type=str, default=None, help="Comma-separated enum choices for classification mode")
    parser.add_argument("--choices_file", type=str, default=None, help="Path to JSON array of strings for classification choices")
    parser.add_argument("--max_output_tokens", type=int, default=None, help="Optional max output tokens")
    parser.add_argument("--temperature", type=float, default=None, help="Optional sampling temperature")
    # Full evaluation flow (noCoT parity)
    parser.add_argument("--region", type=str, default=None, help="Region directory under Translate/ (e.g., IND, EA, SEA)")
    parser.add_argument("--country", type=str, default=None, help="Country code used in JSON filename (e.g., ind, hk, id)")
    parser.add_argument("--language", type=str, default=None, help="Language code used in JSON filename (e.g., en, hi, ja)")
    parser.add_argument("--secondary-filter-var", dest="secondary_filter_var", type=str, default=None)
    parser.add_argument("--secondary-filter-value", dest="secondary_filter_value", type=str, default=None)

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

        scores: List[float] = []
        for question_key in tqdm(questions_map, desc="Gemini logprobs (noCoT parity)"):
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

            # qd1: observed distribution from responses
            qd1 = get_question_distribution(responses_df, question_key)

            # Build concise prompt: require only the option code as output
            lines = [f"Question: {entry['question']}"]
            for code in option_codes:
                lines.append(f"{code}: {options_dict[code]}")
            lines.append(f"\nReply with only one option code from {{{', '.join(option_codes)}}}.")
            prompt = "\n".join(lines)

            # Use plain text to favor single-token numeric outputs
            response = generate_with_logprobs(
                client=client,
                model_id=args.model_id,
                prompt=prompt,
                response_mime_type="text/plain",
                response_schema=None,
                top_alternatives=max(1, min(20, len(option_codes))),
                max_output_tokens=2,
                temperature=0.0,
                GenerateContentConfig=GenerateContentConfig,
            )

            tokens = extract_logprobs(response)
            dist_dict = build_option_distribution_from_step(tokens, option_codes)
            if dist_dict is None:
                # Fallback: uniform over available codes
                n = float(len(option_codes))
                dist_dict = {c: 1.0 / n for c in option_codes}

            # qd2: model distribution as Series, ensure indices align with codes present in qd1
            qd2 = pd.Series(dist_dict, dtype=float)

            score = compare_distributions(qd1, qd2, num_options=len(responses_df[question_key].unique()))
            print(f"Question {question_key} score: {score}")
            scores.append(score)

        print('=' * 20)
        print('Average Representativeness:', sum(scores) / len(scores) if scores else 0)
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


