import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def sum_string_lengths(obj) -> int:
	"""
	Recursively sum the lengths of all string values within a JSON-like object.
	"""
	if isinstance(obj, str):
		return len(obj)
	if isinstance(obj, dict):
		return sum(sum_string_lengths(v) for v in obj.values())
	if isinstance(obj, list):
		return sum(sum_string_lengths(v) for v in obj)
	return 0


def scan_region_for_char_counts(translate_root: Path, region: str) -> Tuple[int, int]:
	"""
	Scan a region directory under Translate/ and return (total_chars, file_count).
	"""
	region_dir = translate_root / region
	if not region_dir.exists():
		raise FileNotFoundError(f"Region directory not found: {region_dir}")

	files: List[Path] = list(region_dir.rglob("*.json"))
	total_chars = 0
	for fp in files:
		try:
			with open(fp, "r", encoding="utf-8") as f:
				data = json.load(f)
			total_chars += sum_string_lengths(data)
		except Exception as e:
			print(f"WARN: Failed to parse {fp}: {e}")

	return total_chars, len(files)


def estimate_tokens_from_chars(total_chars: int) -> int:
	"""
	Approximate tokens as characters/4. This is a rough, model-agnostic heuristic.
	"""
	return int(round(total_chars / 4))


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Estimate total tokens needed for inference across Translate JSONs.")
	parser.add_argument(
		"--translate-root",
		type=str,
		default=str(Path(__file__).parent / "Translate"),
		help="Path to the Translate directory (default: ./Translate)",
	)
	parser.add_argument(
		"--regions",
		nargs="+",
		default=["EA", "SEA", "IND"],
		help="Regions to include (default: EA SEA IND)",
	)
	parser.add_argument(
		"--csv",
		type=str,
		default=None,
		help="Optional path to write a CSV summary (Region,Files,TotalChars,ApproxTokens)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	translate_root = Path(args.translate_root)
	regions: List[str] = args.regions

	print("Region,Files,TotalChars,ApproxTokens(chars/4)")
	rows: List[Tuple[str, int, int, int]] = []
	overall_chars = 0

	for region in regions:
		total_chars, file_count = scan_region_for_char_counts(translate_root, region)
		approx_tokens = estimate_tokens_from_chars(total_chars)
		rows.append((region, file_count, total_chars, approx_tokens))
		overall_chars += total_chars
		print(f"{region},{file_count},{total_chars},{approx_tokens}")

	overall_tokens = estimate_tokens_from_chars(overall_chars)
	print(f"OVERALL,-,{overall_chars},{overall_tokens}")

	if args.csv:
		csv_path = Path(args.csv)
		csv_path.parent.mkdir(parents=True, exist_ok=True)
		try:
			with open(csv_path, "w", encoding="utf-8") as f:
				f.write("Region,Files,TotalChars,ApproxTokens\n")
				for region, file_count, total_chars, approx_tokens in rows:
					f.write(f"{region},{file_count},{total_chars},{approx_tokens}\n")
				f.write(f"OVERALL,-,{overall_chars},{overall_tokens}\n")
			print(f"Wrote CSV to {csv_path}")
		except Exception as e:
			print(f"WARN: Failed to write CSV {csv_path}: {e}")


if __name__ == "__main__":
	main()


