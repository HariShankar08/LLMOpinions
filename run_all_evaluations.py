#!/usr/bin/env python3

import os
import subprocess
import sys
from datetime import datetime
from glob import glob
import argparse
import csv
import json


# --- Configuration ---

# Store the original directory where the script is run
ORIGINAL_DIR = os.getcwd()

# Colors for console output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m' # No Color

# --- Setup Logging ---

# Create a directory for logs
LOGS_DIR = os.path.join(ORIGINAL_DIR, "evaluation_logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Get a unique timestamp for this entire run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MAIN_LOG_PATH = os.path.join(LOGS_DIR, f"evaluation_run_{TIMESTAMP}.log")
AVERAGES_CSV_PATH = os.path.join(LOGS_DIR, f"averages_{TIMESTAMP}.csv")

# Initialize CSV with header
if not os.path.exists(AVERAGES_CSV_PATH):
    with open(AVERAGES_CSV_PATH, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "region", "country", "language", "script", "suffix", "average_representativeness"]) 

def log_message(message, color=Colors.NC):
    """
    Logs a message to both the console (with color) and the main log file.
    """
    # Print to console with color
    print(f"{color}{message}{Colors.NC}")
    
    # Write to main log file without color codes
    with open(MAIN_LOG_PATH, 'a') as f:
        f.write(message + '\n')

def run_evaluation(region, script_name, log_suffix="", model_override=None, steering_override=False):
    """
    Finds all JSON files in a region's directory and runs the specified
    evaluation script for each, capturing logs.
    
    Args:
        region (str): The region to process (e.g., "SEA", "EA", "IND").
        script_name (str): The name of the Python evaluation script to run.
        log_suffix (str): An optional suffix for the individual log file names.
        steering_override (bool): If True, add the '--steering' flag to the evaluation command.
    """
    log_message(f"Processing region: {region}", Colors.GREEN)

    base_dir = os.path.join(ORIGINAL_DIR, "Translate", region)
    
    if not os.path.isdir(base_dir):
        log_message(f"Warning: Directory {base_dir} not found, skipping...", Colors.YELLOW)
        return

    # Find all JSON files in the directory
    json_files = glob(os.path.join(base_dir, "*.json"))

    if not json_files:
        log_message(f"No JSON files found in {base_dir}", Colors.YELLOW)
        return

    # Process each JSON file found
    for json_file in json_files:
        filename = os.path.basename(json_file).replace('.json', '')
        
        try:
            # Assumes filename format: country_language or ind_language
            country, language = filename.split('_', 1)
        except ValueError:
            log_message(f"Warning: Could not parse country/language from '{filename}', skipping...", Colors.YELLOW)
            continue

        # Create a unique log file for this specific evaluation
        eval_log_name = f"{region}_{country}_{language}{log_suffix}_{TIMESTAMP}.log"
        eval_log_path = os.path.join(LOGS_DIR, eval_log_name)

        log_message(f"Running evaluation for {country} ({language}) in {region}...", Colors.BLUE)
        log_message(f"Individual log: {eval_log_path}")
        
        
        start_time = datetime.now()
        
        # Write header to the individual log file
        with open(eval_log_path, 'w') as f:
            f.write(f"=== Evaluation started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(f"Region: {region}\n")
            f.write(f"Country: {country}\n")
            f.write(f"Language: {language}\n")
            f.write(f"Script: {script_name}\n")
            f.write(f"Suffix: {log_suffix or 'N/A'}\n")
            f.write("==================================================\n\n")

        try:
            # Change to the script's directory to run it
            os.chdir(base_dir)
            
            # Prepare the command
            command = [
                sys.executable,  # Use the same python interpreter that is running this script
                script_name,
                "--country", country,
                "--language", language
            ]

            # If a model override is provided, pass it through to the evaluation script
            if model_override:
                command.extend(["--model", model_override])
            # If steering override is enabled, add the flag
            if steering_override:
                command.append("--steering")
            
            # Execute the command and capture output in real-time
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            with open(eval_log_path, 'a') as f:
                for line in iter(process.stdout.readline, ''):
                    sys.stdout.write(line) # Show output on console
                    f.write(line) # Write output to log file

                    # Capture and log average representativeness to CSV
                    if "Average Representativeness:" in line:
                        try:
                            avg_str = line.strip().split(":", 1)[1].strip()
                            average = float(avg_str)
                            with open(AVERAGES_CSV_PATH, 'a', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerow([
                                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    region,
                                    country,
                                    language,
                                    script_name,
                                    (log_suffix or ""),
                                    average
                                ])
                        except Exception:
                            # Ignore parsing errors and continue
                            pass
            
            process.stdout.close()
            return_code = process.wait()
            
            end_time = datetime.now()

            # Check if the process succeeded or failed
            if return_code == 0:
                log_message(f"✓ Successfully completed evaluation for {country} ({language}) in {region}", Colors.GREEN)
                footer_message = f"=== Evaluation completed successfully at {end_time.strftime('%Y-%m-%d %H:%M:%S')} ==="
            else:
                log_message(f"✗ Failed to complete evaluation for {country} ({language}) in {region} (Exit Code: {return_code})", Colors.RED)
                footer_message = f"=== Evaluation FAILED at {end_time.strftime('%Y-%m-%d %H:%M:%S')} ==="

            # Write footer to the individual log file
            with open(eval_log_path, 'a') as f:
                f.write("\n" + footer_message + "\n")

        except FileNotFoundError:
            log_message(f"✗ Error: Script '{script_name}' not found in {base_dir}", Colors.RED)
        except Exception as e:
            log_message(f"✗ An unexpected error occurred: {e}", Colors.RED)
        finally:
            # IMPORTANT: Always change back to the original directory
            os.chdir(ORIGINAL_DIR)
            log_message("") # Add a blank line for readability

def run_logprobs_full(region):
    """Run the Gemini logprobs evaluator in per-question flow (noCoT parity)."""
    log_message(f"Processing Gemini Logprobs (full) for region: {region}", Colors.GREEN)

    base_dir = os.path.join(ORIGINAL_DIR, "Translate", region)
    if not os.path.isdir(base_dir):
        log_message(f"Warning: Directory {base_dir} not found, skipping...", Colors.YELLOW)
        return

    json_files = glob(os.path.join(base_dir, "*.json"))
    if not json_files:
        log_message(f"No JSON files found in {base_dir}", Colors.YELLOW)
        return

    for json_file in json_files:
        filename = os.path.basename(json_file).replace('.json', '')
        try:
            country, language = filename.split('_', 1)
        except ValueError:
            log_message(f"Warning: Could not parse country/language from '{filename}', skipping...", Colors.YELLOW)
            continue

        # Prepare log
        eval_log_name = f"{region}_{country}_{language}_gemini_logprobs_{TIMESTAMP}.log"
        eval_log_path = os.path.join(LOGS_DIR, eval_log_name)
        log_message(f"Running Gemini logprobs (full) for {country} ({language}) in {region}...", Colors.BLUE)
        log_message(f"Individual log: {eval_log_path}")

        # Execute the evaluator
        start_time = datetime.now()
        with open(eval_log_path, 'w', encoding='utf-8') as f:
            f.write(f"=== Gemini Logprobs (full) started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(f"Region: {region}\n")
            f.write(f"Country: {country}\n")
            f.write(f"Language: {language}\n")
            f.write(f"Script: evaluate_gemini_logprobs.py\n")
            f.write("==================================================\n\n")

        try:
            # Run from repo root where the script exists
            os.chdir(ORIGINAL_DIR)
            command = [
                sys.executable,
                "evaluate_gemini_logprobs.py",
                "--region", region,
                "--country", country,
                "--language", language
            ]

            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            with open(eval_log_path, 'a', encoding='utf-8') as f:
                for line in iter(process.stdout.readline, ''):
                    sys.stdout.write(line)
                    f.write(line)
            process.stdout.close()
            return_code = process.wait()

            end_time = datetime.now()
            footer_message = (
                f"=== Gemini Logprobs (full) completed successfully at {end_time.strftime('%Y-%m-%d %H:%M:%S')} ==="
                if return_code == 0 else
                f"=== Gemini Logprobs (full) FAILED at {end_time.strftime('%Y-%m-%d %H:%M:%S')} ==="
            )
            with open(eval_log_path, 'a', encoding='utf-8') as f:
                f.write("\n" + footer_message + "\n")
        except Exception as e:
            log_message(f"✗ An unexpected error occurred in logprobs run: {e}", Colors.RED)
        finally:
            os.chdir(ORIGINAL_DIR)
            log_message("")

def create_summary():
    """Creates a summary file listing all logs generated during the run."""
    summary_file_path = os.path.join(LOGS_DIR, f"summary_{TIMESTAMP}.txt")
    
    log_files = sorted(glob(os.path.join(LOGS_DIR, f"*_{TIMESTAMP}.log")))
    individual_logs = [f for f in log_files if not os.path.basename(f).startswith("evaluation_run_")]
    
    with open(summary_file_path, 'w') as f:
        f.write(f"Evaluation Run Summary - {TIMESTAMP}\n")
        f.write("=====================================\n")
        f.write(f"Main log: {MAIN_LOG_PATH}\n\n")
        f.write("Individual log files:\n")
        for log in individual_logs:
            f.write(f"  - {log}\n")
        f.write("\n")
        f.write(f"Total evaluations run: {len(individual_logs)}\n")
        
    log_message(f"Summary file created: {summary_file_path}")


def main():
    """Main execution function."""
    log_message("Starting evaluation runs for all available country-language combinations...", Colors.BLUE)
    log_message(f"Main log file: {MAIN_LOG_PATH}\n", Colors.BLUE)

    # Parse optional CLI arguments for this runner
    parser = argparse.ArgumentParser(description="Run all evaluations across regions")
    parser.add_argument('--model', type=str, default=None, help='Override model for all evaluations')
    parser.add_argument('--steering', action='store_true', help='Enable steering flag for all evaluations')
    args = parser.parse_args()

    model_override = args.model
    steering_override = args.steering

    # --- Run Standard Evaluations ---
    log_message("=== Running Standard Evaluations ===", Colors.BLUE)
    run_evaluation("SEA", "evaluate_model.py", model_override=model_override, steering_override=steering_override)
    run_evaluation("EA", "evaluate_model.py", model_override=model_override, steering_override=steering_override)
    run_evaluation("IND", "evaluate_model.py", model_override=model_override, steering_override=steering_override)

    # --- Run noCoT Evaluations ---
    log_message("=== Running noCoT Evaluations ===", Colors.BLUE)
    run_evaluation("SEA", "evaluate_model_noCoT.py", log_suffix="_noCoT", model_override=model_override, steering_override=steering_override)
    run_evaluation("EA", "evaluate_model_noCoT.py", log_suffix="_noCoT", model_override=model_override, steering_override=steering_override)
    run_evaluation("IND", "evaluate_model_noCoT.py", log_suffix="_noCoT", model_override=model_override, steering_override=steering_override)

    # --- Run Gemini Evaluations ---
    log_message("=== Running Gemini Evaluations ===", Colors.BLUE)
    run_evaluation("SEA", "evaluate_model_gemini.py", log_suffix="_gemini", model_override=model_override, steering_override=steering_override)
    run_evaluation("EA", "evaluate_model_gemini.py", log_suffix="_gemini", model_override=model_override, steering_override=steering_override)
    run_evaluation("IND", "evaluate_model_gemini.py", log_suffix="_gemini", model_override=model_override, steering_override=steering_override)

    # --- Run Gemini Logprobs Demo (optional, minimal integration) ---
    log_message("=== Running Gemini Logprobs (Full) ===", Colors.BLUE)
    run_logprobs_full("SEA")
    run_logprobs_full("EA")
    run_logprobs_full("IND")

    log_message("All evaluations completed!", Colors.GREEN)
    
    # --- Create Final Summary ---
    log_message("\n=== Summary ===", Colors.BLUE)
    log_message("Check the output above for any errors or warnings.")
    log_message("Results should be available in the respective directories.")
    log_message(f"Individual log files are saved in: {LOGS_DIR}")
    log_message(f"Main log file: {MAIN_LOG_PATH}")
    create_summary()


if __name__ == "__main__":
    main()
