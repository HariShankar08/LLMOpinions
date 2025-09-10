#!/bin/bash

# Script to run all evaluate_model_openrouter.py scripts for all available country-language combinations
# This script automatically detects JSON files and runs the corresponding evaluations with OpenRouter models

set -e  # Exit on any error

ORIGINAL_DIR=$(pwd)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Pick a Python interpreter (allow override via $PYTHON_BIN)
: "${PYTHON_BIN:=}"
if [ -z "$PYTHON_BIN" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        echo -e "${RED}Python is not installed or not in PATH. Install Python 3 and retry.${NC}"
        exit 1
    fi
fi

LOGS_DIR="$ORIGINAL_DIR/evaluation_logs_openrouter"
mkdir -p "$LOGS_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOGS_DIR/evaluation_run_openrouter_${TIMESTAMP}.log"

log_message() {
    local message="$1"
    echo -e "$message" | tee -a "$MAIN_LOG"
}


run_region_evaluations_openrouter() {
    local region=$1
    local base_dir="$ORIGINAL_DIR/Translate/$region"
    local model_name=$2
    local cot_flag=$3
    local model_short=$4
    shift 4
    local extra_args="$@"

    if [ ! -d "$base_dir" ]; then
        log_message "${YELLOW}Warning: Directory $base_dir not found, skipping...${NC}"
        return
    fi

    log_message "${GREEN}Processing region: $region (OpenRouter)${NC}"

    local json_files=$(find "$base_dir" -name "*.json" -type f)

    if [ -z "$json_files" ]; then
        log_message "${YELLOW}No JSON files found in $base_dir${NC}"
        return
    fi


    for json_file in $json_files; do
        local filename=$(basename "$json_file" .json)
        local country=$(echo "$filename" | cut -d'_' -f1)
        local language=$(echo "$filename" | cut -d'_' -f2)
        local eval_log="$LOGS_DIR/${region}_${country}_${language}_${model_short}_openrouter_${TIMESTAMP}.log"

        log_message "${BLUE}Running OpenRouter evaluation for $country ($language) in $region...${NC}"
        log_message "Individual log: $eval_log"

        cd "$base_dir"
        local start_time=$(date +"%Y-%m-%d %H:%M:%S")
        echo "=== OpenRouter Evaluation started at $start_time ===" > "$eval_log"
        echo "Region: $region" >> "$eval_log"
        echo "Country: $country" >> "$eval_log"
        echo "Language: $language" >> "$eval_log"
        echo "Script: evaluate_model_openrouter.py" >> "$eval_log"
        echo "Model: $model_name" >> "$eval_log"
        echo "========================================" >> "$eval_log"
        echo "" >> "$eval_log"

    if "$PYTHON_BIN" "evaluate_model_openrouter.py" --country "$country" --language "$language" --model "$model_name" $cot_flag $extra_args 2>&1 | tee -a "$eval_log"; then
            local end_time=$(date +"%Y-%m-%d %H:%M:%S")
            echo "" >> "$eval_log"
            echo "=== OpenRouter Evaluation completed successfully at $end_time ===" >> "$eval_log"
            log_message "${GREEN}✓ Successfully completed OpenRouter evaluation for $country ($language) in $region${NC}"
        else
            local end_time=$(date +"%Y-%m-%d %H:%M:%S")
            echo "" >> "$eval_log"
            echo "=== OpenRouter Evaluation FAILED at $end_time ===" >> "$eval_log"
            log_message "${RED}✗ Failed to complete OpenRouter evaluation for $country ($language) in $region${NC}"
        fi
        cd "$ORIGINAL_DIR"
        log_message ""
    done
}

# Main execution for OpenRouter
log_message "${BLUE}=== Running OpenRouter Evaluations ===${NC}"
log_message ""


# Example usage: pass model name, model_short, and --cot or blank for no CoT
default_model="openai/gpt-3.5-turbo"
default_model_short="gpt35"
model_name="${1:-$default_model}"
model_short="${2:-$default_model_short}"
cot_flag="${3:-}" # pass --cot if you want CoT
shift 3 || true
extra_args="$@"

run_region_evaluations_openrouter "SEA" "$model_name" "$cot_flag" "$model_short" $extra_args
run_region_evaluations_openrouter "EA" "$model_name" "$cot_flag" "$model_short" $extra_args
run_region_evaluations_openrouter "IND" "$model_name" "$cot_flag" "$model_short" $extra_args

log_message "${GREEN}All OpenRouter evaluations completed!${NC}"

SUMMARY_FILE="$LOGS_DIR/summary_openrouter_${TIMESTAMP}.txt"
echo "OpenRouter Evaluation Run Summary - $TIMESTAMP" > "$SUMMARY_FILE"
echo "=====================================" >> "$SUMMARY_FILE"
echo "Main log: $MAIN_LOG" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Individual log files:" >> "$SUMMARY_FILE"
ls -la "$LOGS_DIR"/*"${TIMESTAMP}"*.log | grep -v "evaluation_run_" | while read line; do
    echo "  $line" >> "$SUMMARY_FILE"
done
echo "" >> "$SUMMARY_FILE"
echo "Total OpenRouter evaluations run: $(ls "$LOGS_DIR"/*"${TIMESTAMP}"*.log | wc -l)" >> "$SUMMARY_FILE"

log_message "Summary file created: $SUMMARY_FILE"
