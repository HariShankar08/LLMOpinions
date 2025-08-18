#!/bin/bash

# Script to run all evaluate_model.py scripts for all available country-language combinations
# This script automatically detects JSON files and runs the corresponding evaluations

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create logs directory
LOGS_DIR="evaluation_logs"
mkdir -p "$LOGS_DIR"

# Get timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOGS_DIR/evaluation_run_${TIMESTAMP}.log"

echo -e "${BLUE}Starting evaluation runs for all available country-language combinations...${NC}"
echo -e "${BLUE}Main log file: $MAIN_LOG${NC}"
echo ""

# Function to log messages to both console and main log
log_message() {
    local message="$1"
    echo -e "$message" | tee -a "$MAIN_LOG"
}

# Function to run evaluation for a specific region
run_region_evaluations() {
    local region=$1
    local script_name=$2
    local base_dir="Translate/$region"
    
    if [ ! -d "$base_dir" ]; then
        log_message "${YELLOW}Warning: Directory $base_dir not found, skipping...${NC}"
        return
    fi
    
    log_message "${GREEN}Processing region: $region${NC}"
    
    # Find all JSON files in the region directory
    local json_files=$(find "$base_dir" -name "*.json" -type f)
    
    if [ -z "$json_files" ]; then
        log_message "${YELLOW}No JSON files found in $base_dir${NC}"
        return
    fi
    
    # Process each JSON file
    for json_file in $json_files; do
        # Extract filename without path and extension
        local filename=$(basename "$json_file" .json)
        
        # Parse country and language from filename (format: country_language.json)
        local country=$(echo "$filename" | cut -d'_' -f1)
        local language=$(echo "$filename" | cut -d'_' -f2)
        
        # Create individual log file for this evaluation
        local eval_log="$LOGS_DIR/${region}_${country}_${language}_${TIMESTAMP}.log"
        
        log_message "${BLUE}Running evaluation for $country ($language) in $region...${NC}"
        log_message "Individual log: $eval_log"
        
        # Change to the region directory
        cd "$base_dir"
        
        # Run the evaluation script and capture output
        local start_time=$(date +"%Y-%m-%d %H:%M:%S")
        echo "=== Evaluation started at $start_time ===" > "$eval_log"
        echo "Region: $region" >> "$eval_log"
        echo "Country: $country" >> "$eval_log"
        echo "Language: $language" >> "$eval_log"
        echo "Script: $script_name" >> "$eval_log"
        echo "========================================" >> "$eval_log"
        echo "" >> "$eval_log"
        
        if python "$script_name" --country "$country" --language "$language" 2>&1 | tee -a "$eval_log"; then
            local end_time=$(date +"%Y-%m-%d %H:%M:%S")
            echo "" >> "$eval_log"
            echo "=== Evaluation completed successfully at $end_time ===" >> "$eval_log"
            log_message "${GREEN}✓ Successfully completed evaluation for $country ($language) in $region${NC}"
        else
            local end_time=$(date +"%Y-%m-%d %H:%M:%S")
            echo "" >> "$eval_log"
            echo "=== Evaluation FAILED at $end_time ===" >> "$eval_log"
            log_message "${RED}✗ Failed to complete evaluation for $country ($language) in $region${NC}"
        fi
        
        # Return to original directory
        cd - > /dev/null
        
        log_message ""
    done
}

# Function to run evaluation for IND region (special case)
run_ind_evaluations() {
    local base_dir="IND"
    
    if [ ! -d "$base_dir" ]; then
        log_message "${YELLOW}Warning: Directory $base_dir not found, skipping...${NC}"
        return
    fi
    
    log_message "${GREEN}Processing region: IND${NC}"
    
    # Find all JSON files in the IND directory
    local json_files=$(find "$base_dir" -name "*.json" -type f)
    
    if [ -z "$json_files" ]; then
        log_message "${YELLOW}No JSON files found in $base_dir${NC}"
        return
    fi
    
    # Process each JSON file
    for json_file in $json_files; do
        # Extract filename without path and extension
        local filename=$(basename "$json_file" .json)
        
        # Parse country and language from filename (format: ind_language.json)
        local country=$(echo "$filename" | cut -d'_' -f1)
        local language=$(echo "$filename" | cut -d'_' -f2)
        
        # Create individual log file for this evaluation
        local eval_log="$LOGS_DIR/IND_${country}_${language}_${TIMESTAMP}.log"
        
        log_message "${BLUE}Running evaluation for $country ($language)...${NC}"
        log_message "Individual log: $eval_log"
        
        # Change to the IND directory
        cd "$base_dir"
        
        # Run the evaluation script and capture output
        local start_time=$(date +"%Y-%m-%d %H:%M:%S")
        echo "=== Evaluation started at $start_time ===" > "$eval_log"
        echo "Region: IND" >> "$eval_log"
        echo "Country: $country" >> "$eval_log"
        echo "Language: $language" >> "$eval_log"
        echo "Script: evaluate_model.py" >> "$eval_log"
        echo "========================================" >> "$eval_log"
        echo "" >> "$eval_log"
        
        if python "evaluate_model.py" --country "$country" --language "$language" 2>&1 | tee -a "$eval_log"; then
            local end_time=$(date +"%Y-%m-%d %H:%M:%S")
            echo "" >> "$eval_log"
            echo "=== Evaluation completed successfully at $end_time ===" >> "$eval_log"
            log_message "${GREEN}✓ Successfully completed evaluation for $country ($language)${NC}"
        else
            local end_time=$(date +"%Y-%m-%d %H:%M:%S")
            echo "" >> "$eval_log"
            echo "=== Evaluation FAILED at $end_time ===" >> "$eval_log"
            log_message "${RED}✗ Failed to complete evaluation for $country ($language)${NC}"
        fi
        
        # Return to original directory
        cd - > /dev/null
        
        log_message ""
    done
}

# Function to run noCoT evaluations for a specific region
run_nocot_evaluations() {
    local region=$1
    local script_name="evaluate_model_noCoT.py"
    local base_dir="Translate/$region"
    
    if [ ! -d "$base_dir" ]; then
        log_message "${YELLOW}Warning: Directory $base_dir not found, skipping...${NC}"
        return
    fi
    
    log_message "${GREEN}Processing noCoT evaluations for region: $region${NC}"
    
    # Find all JSON files in the region directory
    local json_files=$(find "$base_dir" -name "*.json" -type f)
    
    if [ -z "$json_files" ]; then
        log_message "${YELLOW}No JSON files found in $base_dir${NC}"
        return
    fi
    
    # Process each JSON file
    for json_file in $json_files; do
        # Extract filename without path and extension
        local filename=$(basename "$json_file" .json)
        
        # Parse country and language from filename (format: country_language.json)
        local country=$(echo "$filename" | cut -d'_' -f1)
        local language=$(echo "$filename" | cut -d'_' -f2)
        
        # Create individual log file for this evaluation
        local eval_log="$LOGS_DIR/${region}_${country}_${language}_noCoT_${TIMESTAMP}.log"
        
        log_message "${BLUE}Running noCoT evaluation for $country ($language) in $region...${NC}"
        log_message "Individual log: $eval_log"
        
        # Change to the region directory
        cd "$base_dir"
        
        # Run the noCoT evaluation script and capture output
        local start_time=$(date +"%Y-%m-%d %H:%M:%S")
        echo "=== noCoT Evaluation started at $start_time ===" > "$eval_log"
        echo "Region: $region" >> "$eval_log"
        echo "Country: $country" >> "$eval_log"
        echo "Language: $language" >> "$eval_log"
        echo "Script: $script_name" >> "$eval_log"
        echo "=============================================" >> "$eval_log"
        echo "" >> "$eval_log"
        
        if python "$script_name" --country "$country" --language "$language" 2>&1 | tee -a "$eval_log"; then
            local end_time=$(date +"%Y-%m-%d %H:%M:%S")
            echo "" >> "$eval_log"
            echo "=== noCoT Evaluation completed successfully at $end_time ===" >> "$eval_log"
            log_message "${GREEN}✓ Successfully completed noCoT evaluation for $country ($language) in $region${NC}"
        else
            local end_time=$(date +"%Y-%m-%d %H:%M:%S")
            echo "" >> "$eval_log"
            echo "=== noCoT Evaluation FAILED at $end_time ===" >> "$eval_log"
            log_message "${RED}✗ Failed to complete noCoT evaluation for $country ($language) in $region${NC}"
        fi
        
        # Return to original directory
        cd - > /dev/null
        
        log_message ""
    done
}

# Main execution
log_message "${BLUE}=== Running Standard Evaluations ===${NC}"
log_message ""

# Run standard evaluations for each region
run_region_evaluations "SEA" "evaluate_model.py"
run_region_evaluations "EA" "evaluate_model.py"
run_ind_evaluations

log_message "${BLUE}=== Running noCoT Evaluations ===${NC}"
log_message ""

# Run noCoT evaluations for each region
run_nocot_evaluations "SEA"
run_nocot_evaluations "EA"
run_nocot_evaluations "IND"

log_message "${GREEN}All evaluations completed!${NC}"

# Summary
log_message ""
log_message "${BLUE}=== Summary ===${NC}"
log_message "Check the output above for any errors or warnings."
log_message "Results should be available in the respective directories."
log_message "Individual log files are saved in: $LOGS_DIR"
log_message "Main log file: $MAIN_LOG"

# Create a summary file
SUMMARY_FILE="$LOGS_DIR/summary_${TIMESTAMP}.txt"
echo "Evaluation Run Summary - $TIMESTAMP" > "$SUMMARY_FILE"
echo "=====================================" >> "$SUMMARY_FILE"
echo "Main log: $MAIN_LOG" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Individual log files:" >> "$SUMMARY_FILE"
ls -la "$LOGS_DIR"/*"${TIMESTAMP}"*.log | grep -v "evaluation_run_" | while read line; do
    echo "  $line" >> "$SUMMARY_FILE"
done
echo "" >> "$SUMMARY_FILE"
echo "Total evaluations run: $(ls "$LOGS_DIR"/*"${TIMESTAMP}"*.log | wc -l)" >> "$SUMMARY_FILE"

log_message "Summary file created: $SUMMARY_FILE"
