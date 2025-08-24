#!/usr/bin/env python3
"""
Evaluation Log Parser

This script parses evaluation log files and generates a summary table
showing average representativeness scores for each region-country-language combination.
"""

import os
import re
import glob
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def parse_log_filename(filename: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Parse a log filename to extract region, country, language, and evaluation type.
    
    Expected format: {REGION}_{COUNTRY}_{LANGUAGE}_{EVAL_TYPE}_{TIMESTAMP}.log
    
    Returns:
        Tuple of (region, country, language, eval_type) or None if parsing fails
    """
    # Remove .log extension and split by underscore
    base_name = os.path.basename(filename).replace('.log', '')
    parts = base_name.split('_')
    
    if len(parts) < 4:
        return None
    
    # The first 4 parts should be region, country, language, eval_type
    region = parts[0]
    country = parts[1]
    language = parts[2]
    eval_type = parts[3]
    
    return (region, country, language, eval_type)


def extract_representativeness_score(log_content: str) -> Optional[float]:
    """
    Extract the average representativeness score from log content.
    
    Args:
        log_content: The content of the log file
        
    Returns:
        The representativeness score as float, or None if not found
    """
    # Look for the pattern "Average Representativeness: [number]"
    pattern = r'Average Representativeness:\s*([0-9.]+)'
    match = re.search(pattern, log_content)
    
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    
    return None


def parse_log_file(filepath: str) -> Optional[Dict]:
    """
    Parse a single log file and extract relevant information.
    
    Args:
        filepath: Path to the log file
        
    Returns:
        Dictionary with parsed information or None if parsing fails
    """
    try:
        # Parse filename
        filename_info = parse_log_filename(filepath)
        if not filename_info:
            print(f"Warning: Could not parse filename: {filepath}")
            return None
        
        region, country, language, eval_type = filename_info
        
        # Read log content
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract representativeness score
        score = extract_representativeness_score(content)
        if score is None:
            print(f"Warning: Could not find representativeness score in: {filepath}")
            return None
        
        return {
            'region': region,
            'country': country,
            'language': language,
            'eval_type': eval_type,
            'representativeness': score,
            'log_file': os.path.basename(filepath)
        }
        
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None


def parse_all_logs(logs_directory: str) -> pd.DataFrame:
    """
    Parse all log files in the specified directory.
    
    Args:
        logs_directory: Path to the directory containing log files
        
    Returns:
        DataFrame with parsed results
    """
    log_files = glob.glob(os.path.join(logs_directory, "*.log"))
    
    results = []
    for log_file in log_files:
        result = parse_log_file(log_file)
        if result:
            results.append(result)
    
    if not results:
        print("No valid log files found!")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    return df


def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table grouped by region, country, and language.
    
    Args:
        df: DataFrame with parsed log data
        
    Returns:
        Summary DataFrame
    """
    if df.empty:
        return df
    
    # Group by region, country, language and calculate statistics
    summary = df.groupby(['region', 'country', 'language']).agg({
        'representativeness': ['mean', 'std', 'count'],
        'eval_type': lambda x: ', '.join(sorted(set(x))),
        'log_file': lambda x: ', '.join(sorted(set(x)))
    }).round(4)
    
    # Flatten column names
    summary.columns = ['avg_representativeness', 'std_representativeness', 'num_evaluations', 'eval_types', 'log_files']
    
    # Reset index to make region, country, language regular columns
    summary = summary.reset_index()
    
    return summary


def print_summary_table(summary_df: pd.DataFrame):
    """
    Print a nicely formatted summary table.
    
    Args:
        summary_df: Summary DataFrame to print
    """
    if summary_df.empty:
        print("No data to display.")
        return
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS SUMMARY")
    print("="*80)
    
    # Print overall statistics
    print(f"\nTotal evaluations processed: {len(summary_df)}")
    print(f"Overall average representativeness: {summary_df['avg_representativeness'].mean()}")
    print(f"Overall standard deviation: {summary_df['avg_representativeness'].std()}")
    
    # Print by region
    print("\n" + "-"*80)
    print("RESULTS BY REGION")
    print("-"*80)
    
    for region in sorted(summary_df['region'].unique()):
        region_data = summary_df[summary_df['region'] == region]
        print(f"\n{region.upper()} REGION:")
        print(f"  Average representativeness: {region_data['avg_representativeness'].mean()}")
        print(f"  Number of evaluations: {len(region_data)}")
        
        # Print individual results for this region
        for _, row in region_data.sort_values(['country', 'language']).iterrows():
            print(f"    {row['country']} ({row['language']}): {row['avg_representativeness']}")
    
    # Print detailed table
    print("\n" + "-"*80)
    print("DETAILED RESULTS TABLE")
    print("-"*80)
    
    # Format the table for display
    display_df = summary_df.copy()
    display_df['avg_representativeness'] = display_df['avg_representativeness'].apply(lambda x: f"{x}")
    display_df['std_representativeness'] = display_df['std_representativeness'].apply(lambda x: f"{x}")
    
    print(display_df[['region', 'country', 'language', 'avg_representativeness', 'std_representativeness', 'num_evaluations']].to_string(index=False))


def save_summary_to_csv(summary_df: pd.DataFrame, output_file: str = "evaluation_summary.csv"):
    """
    Save the summary table to a CSV file.
    
    Args:
        summary_df: Summary DataFrame to save
        output_file: Output CSV filename
    """
    if not summary_df.empty:
        summary_df.to_csv(output_file, index=False)
        print(f"\nSummary saved to: {output_file}")


def main():
    """Main function to run the log parser."""
    # Default logs directory
    logs_dir = "evaluation_logs"
    
    # Check if logs directory exists
    if not os.path.exists(logs_dir):
        print(f"Error: Logs directory '{logs_dir}' not found!")
        print("Please run this script from the directory containing the evaluation_logs folder.")
        return
    
    print(f"Parsing evaluation logs from: {logs_dir}")
    
    # Parse all logs
    df = parse_all_logs(logs_dir)
    
    if df.empty:
        print("No valid log files found!")
        return
    
    print(f"Successfully parsed {len(df)} log files.")
    
    # Create summary table
    summary_df = create_summary_table(df)
    
    # Print summary
    print_summary_table(summary_df)
    
    # Save to CSV
    save_summary_to_csv(summary_df)
    
    # Additional analysis
    print("\n" + "-"*80)
    print("ADDITIONAL ANALYSIS")
    print("-"*80)
    
    # Best and worst performing combinations
    best_idx = summary_df['avg_representativeness'].idxmax()
    worst_idx = summary_df['avg_representativeness'].idxmin()
    
    best_row = summary_df.loc[best_idx]
    worst_row = summary_df.loc[worst_idx]
    
    print(f"\nBest performing combination:")
    print(f"  {best_row['region']} - {best_row['country']} ({best_row['language']}): {best_row['avg_representativeness']}")
    
    print(f"\nWorst performing combination:")
    print(f"  {worst_row['region']} - {worst_row['country']} ({worst_row['language']}): {worst_row['avg_representativeness']}")
    
    # Performance by evaluation type
    if len(df['eval_type'].unique()) > 1:
        print(f"\nPerformance by evaluation type:")
        for eval_type in sorted(df['eval_type'].unique()):
            eval_data = df[df['eval_type'] == eval_type]
            print(f"  {eval_type}: {eval_data['representativeness'].mean()} (n={len(eval_data)})")


if __name__ == "__main__":
    main()
