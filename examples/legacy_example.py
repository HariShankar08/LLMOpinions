#!/usr/bin/env python3
# Example usage of OpenRouter with legacy evaluation approach

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluate_model_openrouter import ModelEvaluator

def main():
    # Example: Evaluate GPT-4 for Hong Kong in EA region
    evaluator = ModelEvaluator(
        model_name="gpt-4o",  # Short name for OpenRouter model
        use_openrouter=True
    )
    
    print("Example evaluation setup complete!")
    print("To run full evaluation:")
    print("python evaluate_model_openrouter.py --model gpt-4o --country HKG --region EA --use-openrouter")

if __name__ == "__main__":
    main()
