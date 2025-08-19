#!/usr/bin/env python3
# Example usage of OpenRouter with Translate evaluation approach

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Translate.evaluate_model_openrouter import TranslateModelEvaluator

def main():
    # Example: Evaluate Claude for Singapore in English
    evaluator = TranslateModelEvaluator(
        model_name="claude-3.5-sonnet",  # Short name for OpenRouter model
        use_openrouter=True,
        language="en",
        country="sg"
    )
    
    print("Example evaluation setup complete!")
    print("To run full evaluation:")
    print("cd Translate && python evaluate_model_openrouter.py --country sg --language en --model claude-3.5-sonnet --use-openrouter")

if __name__ == "__main__":
    main()
