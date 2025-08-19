#!/usr/bin/env python3
"""
Setup script for OpenRouter integration.
This script helps configure the environment and test the OpenRouter connection.
"""

import os
import sys
import json
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ['requests', 'pandas', 'numpy', 'scipy', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements_openrouter.txt")
        return False
    
    print("✓ All required dependencies are installed")
    return True

def setup_api_key():
    """Setup OpenRouter API key."""
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    if not api_key:
        print("\n" + "="*50)
        print("OPENROUTER API KEY SETUP")
        print("="*50)
        print("You need an OpenRouter API key to use OpenRouter models.")
        print("1. Go to https://openrouter.ai/")
        print("2. Sign up for an account")
        print("3. Get your API key from the dashboard")
        print("4. Set it as an environment variable:")
        print("   export OPENROUTER_API_KEY='your-api-key-here'")
        print("\nOr create a .env file in this directory with:")
        print("   OPENROUTER_API_KEY=your-api-key-here")
        
        # Offer to create .env file
        create_env = input("\nWould you like to create a .env file now? (y/n): ").lower().strip()
        if create_env in ['y', 'yes']:
            api_key = input("Enter your OpenRouter API key: ").strip()
            if api_key:
                env_path = Path('.env')
                with open(env_path, 'w') as f:
                    f.write(f"OPENROUTER_API_KEY={api_key}\n")
                print(f"✓ Created .env file at {env_path.absolute()}")
                os.environ['OPENROUTER_API_KEY'] = api_key
                return api_key
        
        print("\nSkipping API key setup. You can set it later.")
        return None
    else:
        print(f"✓ OpenRouter API key found: {api_key[:10]}...")
        return api_key

def test_openrouter_connection(api_key):
    """Test connection to OpenRouter API."""
    if not api_key:
        print("⚠ Skipping connection test (no API key)")
        return False
    
    try:
        from openrouter_client import OpenRouterClient
        
        print("\nTesting OpenRouter connection...")
        client = OpenRouterClient(api_key)
        
        # Test with a simple completion
        messages = [{"role": "user", "content": "Hello! Please respond with just 'OK' to confirm the connection."}]
        response = client.get_text_response(messages, "openai/gpt-3.5-turbo", max_tokens=10)
        
        print(f"✓ OpenRouter connection successful!")
        print(f"  Test response: {response[:50]}...")
        return True
        
    except Exception as e:
        print(f"✗ OpenRouter connection failed: {e}")
        return False

def show_available_models():
    """Show available OpenRouter models."""
    try:
        from openrouter_client import OPENROUTER_MODELS, list_available_models
        
        print("\n" + "="*50)
        print("AVAILABLE OPENROUTER MODELS")
        print("="*50)
        list_available_models()
        
    except ImportError:
        print("Could not import OpenRouter client to show models")

def create_example_scripts():
    """Create example scripts for testing."""
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # Example script for legacy approach
    legacy_example = """#!/usr/bin/env python3
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
"""
    
    with open(examples_dir / "legacy_example.py", "w") as f:
        f.write(legacy_example)
    
    # Example script for Translate approach
    translate_example = """#!/usr/bin/env python3
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
"""
    
    with open(examples_dir / "translate_example.py", "w") as f:
        f.write(translate_example)
    
    print(f"✓ Created example scripts in {examples_dir.absolute()}/")

def main():
    """Main setup function."""
    print("OpenRouter Integration Setup")
    print("="*30)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup API key
    api_key = setup_api_key()
    
    # Test connection
    connection_ok = test_openrouter_connection(api_key)
    
    # Show available models
    show_available_models()
    
    # Create example scripts
    create_example_scripts()
    
    print("\n" + "="*50)
    print("SETUP SUMMARY")
    print("="*50)
    print(f"✓ Dependencies: OK")
    print(f"{'✓' if api_key else '⚠'} API Key: {'Configured' if api_key else 'Not configured'}")
    print(f"{'✓' if connection_ok else '⚠'} Connection: {'OK' if connection_ok else 'Not tested'}")
    print(f"✓ Example scripts: Created")
    
    if api_key and connection_ok:
        print("\n🎉 OpenRouter integration is ready to use!")
        print("\nNext steps:")
        print("1. Try the example scripts in the examples/ directory")
        print("2. Use evaluate_model_openrouter.py for legacy evaluations")
        print("3. Use Translate/evaluate_model_openrouter.py for multilingual evaluations")
    else:
        print("\n⚠ Setup incomplete. Please configure your API key and test the connection.")
    
    print("\nFor more information, see the OpenRouter documentation:")
    print("https://openrouter.ai/docs")

if __name__ == "__main__":
    main()
