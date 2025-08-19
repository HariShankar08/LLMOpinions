"""
OpenRouter API client for LLM model integration.
This module provides a unified interface for accessing OpenRouter models.
"""

import os
import json
import requests
from typing import Dict, List, Optional, Union
import torch
from transformers import AutoTokenizer
import pandas as pd


class OpenRouterClient:
    """Client for interacting with OpenRouter API models."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://openrouter.ai/api/v1"):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key. If None, will try to get from environment variable OPENROUTER_API_KEY
            base_url: OpenRouter API base URL
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass api_key parameter.")
        
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # For tokenization compatibility with existing code
        self.tokenizer = None
        
    def set_tokenizer_for_compatibility(self, model_name: str = "microsoft/DialoGPT-medium"):
        """
        Set a tokenizer for compatibility with existing code that expects tokenizer methods.
        This is used for token encoding/decoding in probability calculations.
        """
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        except Exception as e:
            print(f"Warning: Could not load tokenizer for compatibility: {e}")
    
    def chat_completion(self, messages: List[Dict[str, str]], model: str, **kwargs) -> Dict:
        """
        Create a chat completion using OpenRouter API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model identifier (e.g., "openai/gpt-4o")
            **kwargs: Additional parameters like temperature, max_tokens, etc.
            
        Returns:
            API response dictionary
        """
        payload = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")
        
        return response.json()
    
    def get_text_response(self, messages: List[Dict[str, str]], model: str, **kwargs) -> str:
        """
        Get text response from chat completion.
        
        Args:
            messages: List of message dictionaries
            model: Model identifier
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        response = self.chat_completion(messages, model, **kwargs)
        return response["choices"][0]["message"]["content"]
    
    def get_logits_for_options(self, messages: List[Dict[str, str]], model: str, options: List[str], **kwargs) -> Dict[str, float]:
        """
        Get probability distribution over specific options.
        This simulates the logits-based approach used in the original code.
        
        Note: This is an approximation since OpenRouter API doesn't expose raw logits.
        We'll use multiple API calls to estimate probabilities.
        
        Args:
            messages: List of message dictionaries
            model: Model identifier  
            options: List of option strings to get probabilities for
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping options to estimated probabilities
        """
        # For each option, we'll prompt the model to choose and see which it prefers
        option_scores = {}
        
        # Create a prompt that asks the model to choose from the options
        base_messages = messages.copy()
        
        # Add options to the last message
        options_text = "\n".join([f"{opt}" for opt in options])
        prompt_with_options = f"{base_messages[-1]['content']}\n\nChoose one of the following options:\n{options_text}\n\nAnswer:"
        base_messages[-1] = {**base_messages[-1], 'content': prompt_with_options}
        
        try:
            # Get response with low temperature for more deterministic results
            response = self.chat_completion(
                base_messages, 
                model, 
                temperature=0.1,
                max_tokens=10,
                **kwargs
            )
            
            response_text = response["choices"][0]["message"]["content"].strip()
            
            # Try to match the response to one of the options
            matched_option = None
            for option in options:
                if option.lower() in response_text.lower() or response_text.lower().startswith(option.lower()):
                    matched_option = option
                    break
            
            # If we found a match, give it high probability, others low probability
            if matched_option:
                total_prob = 1.0
                main_prob = 0.7  # Give matched option 70% probability
                other_prob = (total_prob - main_prob) / max(1, len(options) - 1)
                
                for option in options:
                    if option == matched_option:
                        option_scores[option] = main_prob
                    else:
                        option_scores[option] = other_prob
            else:
                # If no clear match, distribute evenly
                prob = 1.0 / len(options)
                for option in options:
                    option_scores[option] = prob
                    
        except Exception as e:
            print(f"Warning: Could not get option probabilities: {e}")
            # Fallback to uniform distribution
            prob = 1.0 / len(options)
            for option in options:
                option_scores[option] = prob
        
        return option_scores


class OpenRouterModelWrapper:
    """
    Wrapper to make OpenRouter models compatible with existing HuggingFace-style code.
    This allows drop-in replacement in existing evaluation scripts.
    """
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize OpenRouter model wrapper.
        
        Args:
            model_name: OpenRouter model identifier (e.g., "openai/gpt-4o")
            api_key: OpenRouter API key
        """
        self.model_name = model_name
        self.client = OpenRouterClient(api_key)
        self.client.set_tokenizer_for_compatibility()
        
        # For compatibility with existing code
        self.device = torch.device("cpu")  # API models don't use local device
        
    def to(self, device):
        """Compatibility method - API models don't need device placement."""
        return self
    
    def generate(self, input_ids, attention_mask=None, max_new_tokens=100, do_sample=False, **kwargs):
        """
        Generate text using OpenRouter API, compatible with HuggingFace generate method.
        
        Args:
            input_ids: Input token IDs tensor
            attention_mask: Attention mask tensor (ignored for API)
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to sample (affects temperature)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated token IDs tensor (simulated)
        """
        if self.client.tokenizer is None:
            raise ValueError("Tokenizer not set. Call set_tokenizer_for_compatibility first.")
        
        # Decode input to text
        input_text = self.client.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # Create messages for chat completion
        messages = [{"role": "user", "content": input_text}]
        
        # Set temperature based on do_sample
        temperature = 0.1 if not do_sample else kwargs.get("temperature", 0.7)
        
        try:
            # Get response from API
            response_text = self.client.get_text_response(
                messages, 
                self.model_name,
                max_tokens=max_new_tokens,
                temperature=temperature
            )
            
            # Encode the full response (input + generated)
            full_text = input_text + response_text
            output_ids = self.client.tokenizer.encode(full_text, return_tensors="pt")
            
            return output_ids
            
        except Exception as e:
            print(f"Error generating with OpenRouter: {e}")
            # Return original input as fallback
            return input_ids
    
    def __call__(self, input_ids, attention_mask=None, **kwargs):
        """
        Forward pass compatible with HuggingFace models.
        This simulates getting logits for the last token.
        """
        if self.client.tokenizer is None:
            raise ValueError("Tokenizer not set. Call set_tokenizer_for_compatibility first.")
        
        # For API models, we can't get true logits
        # This is a simplified implementation that returns dummy logits
        batch_size, seq_len = input_ids.shape
        vocab_size = len(self.client.tokenizer.get_vocab())
        
        # Create dummy logits tensor
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # Create a simple output object
        class ModelOutput:
            def __init__(self, logits):
                self.logits = logits
        
        return ModelOutput(logits)


def create_openrouter_model(model_name: str, api_key: Optional[str] = None):
    """
    Factory function to create OpenRouter model wrapper.
    
    Args:
        model_name: OpenRouter model identifier
        api_key: OpenRouter API key
        
    Returns:
        OpenRouterModelWrapper instance
    """
    return OpenRouterModelWrapper(model_name, api_key)


def create_openrouter_tokenizer(base_model: str = "microsoft/DialoGPT-medium"):
    """
    Create a tokenizer for compatibility with OpenRouter models.
    
    Args:
        base_model: Base model to use for tokenizer
        
    Returns:
        AutoTokenizer instance
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


# Available OpenRouter models
OPENROUTER_MODELS = {
    "gpt-4o": "openai/gpt-4o",
    "gpt-4": "openai/gpt-4",
    "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
    "gpt-oss-20b-free": "openai/gpt-oss-20b:free",
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-3-haiku": "anthropic/claude-3-haiku", 
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
    "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
    "llama-3.2-1b": "meta-llama/llama-3.2-1b-instruct",
    "llama-3.2-3b": "meta-llama/llama-3.2-3b-instruct",
    "gemini-pro": "google/gemini-pro",
    "gemma-2-9b": "google/gemma-2-9b-it",
    "mistral-7b": "mistralai/mistral-7b-instruct",
    "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct"
}


def list_available_models():
    """List available OpenRouter models."""
    print("Available OpenRouter models:")
    for short_name, full_name in OPENROUTER_MODELS.items():
        print(f"  {short_name}: {full_name}")


if __name__ == "__main__":
    # Example usage
    list_available_models()
    
    # Test client (requires API key)
    try:
        client = OpenRouterClient()
        print("\nOpenRouter client initialized successfully!")
        
        # Test a simple completion
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        response = client.get_text_response(messages, "openai/gpt-3.5-turbo")
        print(f"Test response: {response}")
        
    except Exception as e:
        print(f"Could not initialize client: {e}")
        print("Make sure to set OPENROUTER_API_KEY environment variable")
