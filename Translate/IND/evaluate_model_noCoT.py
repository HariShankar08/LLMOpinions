# By default, let the program compute the representativeness of the model.
# We may filter some of the rows later to take a look at Alignment in specific.

import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import argparse
import os
import pickle

# Set all seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
set_seed(42)


NUM_REASONING_TOKENS = 100
LANGUAGE = 'hi'

# Default model configuration
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_MODEL_SHORT = "llama-3.2-1b-instruct"   

# Global variables to be set after argument parsing
MODEL = None
TOKENIZER = None
DEVICE = None

def initialize_model(model_name):
    """Initialize the model and tokenizer."""
    global MODEL, TOKENIZER, DEVICE
    
    print(f"Loading model: {model_name}")
    MODEL = AutoModelForCausalLM.from_pretrained(model_name)
    TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    TOKENIZER.pad_token = TOKENIZER.eos_token
    TOKENIZER.pad_token_id = TOKENIZER.eos_token_id
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    
    print(f"Model loaded on device: {DEVICE}")

def get_cache_filename(country, language, model_name):
    """Generate cache filename based on country, language, and model."""
    # Clean model name for filename
    clean_model_name = model_name.replace('/', '_').replace('-', '_')
    return f"cache_{country}_{language}_{clean_model_name}.pkl"

def load_cached_distributions(cache_file):
    """Load cached model distributions from file."""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load cache file {cache_file}: {e}")
    return {}

def save_cached_distributions(cache_file, distributions):
    """Save model distributions to cache file."""
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(distributions, f)
        print(f"Saved cached distributions to {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save cache file {cache_file}: {e}")

def get_question_distribution(df, question):
    if question not in df.columns or 'weight' not in df.columns:
        raise ValueError(f"DataFrame must contain both '{question}' and 'weight' columns.")

    temp_df = df[[question, 'weight']].copy()

    temp_df[question] = temp_df[question].astype(str)
    temp_df[question] = temp_df[question].str.strip()
    temp_df = temp_df[temp_df[question].notna() & (temp_df[question] != "") & (temp_df[question].str.lower() != 'nan')]

    weighted_counts = temp_df.groupby(question)['weight'].sum()
    total_weight = weighted_counts.sum()
    if total_weight == 0:
        return pd.Series(dtype=float)

    weighted_distribution = weighted_counts / total_weight
    return weighted_distribution
    

def get_prompt(question, questions):
    # Get the prompt based on the question and its context.
    qdict = questions[question]
    prompt = f"Question: {qdict['question']}\n"
    for option in sorted(qdict['options'], key=lambda x: int(x)):
        prompt += f"{option}: {qdict['options'][option]}\n"
    return prompt


def get_system_prompt(steering=False):
    if LANGUAGE == 'en':
        prompt = 'Answer the following question.'
    elif LANGUAGE == 'hi':
        prompt = 'निम्नलिखित प्रश्न का उत्तर दीजिए।'

    if steering:
        raise NotImplementedError("Steering prompts not yet implemented.")

    return prompt


def get_reasoning_start_prompt():
    if LANGUAGE == 'en':
        return 'Let\'s think step by step.'
    elif LANGUAGE == 'hi':
        return 'चलिए, चरणबद्ध तरीके से सोचते हैं।'
    

def make_model_distribution(logits, question, questions_dict):
    # Make a dictionary to hold the model's answer distribution
    model_distribution = {}

    # `logits` here are actually probabilities after softmax, which is correct
    probs = logits 

    for option in questions_dict[question]['options']:
        # 1. Encode the option string, disable special tokens, and get the first token ID.
        #    This ensures you get the ID for just the character (e.g., "1") itself.
        option_token_id = TOKENIZER.encode(option, add_special_tokens=False)[0]

        # 2. Get the probability for that single token ID and convert it to a float.
        model_distribution[option] = probs[0, option_token_id].item()

    # Convert to a pandas Series and normalize to sum to 1
    series = pd.Series(model_distribution, dtype=float)
    total = series.sum()
    if total > 0:
        series = series / total
    return series

def get_model_distribution(df, question, questions, chat_model=True, cached_distributions=None):
    # Check if we have a cached distribution for this question
    if cached_distributions is not None and question in cached_distributions:
        print(f"Using cached distribution for question: {question}")
        return cached_distributions[question]
    
    print(f"Computing distribution for question: {question}")
    
    # Get the prompt based on the question.
    prompt = get_prompt(question, questions)
    system_prompt = get_system_prompt(steering=False)
    reasoning_start_prompt = "Answer: "
    # If the model is a chat_model, use a chat_template and tokenize.
    # Otherwise, tokenize it directly.
    if chat_model:
        messages = [
            {'role': "system", 'content': system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reasoning_start_prompt}

        ]
        inputs = TOKENIZER.apply_chat_template(messages, tokenize=True, return_tensors="pt", return_dict=True)
    else:
        inputs = TOKENIZER(f"{system_prompt}\n{prompt}\n{reasoning_start_prompt}\n", return_tensors="pt", return_dict=True)

    # Move everything to the appropriate device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    MODEL.to(DEVICE)

    """
    # Generate 20 tokens to serve as the model's reasoning. (Deterministic)
    with torch.no_grad():
        outputs = MODEL.generate(**inputs, max_new_tokens=NUM_REASONING_TOKENS, do_sample=False)

    # Decode the output
    generated = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    # Add to the generated prompt, prompt for the final answer.
    generated += '\nAnswer:'

    inputs = TOKENIZER(generated, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    """
    with torch.no_grad():
        # Run the model on the new input. Get the logits of the last token.
        outputs = MODEL(**inputs)
        logits = outputs.logits
        last_token_logits = logits[:, -1, :]
        probs = last_token_logits.softmax(dim=-1)
        
    dist = make_model_distribution(probs, question, questions)
    return dist



def compare_distributions(d1, d2, num_options):
    # Compute the Wasserstein distance between two distributions
    # d1 and d2 are pandas Series. Ensure they have the same index.
    if not d1.index.equals(d2.index):
        d1 = d1.reindex(d2.index, fill_value=0)
    
    wd = wasserstein_distance(d1, d2)
    if num_options == 1:
        return 1  # There is no diversity if there is only one option.
    return 1 - (wd / (num_options - 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--country', type=str, default=None)
    parser.add_argument('--language', type=str, default=LANGUAGE)
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_NAME, 
                       help='Model name to use for evaluation (e.g., "meta-llama/Llama-3.2-1B-Instruct")')
    parser.add_argument('--secondary-filter-var', type=str, default=None)
    parser.add_argument('--secondary-filter-value', type=int, default=None)
    args = parser.parse_args()
    
    LANGUAGE = args.language
    MODEL_NAME = args.model
    SECONDARY_FILTER_VAR = args.secondary_filter_var
    SECONDARY_FILTER_VALUE = args.secondary_filter_value
    COUNTRY = 'ind'  # Default for IND region

    # Initialize model after parsing arguments
    initialize_model(MODEL_NAME)

    # Enable caching
    cache_file = get_cache_filename('ind', LANGUAGE, MODEL_NAME)
    cached_distributions = load_cached_distributions(cache_file)
    new_distributions = {}

    responses = pd.read_csv('responses.csv')
    with open(f'ind_{LANGUAGE}.json') as f:
        questions = json.load(f)
    
    if SECONDARY_FILTER_VAR is not None and SECONDARY_FILTER_VALUE is not None:
        responses = responses[responses[SECONDARY_FILTER_VAR] == SECONDARY_FILTER_VALUE]

    scores = []
    for question in tqdm(questions):
        if question in ['COUNTRY', 'QRID', 'weight', 'QMLangRec']:
            continue
        if 'question' in questions[question] and 'options' in questions[question]:
            qd1 = get_question_distribution(responses, question)
            
            # qd1 represents the distribution from the responses file.
            # We now need qd2, which represents the distribution from the model's response.
            # We first need to generate the model response.
            # Assuming we have a function to generate model responses
            qd2 = get_model_distribution(responses, question, questions, cached_distributions=cached_distributions)

            # Save the distribution for future use
            new_distributions[question] = qd2

            if qd1.sum() == 0 or qd2.sum() == 0:
                print(f"Skipping question {question}: zero-sum distribution detected")
                continue
            score = compare_distributions(qd1, qd2, num_options=len(responses[question].unique()))
            print(f"Question {question} score: {score}")
            scores.append(score)
    
    # Save all distributions to cache
    all_distributions = {**cached_distributions, **new_distributions}
    save_cached_distributions(cache_file, all_distributions)
    scores = [s for s in scores if s != 1]
    print('=' * 20)
    print('Average Representativeness:', sum(scores) / len(scores) if scores else 0)

