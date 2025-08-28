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
LANGUAGE = 'en'
COUNTRY = 'ca'

# Default model configuration
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_MODEL_SHORT = "llama-3.2-1b-instruct"   

# Global variables to be set after argument parsing
MODEL = None
TOKENIZER = None
DEVICE = None

country_ids = {
    'ca': 1,
    'id': 2,
    'ms': 3,
    'sg': 5,
    'sl': 6,
    'th': 7
}

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
        prompt += f'{option}: {qdict["options"][option]}\n'
    return prompt


def get_system_prompt(steering=False):
    country_dict_en = {
        'ca': 'Cambodia',
        'id': 'Indonesia', 
        'ms': 'Malaysia',
        'sg': 'Singapore',
        'sl': 'Sri Lanka',
        'th': 'Thailand'
    }

    country_dict_km = {
        'ca': 'កម្ពុជា',
        'id': 'ឥណ្ឌូណេស៊ី',
        'ms': 'ម៉ាឡេស៊ី',
        'sg': 'សិង្ហបុរី',
        'sl': 'ស្រីលង្កា',
        'th': 'ថៃ'
    }

    country_dict_zh = {
        'ca': '柬埔寨',
        'id': '印度尼西亚',
        'ms': '马来西亚',
        'sg': '新加坡',
        'sl': '斯里兰卡',
        'th': '泰国'
    }

    country_dict_ma = {
        'ca': 'Kemboja',
        'id': 'Indonesia',
        'ms': 'Malaysia',
        'sg': 'Singapura',
        'sl': 'Sri Lanka',
        'th': 'Thailand'
    }  # Malay

    country_dict_id = {
        'ca': 'Kamboja',
        'id': 'Indonesia',
        'ms': 'Malaysia',
        'sg': 'Singapura',
        'sl': 'Sri Lanka',
        'th': 'Thailand'
    }

    country_dict_ta = {
        'ca': 'கம்போடியா',
        'id': 'இந்தோனேசியா',
        'ms': 'மலேசியா',
        'sg': 'சிங்கப்பூர்',
        'sl': 'ஸ்ரீலங்கா',
        'th': 'தாய்லாந்து'
    }  # Tamil

    country_dict_si = {
        'ca': 'කම්බෝජ',
        'id': 'ඉන්දෝනීසිය',
        'ms': 'මාලේසියාව',
        'sg': 'සින්ගප්පූර්',
        'sl': 'ශ්‍රී ලංකාව',
        'th': 'තායිලන්තය'
    } # Sinhala

    country_dict_th = {
        'ca': 'កម្ពុជា',
        'id': 'ឥណ្ឌូណេស៊ី',
        'ms': 'ម៉ាឡេស៊ី',
        'sg': 'សិង្ហបុរី',
        'sl': 'ស្រីលង្កា',
        'th': 'ថៃ'
    } # Thai

    if LANGUAGE == 'en':
        prompt = 'Answer the following question.'
    elif LANGUAGE == 'km': # Translation of "Answer the following question."
        prompt = 'សូមឆ្លើយសំណួរ'
    elif LANGUAGE == 'zh': # Translation of "Answer the following question."        
        prompt = '请回答以下问题。'
    elif LANGUAGE == 'ma':
        prompt = 'Sila jawab soalan berikut.'
    elif LANGUAGE == 'id':
        prompt = 'Silakan jawab pertanyaan berikut.'
    elif LANGUAGE == 'ta':
        prompt = 'தயவுசெய்து கீழ்காணும் கேள்விக்கு பதிலளிக்கவும்.'
    elif LANGUAGE == 'si':
        prompt = 'කරුණාකර පහත ප්‍රශ්නයට පිළිතුරු දෙන්න.'
    elif LANGUAGE == 'th':  # Translation of "Answer the following question."
        prompt = 'กรุณาตอบคำถามต่อไปนี้'
    if steering:
        raise NotImplementedError("Steering prompts not yet implemented.")

    return prompt


def get_reasoning_start_prompt():
    if LANGUAGE == 'en':
        prompt = f'Let’s think step by step.'
    elif LANGUAGE == 'km':
        prompt = f'មកគិតជាដំណាក់កាលៗគ្នា។'
    elif LANGUAGE == 'zh':
        prompt = f'让我们一步一步思考。'
    elif LANGUAGE == 'ma':
        prompt = f'Sila fikir langkah demi langkah.'
    elif LANGUAGE == 'id':
        prompt = f'Silakan berpikir langkah demi langkah.'
    elif LANGUAGE == 'ta':
        prompt = 'யோசித்து கேள்விக்கு பதிலளிப்போம்.'
    elif LANGUAGE == 'si':  
        prompt = 'අපි ඒ ගැන පියවරෙන් පියවර සිතමු.'
    elif LANGUAGE == 'th':
        prompt = 'กรุณาคิดทีละขั้นตอน'
    return prompt

    

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
    parser.add_argument('--country', type=str, default=COUNTRY)
    parser.add_argument('--language', type=str, default=LANGUAGE)
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_NAME, 
                       help='Model name to use for evaluation (e.g., "meta-llama/Llama-3.2-1B-Instruct")')
    parser.add_argument('--secondary-filter-var', type=str, default=None)
    parser.add_argument('--secondary-filter-value', type=str, default=None)
    args = parser.parse_args()
    
    COUNTRY = args.country
    LANGUAGE = args.language
    MODEL_NAME = args.model
    SECONDARY_FILTER_VAR = args.secondary_filter_var
    SECONDARY_FILTER_VALUE = args.secondary_filter_value

    # Initialize model after parsing arguments
    initialize_model(MODEL_NAME)

    # Disable caching
    cached_distributions = {}

    responses = pd.read_csv('responses.csv')
    with open(f'{COUNTRY}_{LANGUAGE}.json') as f:
        questions = json.load(f)

    country = country_ids[COUNTRY]
    responses = responses[responses['COUNTRY'] == country]
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
            qd2 = get_model_distribution(responses, question, questions, cached_distributions=None)

            if qd1.sum() == 0 or qd2.sum() == 0:
                print(f"Skipping question {question}: zero-sum distribution detected")
                continue
            score = compare_distributions(qd1, qd2, num_options=len(responses[question].unique()))
            print(f"Question {question} score: {score}")
            scores.append(score)
    
    print('=' * 20)
    print('Average Representativeness:', sum(scores) / len(scores) if scores else 0)

