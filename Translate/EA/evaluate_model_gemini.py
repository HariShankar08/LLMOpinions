# By default, let the program compute the representativeness of the model.
# We may filter some of the rows later to take a look at Alignment in specific.

import pandas as pd
import json
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import argparse
import google.generativeai as genai
import math
from time import sleep
import os
import pickle

# Configure Gemini API
API_KEY = ''  # Set your Google API key here
genai.configure(api_key=API_KEY)

NUM_REASONING_TOKENS = 100
LANGUAGE = 'en'
COUNTRY = 'hk'

# Default model configuration
DEFAULT_MODEL = 'meta-llama/Llama-3.2-1B-Instruct'

# Country mappings for EA
survey_publics = {
    'hk': 1,
    'jp': 2,
    'ko': 4,
    'tw': 5,
    'vi': 6
}

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
    country_dict_en = {
        'hk': 'Hong Kong',
        'jp': 'Japan',
        'ko': 'South Korea',
        'tw': 'Taiwan',
        'vi': 'Vietnam'
    }

    country_dict_zh = {
        'hk': '香港',    # Hong Kong (no change)
        'jp': '日本',    # Japan (no change)
        'ko': '韓國',    # South Korea (韩国 -> 韓國)
        'tw': '臺灣',    # Taiwan (台湾 -> 臺灣)
        'vi': '越南'     # Vietnam (no change)
    }

    country_dict_ja = {
        'hk': '香港',
        'jp': '日本',
        'ko': '韓国',
        'tw': '台湾',
        'vi': 'ベトナム'
    }

    country_dict_ko = {
        'hk': '홍콩',
        'jp': '일본',
        'ko': '한국',
        'tw': '대만',
        'vi': '베트남'
    }

    country_dict_vi = {
        'hk': 'Hồng Kông',       # Hong Kong
        'jp': 'Nhật Bản',        # Japan
        'ko': 'Hàn Quốc',        # South Korea
        'tw': 'Đài Loan',        # Taiwan
        'vi': 'Việt Nam'         # Vietnam
    }

    if LANGUAGE == 'en':
        prompt = 'Answer the following question, after providing a brief explanation within 50 words.'
    elif LANGUAGE == 'zh':
        prompt = '請回答以下問題，並在50個字以內提供簡要說明。'
    elif LANGUAGE == 'ja':
        prompt = '以下の質問に答えてください。50字以内で簡潔な説明を加えてください。'
    elif LANGUAGE == 'ko':
        prompt = '다음 질문에 답하십시오. 50자 이내로 간단한 설명을 추가하십시오.'
    elif LANGUAGE == 'vi':
        prompt = 'Xin hãy trả lời câu hỏi sau, sau khi cung cấp một lời giải thích ngắn gọn trong vòng 50 từ.'

    if steering:
        raise NotImplementedError("Steering prompts not yet implemented.")

    return prompt


def get_reasoning_start_prompt():
    if LANGUAGE == 'en':
        prompt = f'Let\'s think step by step.'
    elif LANGUAGE == 'zh':
        prompt = f'讓我們一步一步思考。'
    elif LANGUAGE == 'ja':
        prompt = f'一歩ずつ考えてみましょう。'
    elif LANGUAGE == 'ko':
        prompt = f'단계별로 생각해 봅시다.'
    elif LANGUAGE == 'vi':
        prompt = f'Chúng ta hãy suy nghĩ từng bước một.'

    return prompt


def get_model_distribution(df, question, questions, cot=True, cached_distributions=None, model_name=DEFAULT_MODEL):
    # Check if we have a cached distribution for this question
    if cached_distributions is not None and question in cached_distributions:
        print(f"Using cached distribution for question: {question}")
        return cached_distributions[question]
    
    print(f"Computing distribution for question: {question}")
    
    prompt = get_prompt(question, questions)
    system_prompt = get_system_prompt(steering=False)
    reasoning_start_prompt = get_reasoning_start_prompt()

    # Initialize Gemini model
    model = genai.GenerativeModel(model_name)
    
    if cot:
        # Chain-of-thought: generate reasoning, then ask for answer
        full_prompt = f"{system_prompt}\n\n{prompt}\n\n{reasoning_start_prompt}"
        
        try:
            reasoning_response = model.generate_content(full_prompt)
            reasoning = reasoning_response.text.strip()
            
            # Now ask for the final answer
            answer_prompt = f"{system_prompt}\n\n{prompt}\n\n{reasoning}\n\nSelected Option:"
        except Exception as e:
            print(f"Error in reasoning generation: {e}")
            # Fallback to direct answer
            answer_prompt = f"{system_prompt}\n\n{prompt}\n\nSelected Option:"
    else:
        # No CoT: directly ask for the answer
        answer_prompt = f"{system_prompt}\n\n{prompt}\n\nSelected Option:"

    try:
        # Get the model's response
        response = model.generate_content(answer_prompt)
        answer_text = response.text.strip()
        
        # Extract the option from the response
        model_distribution = {}
        options = questions[question]['options']
        
        # Try to find the option in the response
        found_option = None
        for option in options:
            if option.strip().lower() in answer_text.lower():
                found_option = option
                break
        
        if found_option:
            # Assign high probability to the found option, low to others
            for option in options:
                if option == found_option:
                    model_distribution[option] = 0.8
                else:
                    model_distribution[option] = 0.2 / (len(options) - 1)
        else:
            # If no clear option found, use uniform distribution
            uniform_prob = 1.0 / len(options)
            model_distribution = {option: uniform_prob for option in options}
        
        return pd.Series(model_distribution)
        
    except Exception as e:
        print(f"Error getting Gemini response for model {model_name}: {e}")
        print("Falling back to uniform distribution")
        return get_model_distribution_fallback(questions, question)


def get_model_distribution_fallback(questions, question):
    """Fallback method when Gemini API fails."""
    # Use uniform distribution as fallback
    options = questions[question]['options']
    uniform_prob = 1.0 / len(options)
    model_distribution = {option: uniform_prob for option in options}
    return pd.Series(model_distribution)


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
    parser.add_argument('--cot', action='store_true', help='Enable chain-of-thought reasoning')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, 
                       help='Gemini model to use for evaluation (e.g., "gemini-1.5-pro")')
    
    args = parser.parse_args()
    
    COUNTRY = args.country
    LANGUAGE = args.language
    COT = args.cot
    MODEL_NAME = args.model

    # Setup caching
    cache_file = get_cache_filename(COUNTRY, LANGUAGE, MODEL_NAME)
    cached_distributions = load_cached_distributions(cache_file)
    new_distributions = {}

    responses = pd.read_csv('responses.csv')
    with open(f'{COUNTRY}_{LANGUAGE}.json') as f:
        questions = json.load(f)

    survey_public = survey_publics[COUNTRY]
    responses = responses[responses['SurveyPublic'] == survey_public]

    scores = []
    # Rate limiting: sleep between requests
    count = 0
    for question in tqdm(questions):
        if question in ['COUNTRY', 'QRID', 'weight', 'QMLangRec']:
            continue
        if 'question' in questions[question] and 'options' in questions[question]:
            qd1 = get_question_distribution(responses, question)
            qd2 = get_model_distribution(responses, question, questions, cot=COT, 
                                       cached_distributions=cached_distributions, model_name=MODEL_NAME)
            
            # Save the distribution for future use
            new_distributions[question] = qd2
            
            score = compare_distributions(qd1, qd2, num_options=len(responses[question].unique()))
            scores.append(score)
            sleep(2)  # Rate limiting for Gemini API
            count += 1
            if count % 10 == 0:
                sleep(5)  # Longer pause every 10 requests

    # Save all distributions to cache
    all_distributions = {**cached_distributions, **new_distributions}
    save_cached_distributions(cache_file, all_distributions)
    scores = [s for s in scores if s != 1]
    print('=' * 20)
    print('Average Representativeness:', sum(scores) / len(scores) if scores else 0)
