# By default, let the program compute the representativeness of the model.
# We may filter some of the rows later to take a look at Alignment in specific.


import pandas as pd
import json
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import argparse
from openai import openai
import math



API_KEY = ''  # Set your OpenRouter API key here

client = openai.Client(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)


NUM_REASONING_TOKENS = 100
LANGUAGE = 'en'
COUNTRY = 'hk'
country_ids = {
    'ca': 1,
    'id': 2,
    'ms': 3,
    'sg': 5,
    'sl': 6,
    'th': 7
}

MODEL='meta-llama/Llama-3.2-1B-Instruct'

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
        prompt = f"Answer the following question, after providing a brief explanation within 50 words. Answer the following question as though you were a citizen of {country_dict_en[COUNTRY]}."
    elif LANGUAGE == 'km':
        prompt = f"សូមឆ្លើយសំណួរ បន្ទាប់ពីផ្តល់ការពន្យល់ខ្លីក្នុងរយៈពេល 50 ពាក្យ។ សូមឆ្លើយដូចជាអ្នកជាពលរាស្ត្រនៃ {country_dict_km[COUNTRY]}."
    elif LANGUAGE == 'zh':
        prompt = f"请回答以下问题，并用50字以内的内容进行简要说明。请以{country_dict_zh[COUNTRY]}公民的身份作答。"
    elif LANGUAGE == 'ma':
        prompt = f"Sila jawab soalan berikut, selepas memberikan penjelasan ringkas dalam 50 patah perkataan. Jawablah seolah-olah anda warganegara {country_dict_ma[COUNTRY]}."
    elif LANGUAGE == 'id':
        prompt = f"Silakan jawab pertanyaan berikut, setelah memberikan penjelasan singkat dalam 50 kata. Jawablah seolah-olah Anda adalah warga negara {country_dict_id[COUNTRY]}."
    elif LANGUAGE == 'ta':
        prompt = f"தயவுசெய்து கீழ்காணும் கேள்விக்கு பதிலளிக்கவும், 50 வார்த்தைகளுக்குள் சுருக்கமான விளக்கத்தை வழங்கவும். நீங்கள் {country_dict_ta[COUNTRY]} நாட்டின் குடிமகனாக இருப்பது போல பதிலளிக்கவும்."
    elif LANGUAGE == 'si':
        prompt = f"කරුණාකර පහත ප්‍රශ්නයට පිළිතුරු දෙන්න, වචන 50 කින් කෙටි විස්තරයක් ලබා දී. ඔබ {country_dict_si[COUNTRY]} පුරවැසියෙකු ලෙස පිළිතුරු දෙන්න."
    elif LANGUAGE == 'th':
        prompt = f"กรุณาตอบคำถามต่อไปนี้ หลังจากให้คำอธิบายสั้น ๆ ภายใน 50 คำ กรุณาตอบราวกับว่าคุณเป็นพลเมืองของ {country_dict_th[COUNTRY]}."
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

    # Convert the model_distribution to a pandas series.
    # The values will now be floats, as expected.
    return pd.Series(model_distribution)




def get_model_distribution(df, question, questions, cot=True):
    prompt = get_prompt(question, questions)
    system_prompt = get_system_prompt(steering=False)
    reasoning_start_prompt = get_reasoning_start_prompt()

    if cot:
        # Chain-of-thought: generate reasoning, then ask for answer
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reasoning_start_prompt}
        ]
        reasoning_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=NUM_REASONING_TOKENS,
            temperature=0.0
        )
        reasoning = reasoning_response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": reasoning})
        messages.append({"role": "user", "content": "Answer:"})
    else:
        # No CoT: directly ask for the answer
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt + "\nAnswer:"}
        ]

    try:
        answer_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=10
        )

        # Check if logprobs are available
        if (answer_response.choices[0].logprobs is None or 
            not answer_response.choices[0].logprobs.content or
            not answer_response.choices[0].logprobs.content[0].top_logprobs):
            print(f"Warning: Logprobs not available for model {MODEL}, using fallback method")
            return get_model_distribution_fallback(questions, question)

        logprobs = answer_response.choices[0].logprobs.content[0].top_logprobs

        model_distribution = {}
        for option in questions[question]['options']:
            prob = 0.0
            for token, logp in logprobs.items():
                if token.strip() == option:
                    prob = float(pow(10, logp / math.e))
                    break
            model_distribution[option] = prob

        total = sum(model_distribution.values())
        if total > 0:
            for k in model_distribution:
                model_distribution[k] /= total
        else:
            # If no probabilities found, use uniform distribution
            print(f"Warning: No valid probabilities found for question {question}, using uniform distribution")
            return get_model_distribution_fallback(questions, question)

        return pd.Series(model_distribution)
        
    except Exception as e:
        print(f"Error getting logprobs for model {MODEL}: {e}")
        print("Falling back to alternative method")
        return get_model_distribution_fallback(questions, question)


def get_model_distribution_fallback(questions, question):
    """Fallback method when logprobs are not available."""
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
    parser.add_argument('--model', type=str, default=MODEL, help='Model to use for evaluation')
    
    args = parser.parse_args()
    COUNTRY = args.country
    LANGUAGE = args.language
    COT = args.cot
    MODEL = args.model

    responses = pd.read_csv('responses.csv')
    with open(f'{COUNTRY}_{LANGUAGE}.json') as f:
        questions = json.load(f)

    survey_public = survey_publics[COUNTRY]
    responses = responses[responses['SurveyPublic'] == survey_public]

    scores = []
    for question in tqdm(questions):
        if question in ['COUNTRY', 'QRID', 'weight', 'QMLangRec']:
            continue
        if 'question' in questions[question] and 'options' in questions[question]:
            qd1 = get_question_distribution(responses, question)
            qd2 = get_model_distribution(responses, question, questions, cot=COT)
            score = compare_distributions(qd1, qd2, num_options=len(responses[question].unique()))
            scores.append(score)

    print('=' * 20)
    print('Average Representativeness:', sum(scores) / len(scores) if scores else 0)

