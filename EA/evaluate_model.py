import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import torch
from tqdm import tqdm
import pandas as pd
from scipy.spatial import distance
from matplotlib import pyplot as plt
import re 
from parrot import Parrot
import argparse
from model_questions import *


login(token='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')

paraphraser = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5") 

torch.manual_seed(42)
torch.cuda.manual_seed(42)

def get_paraphrased_prompts(prompt, n=5):
    paraphrases = paraphraser.augment(prompt)
    if paraphrases is not None:
        paraphrases = paraphrases[:n]
    else:
        paraphrases = []
    return [prompt, *paraphrases]


def make_prompt(question, questions):
    if 'question' not in questions[question] or 'options' not in questions[question]:
        print(f'Question {question} is missing question or options')
        return None
    
    prompt = questions[question]['question']
    _, prompt = prompt.split('.', maxsplit=1)
    paraphrased_prompts = get_paraphrased_prompts(prompt)

    paraphrased_prompts = [f'Question: {prompt}' for prompt in paraphrased_prompts]
    
    options = questions[question]['options']

    prompts = []
    expected_answers = []
    for paraphrased_prompt in paraphrased_prompts:
        p = f'{paraphrased_prompt}\n'
        
        for key, value in sorted(options.items(), key=lambda x: x[0]):
            value = re.sub(r'\(.+\)', '', value)
            value = re.sub(r'\[.+\]', '', value)

            p += f'{key}: {value}\n'
            expected_answers.append(key)
        p += '\nAnswer: '
        prompts.append(p)
    
    if question in ['QAGE', 'QFERT', 'HH1', 'HH2']:
        expected_answers = [str(i) for i in range(99)] 

    return prompts, expected_answers

def get_model_vector(model, cols, questions):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    
    model = model.to(device)

    model_vector = {}
    
    for question in tqdm(cols):
        out = make_prompt(question, questions)
        if out is None:
            continue
        prompts, expected_answers = out
        # print(prompts)
        candidate_answers = []
        for prompt in prompts:
            for ablation_seed in (42, 0, 1, 7, 177013):
                random.seed(ablation_seed)
                torch.manual_seed(ablation_seed)
                torch.cuda.manual_seed(ablation_seed)
                
                inputs = tokenizer(prompt, return_tensors='pt')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)

                logits = outputs.logits[:, -1, :]
                probabilities = torch.softmax(logits, dim=-1)

                top_k = 100
                top_probabilities, top_indices = torch.topk(probabilities, top_k)
                top_words = tokenizer.convert_ids_to_tokens(top_indices[0].tolist())

                # print(top_words)

                # Display results
                probabilities = {}
                for word, prob in zip(top_words, top_probabilities[0].tolist()):
                    # print(f"Word: {word}, Log Probability: {prob:.4f}")
                    probabilities[word] = prob
                
                distribution = {}
                for ans in expected_answers:
                    if ans in probabilities:
                        distribution[ans] = probabilities[ans]
                # print(distribution)
        
                candidate_answers.append(max(distribution, key=distribution.get))

        # Fill the model vector with the most common answer
        print(candidate_answers)
        model_vector[question] = max(set(candidate_answers), key=candidate_answers.count)
        print(f'Question: {question}, Answer: {model_vector[question]}')
    return model_vector


def get_human_vectors(cols, country):
    df = pd.read_csv('responses.csv')
    country_to_id = {
        'HKG': 1,
        'JPN': 2,
        'KOR': 4,
        'TWN': 5,
        'VNM': 6
    }

    df = df[df['SurveyPublic'] == country_to_id[country]]
    cols = ['QRID', *cols]
    df = df[cols]
    
    return df

def compute_scores(model_vector, human_vectors_df):
    df = human_vectors_df.copy()
    qrids = df['QRID'].tolist()

    # Drop the QRID column
    df = df.drop('QRID', axis=1)
    
    # Ensure the model_vector is a DataFrame row for consistency
    model_vector_df = pd.DataFrame([model_vector])

    hds = []

    for i, row in df.iterrows():
        # Create a temporary DataFrame with the current row and the model vector
        temp_df = pd.concat([pd.DataFrame([row]), model_vector_df], ignore_index=True)

        # Drop columns containing spaces in either row
        temp_df = temp_df.replace(' ', pd.NA).dropna(axis=1)

        # One-hot encode both rows
        temp_df = pd.get_dummies(temp_df)

        # Calculate the Hamming distance between the two rows
        hamming_dist = distance.hamming(
            temp_df.iloc[0].tolist(), temp_df.iloc[1].tolist()
        )
        hds.append(hamming_dist)

    # Add a column to the dataframe that stores the Hamming distance
    df2 = pd.DataFrame({'QRID': qrids, 'Hamming Distance': hds})

    return df2
    

def get_demographic_info(sorted_scores, country, DEMOGRAPHIC_QUESTIONS):
    responses = pd.read_csv('responses.csv')

    country_to_id = {
        'HKG': 1,
        'JPN': 2,
        'KOR': 4,
        'TWN': 5,
        'VNM': 6
    }

    responses = responses[responses['SurveyPublic'] == country_to_id[country]]
    
    # Only refer to the rows having QRIDs in the sorted_scores
    responses = responses[responses['QRID'].isin(sorted_scores['QRID'])]

    demographic_info = {}
    # For each of the demographic questions, get a frequency per option for the top 1000 respondents
    for question in DEMOGRAPHIC_QUESTIONS:
        demographics = responses[question]
        demographic_info[question] = demographics.value_counts()
    print(demographic_info)
    return demographic_info


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Model to Evaluate', required=True)
parser.add_argument('--country', type=str, help='Country of respondents', required=True)

if __name__ == '__main__':
    args = parser.parse_args()

    model = args.model 
    country = args.country

    allowed_countries = ['HKG','JPN','KOR','TWN','VNM']
    if country not in allowed_countries:
        raise ValueError("Expecting one of ['HKG','JPN','KOR','TWN','VNM']")
    
    with open('questions.json') as f:
        questions = json.load(f)

    cols = list(ASK_ALL_QUESTIONS)    

    if country == 'JPN':
        cols.extend(ASK_JP_QUESTIONS)
    elif country == 'HKG':
        cols.extend(ASK_HK_QUESTIONS)
    elif country == 'KOR':
        cols.extend(ASK_SK_QUESTIONS)
    elif country == 'TWN':
        cols.extend(ASK_TW_QUESTIONS)
    elif country == 'VNM':
        cols.extend(ASK_VIET_QUESTIONS)
    
    mv = get_model_vector(model, cols, questions)

    hv = get_human_vectors(cols, country)    
    
    scores_df = compute_scores(mv, hv)

    sorted_scores = scores_df.sort_values(by='Hamming Distance', ascending=True)
    sorted_scores = sorted_scores.head(100)

    sorted_scores.to_csv('sorted-scores.csv')

    demographic_info = get_demographic_info(sorted_scores, country, DEMOGRAPHIC_QUESTIONS=DEMOGRAPHIC_QUESTIONS)
