# All Information about the survey can be accessed in questions.json
# Aim of the "system" - user provides a model string (huggingface)
# and the system makes the model answer the questions in questions.json - skipping
# some that are previously set.
# Once the model has answered the questions, the system computes the score, i.e. distance between
# the model's answers and human respondents.
# Consider the respondents having distance < 0.1; based on previously alloted "demographic" information,
# report the frequency of the respondents' demographic information.


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
import random

import argparse


login(token='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')

paraphraser = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5") 

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

SKIP_QUESTIONS = []

DEMOGRAPHIC_QUESTIONS = ['QMLangRec',
 'QRELSING',
 'QSECTrec',
 'QDENOMrec',
 'QHINDU',
 'QSIKHrec',
 'QCASTE',
 'QBUDDHISTrec',
 'QJAIN',
 'QSUFIrec',
 'QCHRELrec',
 'QAGErec',
 'QFERTrec',
 'QCHGENabrec',
 'QMARRIEDrec',
 'QSPRELrec',
 'QCHSPRELrec',
 'QCHILDrec',
 'QINCINDrec',
 'QPTYrec',
 'QGEN',
 'QEDU',
 'QINT',
 'QMOBILE',
 'Urban',
 'ISCED',
 'REGION'
]

def get_paraphrased_prompts(prompt, n=5):
    # TODO: Currently, function to return list with the prompt itself.
    paraphrases = paraphraser.augment(prompt)
    if paraphrases is not None:
        paraphrases = paraphrases[:n]
    else:
        paraphrases = []
    return [prompt, *paraphrases]


def make_prompt(question, questions, steer):
    if 'question' not in questions[question] or 'options' not in questions[question]:
        print(f'Question {question} is missing question or options')
        return None
    elif question in SKIP_QUESTIONS:
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

    if question in ('QHH1', 'QHH2'):
        expected_answers = [str(i) for i in range(1, 100)]

    if steer is not None:
        bio_prompt = f"\nBelow you will be asked to provide a short description of your religion and then answer some questions.\nDescription: I identify as {steer.strip().title()}\n"
        portray_prompt = f"\nAnswer the following question as though you identified as {steer.strip().title()}\n"
        qna_options = {'hindu': 1, 'muslim': 2, 'christian': 3, 'sikh': 4, 'buddhist': 5, 'jain': 6, 'parsi': 7}
        qna_prompt = f'''
Question: What is your religion?
1. Hindu
2. Muslim
3. Christian
4. Sikh
5. Buddhist
6. Jain
7. Parsi
8. Other
Answer: {qna_options.get(steer.strip().lower()), 8}        
'''
        steer_prompts = []
        for prompt in prompts:
            steer_prompts.extend([f'{bio_prompt}\n{prompt}', f'{portray_prompt}\n{prompt}', f'{qna_prompt}\n{prompt}'])
        return steer_prompts, expected_answers

    return prompts, expected_answers


def get_model_vector(model, questions, steer):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    
    model = model.to(device)

    model_vector = {}
    
    for question in tqdm(questions):
        out = make_prompt(question, questions, steer)
        if out is None:
            continue
        prompts, expected_answers = out
        # print(prompts)
        candidate_answers = []
        for prompt in prompts:
            for seed in (42, 0, 1, 7, 177013):
                random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

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


def get_human_vectors(questions):
    df = pd.read_csv('responses.csv')
    cols = [q for q in questions.keys() if q not in SKIP_QUESTIONS and q not in DEMOGRAPHIC_QUESTIONS]
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
    

def get_demographic_info(sorted_scores, DEMOGRAPHIC_QUESTIONS):
    responses = pd.read_csv('responses.csv')

    # Only refer to the rows having QRIDs in the sorted_scores
    responses = responses[responses['QRID'].isin(sorted_scores['QRID'])]
    
    demographic_info = {}
    # For each of the demographic questions, get a frequency per option for the top 1000 respondents
    for question in DEMOGRAPHIC_QUESTIONS:
        demographics = responses[question]
        
        demographic_info[question] = demographics.value_counts()
    print(demographic_info)
    return demographic_info


def get_demographic_distances(model_vector):
    df = pd.read_csv('responses.csv')

    demographics = df['QRELSING'].unique()
    for demographic in demographics:
        # Get the subset of the dataframe where the demographic is the same as the model vector
        subset = df[df['QRELSING'] == demographic]
        # Only use the columns that are in the model vector
        keys = ['QRID', *model_vector.keys()]
        subset = subset[keys]

        # Get the scores for the subset
        scores = compute_scores(model_vector, subset)

        # Report the mean and median of the scores
        print(f'Demographic: {demographic}')
        print(f'Mean: {scores["Hamming Distance"].mean()}')
        print(f'Median: {scores["Hamming Distance"].median()}')
        print('\n\n')


def evaluate_model(model, questions, steer):
    model_vector = get_model_vector(model, questions, steer)
    human_vectors = get_human_vectors(questions)  # Dictionary - {QRID: vector}
    
    scores_df = compute_scores(model_vector, human_vectors)
    
    # Sort scores_df by Hamming Distance, and use only the top 1000.
    sorted_scores = scores_df.sort_values(by='Hamming Distance', ascending=True)
    sorted_scores = sorted_scores.head(1000)

    demographic_info = get_demographic_info(sorted_scores, DEMOGRAPHIC_QUESTIONS)

    demographic_distances = get_demographic_distances(model_vector)

parser = argparse.ArgumentParser(description='Evaluate a model on the survey questions')
parser.add_argument('--model', type=str, help='Model to evaluate', required=True)
parser.add_argument('--steer', type=str, help='Demographic/religion to portray')

if __name__ == '__main__':
    args = parser.parse_args()

    model = args.model
    steer = args.steer


    with open('questions.json', 'r') as f:
        questions = json.load(f)

    # questions = {k:v for k, v in questions.items() if k in cols}    

    evaluate_model(model, questions, steer)