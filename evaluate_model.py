# All Information about the survey can be accessed in questions.json
# Aim of the "system" - user provides a model string (huggingface)
# and the system makes the model answer the questions in questions.json - skipping
# some that are previously set.
# Once the model has answered the questions, the system computes the score, i.e. distance between
# the model's answers and human respondents.
# Consider the respondents having distance < 0.1; based on previously alloted "demographic" information,
# report the frequency of the respondents' demographic information.


import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch
from tqdm import tqdm
import pandas as pd
from scipy.spatial import distance
from matplotlib import pyplot as plt
import re


login(token='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')

cols = ['Q2', 'Q6a', 'Q6b', 'Q6c', 'Q6d', 'Q6e', 'Q7', 'Q9', 'ABORT', 'Q10', 
            'Q11ya', 'Q11yb', 'Q11yc', 'Q11yd', 'Q11ye', 'Q11yf', 'Q11yg', 'Q11yh',
            'Q11yi', 'Q12', 'Q17a', 'Q17b', 'Q19a', 'Q19b', 'QW21a', 'QW21b', 'QM21a',
            'QM21b', 'Q22a', 'Q22b', 'Q22c', 'Q23a', 'Q23b', 'Q24', 'Q26', 'Q35', 'Q70']
    

SKIP_QUESTIONS = ['Q44rec_12', 'Q86b_3', 'Q44rec_99', 'Q86b_21', 'Q86b_8', 'QSECTrec',
                   'Q3rec_7', 'Q86b_6', 'QSUFIrec', 'Q86b_16', 'Q44rec_6', 'Q44rec_14', 'QCHGENabrec',
                     'QAGErec', 'q73abrec', 'Q86b_20', 'Q28rec', 'Q44rec_7', 'Q44rec_8', 'Q3132rec', 'QMARRIEDrec', 
                     'Q44rec_4', 'Q86b_11', 'Q44rec_1', 'Q77arec', 'Q44rec_2', 'Q3rec_5', 'Q86b_2', 'Q86b_1', 'QCHSPRELrec',
                       'Q3rec_99', 'QSIKHrec', 'Q3rec_98', 'QBUDDHISTrec', 'Q86b_7', 'Q44rec_5', 'QCHRELrec', 'q85arec', 'Q86b_19',
                         'Q86b_17', 'weight', 'Q86b_18', 'Q86b_15', 'Q44rec_9', 'Q3rec_4', 'Q44rec_15', 'QFERTrec', 'Q44rec_11',
                           'COUNTRY', 'Q44rec_10', 'Q86b_99', 'q29cdrec', 'Q86b_4', 'Q86b_98', 'Q86b_12', 'Q86b_14',
                             'Q44rec_17', 'Q86b_9', 'QSPRELrec', 'Q3rec_2', 'QPTYrec', 'QCHILDrec', 'Q44rec_13', 'Q86b_5',
                               'QDENOMrec', 'Q86b_97', 'q30derec', 'Q44rec_18', 'Q3rec_6', 'Q86b_10', 'Q3rec_1', 'Q3rec_3',
                                 'Q44rec_3', 'Q86b_13', 'Q44rec_98', 'Q44rec_16', 'QINCINDrec', 'QIV7', 'QHH1', 'QHH2']

DEMOGRAPHIC_QUESTIONS = ['QMLangRec',
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

def get_paraphrased_prompts(prompt):
    # TODO: Currently, function to return list with the prompt itself.
    return [prompt]


def make_prompt(question, questions):
    if 'question' not in questions[question] or 'options' not in questions[question]:
        print(f'Question {question} is missing question or options')
        return None
    elif question in SKIP_QUESTIONS:
        return None
    
    prompt = questions[question]['question']
    _, prompt = prompt.split('.', maxsplit=1)
    prompt = f'Question: {prompt}'
    
    
    paraphrased_prompts = get_paraphrased_prompts(prompt)
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

    return prompts, expected_answers


def get_model_vector(model, questions):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    
    model = model.to(device)

    model_vector = {}
    
    for question in tqdm(questions):
        if question in DEMOGRAPHIC_QUESTIONS or question in SKIP_QUESTIONS:
            continue
        out = make_prompt(question, questions)
        if out is None:
            continue
        prompts, expected_answers = out
        print(prompts)
        candidate_answers = []
        for prompt in prompts:
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
        model_vector[question] = max(set(candidate_answers), key=candidate_answers.count)
        print(f'Question: {question}, Answer: {model_vector[question]}')
    return model_vector


def get_human_vectors(questions):
    df = pd.read_csv('responses.csv')
    cols = [q for q in questions.keys() if q not in SKIP_QUESTIONS and q not in DEMOGRAPHIC_QUESTIONS]
    human_vectors = {}
    
    # Make QRID the index.
    # df = df.set_index('QRID')
    df = df[cols]
    # df = df.set_index('QRID')
    
    return df

def compute_scores(model_vector, human_vectors_df):
    df = human_vectors_df.copy()
    qrids = df['QRID']

    # Drop the QRID column
    df = df.drop('QRID', axis=1)
    
    df = df.replace(' ', 99)
    df = df.fillna(99)

    df = pd.get_dummies(df)
    print(df.columns)
    
    human_vector = df.iloc[-1]
    df = df.drop(df.index[-1])

    hds = []

    for i, row in df.iterrows():
        print(distance.hamming(row.tolist(), human_vector.tolist()))
        hds.append(distance.hamming(row.tolist(), human_vector.tolist()))

    plt.hist(hds)
    plt.savefig('hist.png')


    # Add a column to the dataframe that stores the hamming distance between the model vector and the human vector
    df2 = pd.DataFrame(hds, columns=['Hamming Distance'])
    df2 = pd.concat([qrids, df2], axis=1)
    
    return df2
    

def get_demographic_info(sorted_scores, questions, DEMOGRAPHIC_QUESTIONS):
    # For each of the Demographic Questions, get the frequency of the respondents
    # in the top 1000.
    demographic_info = {}
    for question in DEMOGRAPHIC_QUESTIONS:
        if question not in questions:
            continue
        demographic_info[question] = {}
        for i, row in sorted_scores.iterrows():
            value = questions[question]['options'][qrid]
            if value not in demographic_info[question]:
                demographic_info[question][value] = 0
            demographic_info[question][value] += 1


def evaluate_model(model, questions):
    model_vector = get_model_vector(model, questions)
    human_vectors = get_human_vectors(questions)  # Dictionary - {QRID: vector}
    
    scores_df = compute_scores(model_vector, human_vectors)
    
    # Sort scores_df by Hamming Distance, and use only the top 1000.
    sorted_scores = scores_df.sort_values(by='Hamming Distance', ascending=True)
    sorted_scores = sorted_scores.head(1000)

    demographic_info = get_demographic_info(sorted_scores, DEMOGRAPHIC_QUESTIONS)


    

if __name__ == '__main__':
    with open('questions.json', 'r') as f:
        questions = json.load(f)

    # questions = {k:v for k, v in questions.items() if k in cols}    

    model = 'meta-llama/Llama-3.2-1B'
    evaluate_model(model, questions)