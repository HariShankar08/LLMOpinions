# By default, let the program compute the representativeness of the model.
# We may filter some of the rows later to take a look at Alignment in specific.

import pandas as pd
import json
from transformers import AutoModelForCausalLM
from collections import Counter
import torch
from scipy.stats import wasserstein_distance

NUM_REASONING_TOKENS = 20

def get_question_distribution(df, question, logits=None):
    # Get the distribution of answers for a specific question
    question_data = df[df['question'] == question]
    if logits is None:
        return question_data['answer'].value_counts(normalize=True)


def compare_distributions(d1, d2, num_options):
    # Compute the Wasserstein distance between two distributions
    wd = wasserstein_distance(d1, d2)
    return (1 - wd) / (num_options - 1)

if __name__ == "__main__":
    responses = pd.read_csv('responses.csv')
    with open('questions.json') as f:
        questions = json.load(f)
    
    for question in questions:
        qd1 = get_question_distribution(responses, question)
        qd2 = get_question_distribution(responses, question)
        print(question, compare_distributions(qd1, qd2, num_options=len(responses['answer'].unique())))

