# By default, let the program compute the representativeness of the model.
# We may filter some of the rows later to take a look at Alignment in specific.

import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
from scipy.stats import wasserstein_distance
from tqdm import tqdm

# Set all seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
set_seed(42)


NUM_REASONING_TOKENS = 100
LANGUAGE = 'hi'
MODEL = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
TOKENIZER.pad_token = TOKENIZER.eos_token
TOKENIZER.pad_token_id = TOKENIZER.eos_token_id
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

def get_question_distribution(df, question):
    # Get the distribution of answers for a specific question
    question_data = df[question]
    # Convert all values to strings.
    question_data = question_data.astype(str)
    # Ignore rows with NaN values, space, or empty strings
    question_data = question_data[question_data.notna() & (question_data != "") & (question_data.str.strip() != "")]

    return question_data.value_counts(normalize=True)
    

def get_prompt(question, questions):
    # Get the prompt based on the question and its context.
    qdict = questions[question]
    prompt = qdict['question'] + '\n'
    for option in sorted(qdict['options'], key=lambda x: int(x)):
        prompt += f'{option}: {qdict["options"][option]}\n'
    return prompt


def get_system_prompt(steering=False):
    if LANGUAGE == 'en':
        prompt = 'Answer the following question as though you were a citizen of India, after providing a brief explanation within 100 words.'
    elif LANGUAGE == 'hi':
        prompt = 'निम्नलिखित प्रश्न का उत्तर इस प्रकार दीजिए जैसे कि आप भारत के नागरिक हों, तथा 100 शब्दों में संक्षिप्त व्याख्या दीजिए।'

    if steering:
        raise NotImplementedError("Steering prompts not yet implemented.")

    return prompt


def get_reasoning_start_prompt():
    if LANGUAGE == 'en':
        return 'Let’s think step by step.'
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

    # Convert the model_distribution to a pandas series.
    # The values will now be floats, as expected.
    return pd.Series(model_distribution)

def get_model_distribution(df, question, questions, chat_model=True):
    # Get the prompt based on the question.
    prompt = get_prompt(question, questions)
    system_prompt = get_system_prompt(steering=False)
    reasoning_start_prompt = get_reasoning_start_prompt()
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

    # Generate 20 tokens to serve as the model's reasoning. (Deterministic)
    with torch.no_grad():
        outputs = MODEL.generate(**inputs, max_new_tokens=NUM_REASONING_TOKENS, do_sample=False)

    # Decode the output
    generated = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    # Add to the generated prompt, prompt for the final answer.
    generated += '\nAnswer:'

    inputs = TOKENIZER(generated, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

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
    return 1 - (wd / (num_options - 1))

if __name__ == "__main__":
    responses = pd.read_csv('responses.csv')
    with open(f'ind_{LANGUAGE}.json') as f:
        questions = json.load(f)
    
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
            qd2 = get_model_distribution(responses, question, questions)

            score = compare_distributions(qd1, qd2, num_options=len(responses[question].unique()))
            scores.append(score)
    
    print('=' * 20)
    print('Average Representativeness:', sum(scores) / len(scores) if scores else 0)

