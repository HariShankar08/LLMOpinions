"""
Enhanced evaluation script that supports both HuggingFace models and OpenRouter API models.
This script can be used as a drop-in replacement for the original evaluate_model.py scripts.
"""

import json
import torch
import pandas as pd
from scipy.spatial import distance
import re 
import argparse
from tqdm import tqdm
import os
import random

# Import OpenRouter client
from openrouter_client import OpenRouterClient, OpenRouterModelWrapper, OPENROUTER_MODELS

# Try to import HuggingFace components (they may not be available in all environments)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from huggingface_hub import login
    HF_AVAILABLE = True
except ImportError:
    print("Warning: HuggingFace transformers not available. Only OpenRouter models will work.")
    HF_AVAILABLE = False

try:
    from parrot import Parrot
    PARROT_AVAILABLE = True
except ImportError:
    print("Warning: Parrot paraphraser not available. Using original prompts only.")
    PARROT_AVAILABLE = False


class ModelEvaluator:
    """Unified model evaluator that works with both HuggingFace and OpenRouter models."""
    
    def __init__(self, model_name: str, use_openrouter: bool = False, openrouter_api_key: str = None):
        """
        Initialize the model evaluator.
        
        Args:
            model_name: Model identifier (HuggingFace repo or OpenRouter model)
            use_openrouter: Whether to use OpenRouter API
            openrouter_api_key: OpenRouter API key (if not set as environment variable)
        """
        self.model_name = model_name
        self.use_openrouter = use_openrouter
        
        # Set up seeds for reproducibility
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        random.seed(42)
        
        if use_openrouter:
            self._setup_openrouter_model(openrouter_api_key)
        else:
            self._setup_huggingface_model()
        
        # Initialize paraphraser if available
        self.paraphraser = None
        if PARROT_AVAILABLE:
            try:
                self.paraphraser = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")
            except Exception as e:
                print(f"Warning: Could not initialize paraphraser: {e}")
    
    def _setup_openrouter_model(self, api_key: str = None):
        """Setup OpenRouter model."""
        # Convert short name to full OpenRouter model name if needed
        if self.model_name in OPENROUTER_MODELS:
            openrouter_model_name = OPENROUTER_MODELS[self.model_name]
        else:
            openrouter_model_name = self.model_name
        
        self.client = OpenRouterClient(api_key)
        self.model_wrapper = OpenRouterModelWrapper(openrouter_model_name, api_key)
        
        # Set up tokenizer for compatibility
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.device = torch.device("cpu")  # API models don't use local device
        print(f"Initialized OpenRouter model: {openrouter_model_name}")
    
    def _setup_huggingface_model(self):
        """Setup HuggingFace model."""
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace transformers not available. Install with: pip install transformers")
        
        # Login to HuggingFace if token is available
        token = os.getenv('HF_TOKEN', '')
        if token:
            login(token=token)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        
        self.model = self.model.to(self.device)
        print(f"Initialized HuggingFace model: {self.model_name} on {self.device}")
    
    def get_paraphrased_prompts(self, prompt: str, n: int = 5) -> list:
        """Get paraphrased versions of a prompt."""
        if self.paraphraser is None:
            return [prompt]
        
        try:
            paraphrases = self.paraphraser.augment(prompt)
            if paraphrases is not None:
                paraphrases = [p[0] for p in paraphrases[:n]]  # Extract text from tuples
            else:
                paraphrases = []
            return [prompt, *paraphrases]
        except Exception as e:
            print(f"Warning: Paraphrasing failed: {e}")
            return [prompt]
    
    def make_prompt(self, question: str, questions: dict) -> tuple:
        """Create prompt and expected answers for a question."""
        if 'question' not in questions[question] or 'options' not in questions[question]:
            print(f'Question {question} is missing question or options')
            return None
        
        prompt = questions[question]['question']
        if '.' in prompt:
            _, prompt = prompt.split('.', maxsplit=1)
        
        paraphrased_prompts = self.get_paraphrased_prompts(prompt)
        paraphrased_prompts = [f'Question: {p}' for p in paraphrased_prompts]
        
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
        
        # Special handling for certain questions
        if question in ['QAGE', 'QFERT', 'HH1', 'HH2']:
            expected_answers = [str(i) for i in range(99)]
        
        return prompts, expected_answers
    
    def get_model_response_openrouter(self, prompt: str, expected_answers: list) -> str:
        """Get model response using OpenRouter API."""
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Get probabilities for each expected answer
            option_probs = self.client.get_logits_for_options(
                messages, 
                self.model_wrapper.model_name, 
                expected_answers
            )
            
            # Return the option with highest probability
            return max(option_probs, key=option_probs.get)
            
        except Exception as e:
            print(f"Error getting OpenRouter response: {e}")
            # Fallback: try to get a direct response and parse it
            try:
                response = self.client.get_text_response(messages, self.model_wrapper.model_name, max_tokens=10)
                # Try to match response to expected answers
                for answer in expected_answers:
                    if answer.lower() in response.lower():
                        return answer
                # If no match, return first expected answer
                return expected_answers[0] if expected_answers else "1"
            except:
                return expected_answers[0] if expected_answers else "1"
    
    def get_model_response_huggingface(self, prompt: str, expected_answers: list) -> str:
        """Get model response using HuggingFace model."""
        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
            probabilities = torch.softmax(logits, dim=-1)
            
            top_k = 100
            top_probabilities, top_indices = torch.topk(probabilities, top_k)
            top_words = self.tokenizer.convert_ids_to_tokens(top_indices[0].tolist())
            
            # Create probability distribution for expected answers
            prob_dict = {}
            for word, prob in zip(top_words, top_probabilities[0].tolist()):
                prob_dict[word] = prob
            
            distribution = {}
            for ans in expected_answers:
                if ans in prob_dict:
                    distribution[ans] = prob_dict[ans]
            
            if distribution:
                return max(distribution, key=distribution.get)
            else:
                return expected_answers[0] if expected_answers else "1"
    
    def get_model_vector(self, questions_list: list, questions_dict: dict) -> dict:
        """Get model response vector for all questions."""
        model_vector = {}
        
        for question in tqdm(questions_list, desc="Processing questions"):
            out = self.make_prompt(question, questions_dict)
            if out is None:
                continue
            
            prompts, expected_answers = out
            candidate_answers = []
            
            for prompt in prompts:
                for ablation_seed in (42, 0, 1, 7, 177013):
                    random.seed(ablation_seed)
                    torch.manual_seed(ablation_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(ablation_seed)
                    
                    if self.use_openrouter:
                        answer = self.get_model_response_openrouter(prompt, expected_answers)
                    else:
                        answer = self.get_model_response_huggingface(prompt, expected_answers)
                    
                    candidate_answers.append(answer)
            
            # Use most common answer
            if candidate_answers:
                model_vector[question] = max(set(candidate_answers), key=candidate_answers.count)
            else:
                model_vector[question] = expected_answers[0] if expected_answers else "1"
            
            print(f'Question: {question}, Answer: {model_vector[question]}')
        
        return model_vector


def get_human_vectors(cols: list, country: str, region: str = "EA") -> pd.DataFrame:
    """Get human response vectors from survey data."""
    df = pd.read_csv('responses.csv')
    
    if region == "EA":
        country_to_id = {
            'HKG': 1,
            'JPN': 2,
            'KOR': 4,
            'TWN': 5,
            'VNM': 6
        }
        df = df[df['SurveyPublic'] == country_to_id[country]]
    elif region == "SEA":
        country_to_id = {
            'KHM': 1,
            'IDN': 2,
            'MYS': 3,
            'SGP': 5,
            'LKA': 6,
            'THA': 7
        }
        df = df[df['COUNTRY'] == country_to_id[country]]
    elif region == "IND":
        # India uses the full dataset
        pass
    
    cols = ['QRID', *cols]
    df = df[cols]
    return df


def compute_scores(model_vector: dict, human_vectors_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Hamming distance scores between model and human responses."""
    df = human_vectors_df.copy()
    qrids = df['QRID'].tolist()
    
    df = df.drop('QRID', axis=1)
    model_vector_df = pd.DataFrame([model_vector])
    
    hds = []
    for i, row in df.iterrows():
        temp_df = pd.concat([pd.DataFrame([row]), model_vector_df], ignore_index=True)
        temp_df = temp_df.replace(' ', pd.NA).dropna(axis=1)
        temp_df = pd.get_dummies(temp_df)
        
        hamming_dist = distance.hamming(
            temp_df.iloc[0].tolist(), temp_df.iloc[1].tolist()
        )
        hds.append(hamming_dist)
    
    return pd.DataFrame({'QRID': qrids, 'Hamming Distance': hds})


def main():
    parser = argparse.ArgumentParser(description="Evaluate models using HuggingFace or OpenRouter")
    parser.add_argument('--model', type=str, required=True, help='Model identifier')
    parser.add_argument('--country', type=str, required=True, help='Country code')
    parser.add_argument('--region', type=str, default='EA', choices=['EA', 'SEA', 'IND'], 
                        help='Region (EA, SEA, or IND)')
    parser.add_argument('--use-openrouter', action='store_true', 
                        help='Use OpenRouter API instead of HuggingFace')
    parser.add_argument('--openrouter-api-key', type=str, 
                        help='OpenRouter API key (or set OPENROUTER_API_KEY env var)')
    parser.add_argument('--questions-file', type=str, default='questions.json',
                        help='Questions JSON file')
    
    args = parser.parse_args()
    
    # Validate country based on region
    if args.region == "EA":
        allowed_countries = ['HKG', 'JPN', 'KOR', 'TWN', 'VNM']
    elif args.region == "SEA":
        allowed_countries = ['KHM', 'IDN', 'SGP', 'MYS', 'LKA', 'THA']
    elif args.region == "IND":
        allowed_countries = ['IND']
    
    if args.country not in allowed_countries:
        raise ValueError(f"For region {args.region}, expecting one of {allowed_countries}")
    
    # Load questions
    with open(args.questions_file) as f:
        questions = json.load(f)
    
    # Import region-specific question sets
    if args.region == "EA":
        import EA.model_questions as ea_questions
        cols = list(ea_questions.ASK_ALL_QUESTIONS)
        if args.country == 'JPN':
            cols.extend(ea_questions.ASK_JP_QUESTIONS)
        elif args.country == 'HKG':
            cols.extend(ea_questions.ASK_HK_QUESTIONS)
        elif args.country == 'KOR':
            cols.extend(ea_questions.ASK_SK_QUESTIONS)
        elif args.country == 'TWN':
            cols.extend(ea_questions.ASK_TW_QUESTIONS)
        elif args.country == 'VNM':
            cols.extend(ea_questions.ASK_VIET_QUESTIONS)
    elif args.region == "SEA":
        import SEA.model_questions as sea_questions
        cols = list(sea_questions.ASK_ALL_QUESTIONS)
        if args.country == 'KHM':
            cols.extend(sea_questions.ASK_CAM_QUESTIONS)
        elif args.country == 'IDN':
            cols.extend(sea_questions.ASK_ID_QUESTIONS)
        elif args.country == 'SGP':
            cols.extend(sea_questions.ASK_SG_QUESTIONS)
        elif args.country == 'MYS':
            cols.extend(sea_questions.ASK_MALAY_QUESTIONS)
        elif args.country == 'LKA':
            cols.extend(sea_questions.ASK_SL_QUESTIONS)
        elif args.country == 'THA':
            cols.extend(sea_questions.ASK_TH_QUESTIONS)
    elif args.region == "IND":
        # For India, we'll use all available questions from the JSON file
        cols = [q for q in questions.keys() if isinstance(questions[q], dict) and 
                'question' in questions[q] and 'options' in questions[q]]
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        args.model, 
        use_openrouter=args.use_openrouter,
        openrouter_api_key=args.openrouter_api_key
    )
    
    # Get model responses
    print(f"Evaluating model {args.model} for {args.country} in {args.region} region...")
    model_vector = evaluator.get_model_vector(cols, questions)
    
    # Get human responses
    human_vectors = get_human_vectors(cols, args.country, args.region)
    
    # Compute scores
    scores_df = compute_scores(model_vector, human_vectors)
    sorted_scores = scores_df.sort_values(by='Hamming Distance', ascending=True)
    sorted_scores = sorted_scores.head(100)
    
    # Save results
    output_prefix = "openrouter" if args.use_openrouter else "huggingface"
    model_safe_name = args.model.replace('/', '-').replace('\\', '-')
    output_file = f"{output_prefix}_{model_safe_name}_{args.country}_scores.csv"
    sorted_scores.to_csv(output_file, index=False)
    
    print(f"Results saved to {output_file}")
    print(f"Average Hamming Distance: {sorted_scores['Hamming Distance'].mean():.4f}")


if __name__ == '__main__':
    main()
