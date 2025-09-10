"""
Enhanced evaluation script for Translate directory that supports both HuggingFace and OpenRouter models.
This version supports the chain-of-thought reasoning approach with multilingual capabilities.
"""

import pandas as pd
import json
import torch
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import argparse
import os
import sys
from openai import OpenAI
import math

# Initialize client using OPENAI_API_KEY if available; otherwise fall back to OPENROUTER_API_KEY
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')

if OPENAI_API_KEY:
    OPENROUTER_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
elif OPENROUTER_API_KEY:
    OPENROUTER_CLIENT = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
else:
    OPENROUTER_CLIENT = None

# Determine which token limit parameter to use based on provider
USE_MAX_COMPLETION_TOKENS = bool(OPENAI_API_KEY)

# Try to import HuggingFace components
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    HF_AVAILABLE = True
except ImportError:
    print("Warning: HuggingFace transformers not available. Only OpenRouter models will work.")
    HF_AVAILABLE = False

# Set all seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if HF_AVAILABLE:
    set_seed(42)


class TranslateModelEvaluator:
    """Model evaluator for the Translate directory approach with multilingual support."""
    
    def __init__(self, model_name: str = None, use_openrouter: bool = False, 
                 openrouter_api_key: str = None, language: str = 'en', country: str = 'ca',
                 temperature: float = 0.0, top_p: float = None, top_logprobs: int = None,
                 num_reasoning_tokens: int = 100, seed: int = None, repeats: int = 1):
        """
        Initialize the model evaluator.
        
        Args:
            model_name: Model identifier
            use_openrouter: Whether to use OpenRouter API
            openrouter_api_key: OpenRouter API key
            language: Language code for prompts
            country: Country code for prompts
        """
        self.use_openrouter = use_openrouter
        self.language = language
        self.country = country
        self.num_reasoning_tokens = int(num_reasoning_tokens)
        self.temperature = float(temperature) if temperature is not None else 0.0
        self.top_p = None if top_p is None else float(top_p)
        self.top_logprobs = None if top_logprobs is None else int(top_logprobs)
        self.seed = None if seed is None else int(seed)
        self.repeats = max(1, int(repeats))
        
        if use_openrouter:
            self._setup_openrouter_model(model_name, openrouter_api_key)
        else:
            self._setup_huggingface_model(model_name)
    
    def _setup_openrouter_model(self, model_name: str, api_key: str = None):
        """Setup OpenRouter/OpenAI client model."""
        # Check for API keys at runtime (not import time)
        runtime_openai_key = os.environ.get('OPENAI_API_KEY')
        runtime_openrouter_key = os.environ.get('OPENROUTER_API_KEY')
        
        if runtime_openai_key:
            self.client = OpenAI(api_key=runtime_openai_key)
            print(f"Using OpenAI API with model: {model_name}")
        elif runtime_openrouter_key:
            self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=runtime_openrouter_key)
            print(f"Using OpenRouter API with model: {model_name}")
        elif api_key:
            self.client = OpenAI(api_key=api_key)
            print(f"Using provided API key with model: {model_name}")
        else:
            raise RuntimeError("No API key found. Set OPENAI_API_KEY or OPENROUTER_API_KEY")
            
        self.model_name = model_name
        
        # Skip tokenizer for API-based models - not needed for OpenAI/OpenRouter
        self.tokenizer = None
        
        self.device = torch.device("cpu")  # API models don't use local device
    
    def _setup_huggingface_model(self, model_name: str):
        """Setup HuggingFace model."""
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace transformers not available. Install with: pip install transformers")
        
        if model_name is None:
            model_name = "meta-llama/Llama-3.2-1B-Instruct"
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        
        self.model = self.model.to(self.device)
        print(f"Initialized HuggingFace model: {model_name} on {self.device}")
        
        # Optionally reseed if provided (affects only local HF models)
        # Note: This does not control remote API sampling.
        try:
            if hasattr(self, 'seed') and self.seed is not None and HF_AVAILABLE:
                set_seed(self.seed)
        except Exception:
            pass
    
    def get_question_distribution(self, df: pd.DataFrame, question: str) -> pd.Series:
        """Get weighted, cleaned distribution of answers for a specific question."""
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

        return weighted_counts / total_weight
    
    def get_prompt(self, question: str, questions: dict) -> str:
        """Get the prompt based on the question and its context."""
        qdict = questions[question]
        prompt = f'Question: {qdict["question"]}\n'
        for option in sorted(qdict['options'], key=lambda x: int(x)):
            prompt += f'{option}: {qdict["options"][option]}\n'
        return prompt
    
    def get_system_prompt(self, steering: bool = False) -> str:
        """Get system prompt based on language and country."""
        country_dict_en = {
            'ca': 'Cambodia', 'id': 'Indonesia', 'ms': 'Malaysia',
            'sg': 'Singapore', 'sl': 'Sri Lanka', 'th': 'Thailand'
        }
        
        country_dict_km = {
            'ca': 'កម្ពុជា', 'id': 'ឥណ្ឌូណេស៊ី', 'ms': 'ម៉ាឡេស៊ី',
            'sg': 'សិង្ហបុរី', 'sl': 'ស្រីលង្កា', 'th': 'ថៃ'
        }
        
        country_dict_zh = {
            'ca': '柬埔寨', 'id': '印度尼西亚', 'ms': '马来西亚',
            'sg': '新加坡', 'sl': '斯里兰卡', 'th': '泰国'
        }
        
        country_dict_ma = {
            'ca': 'Kemboja', 'id': 'Indonesia', 'ms': 'Malaysia',
            'sg': 'Singapura', 'sl': 'Sri Lanka', 'th': 'Thailand'
        }
        
        country_dict_id = {
            'ca': 'Kamboja', 'id': 'Indonesia', 'ms': 'Malaysia',
            'sg': 'Singapura', 'sl': 'Sri Lanka', 'th': 'Thailand'
        }
        
        country_dict_ta = {
            'ca': 'கம்போடியா', 'id': 'இந்தோனேசியா', 'ms': 'மலேசியா',
            'sg': 'சிங்கப்பூர்', 'sl': 'ஸ்ரீலங்கா', 'th': 'தாய்லாந்து'
        }
        
        country_dict_si = {
            'ca': 'කම්බෝජ', 'id': 'ඉන්දෝනීසිය', 'ms': 'මාලේසියාව',
            'sg': 'සින්ගප්පූර්', 'sl': 'ශ්‍රී ලංකාව', 'th': 'තායිලන්තය'
        }
        
        country_dict_th = {
            'ca': 'កម្ពុជា', 'id': 'ឥណ្ឌូណេស៊ី', 'ms': 'ម៉ាឡេស៊ី',
            'sg': 'សិង្ហបុរី', 'sl': 'ស្រីលង្កា', 'th': 'ថៃ'
        }
        
        # For India
        if self.country == 'ind':
            if self.language == 'en':
                return 'Answer the following question, after providing a brief explanation within 50 words.'
            elif self.language == 'hi':
                return 'निम्नलिखित प्रश्न का उत्तर दीजिए, तथा 50 शब्दों में संक्षिप्त व्याख्या दीजिए।'
        
        # For other countries
        if self.language == 'en':
            return 'Answer the following question, after providing a brief explanation within 50 words.'
        elif self.language == 'km':
            return 'សូមឆ្លើយសំណួរ បន្ទាប់ពីផ្តល់ការពន្យល់ខ្លីក្នុងរយៈពេល 50 ពាក្យ។'
        elif self.language == 'zh':
            return '請回答以下問題，並在50個字以內提供簡要說明。'
        elif self.language == 'ma':
            return 'Sila jawab soalan berikut, selepas memberikan penjelasan ringkas dalam 50 patah perkataan.'
        elif self.language == 'id':
            return 'Silakan jawab pertanyaan berikut, setelah memberikan penjelasan singkat dalam 50 kata.'
        elif self.language == 'ta':
            return 'தயவுசெய்து கீழ்காணும் கேள்விக்கு பதிலளிக்கவும், 50 வார்த்தைகளுக்குள் சுருக்கமான விளக்கத்தை வழங்கவும்.'
        elif self.language == 'si':
            return 'කරුණාකර පහත ප්‍රශ්නයට පිළිතුරු දෙන්න, වචන 50 කින් කෙටි විස්තරයක් ලබා දී.'
        elif self.language == 'th':
            return 'กรุณาตอบคำถามต่อไปนี้ หลังจากให้คำอธิบายสั้น ๆ ภายใน 50 คำ'
        elif self.language == 'hi':
            return 'निम्नलिखित प्रश्न का उत्तर दीजिए, तथा 50 शब्दों में संक्षिप्त व्याख्या दीजिए।'
        
        if steering:
            raise NotImplementedError("Steering prompts not yet implemented.")
        
        return 'Answer the following question, providing a brief explanation within 50 words.'
    
    def get_reasoning_start_prompt(self) -> str:
        """Get reasoning start prompt based on language."""
        if self.language == 'en':
            return 'Let\'s think step by step.'
        elif self.language == 'km':
            return 'មកគិតជាដំណាក់កាលៗគ្នា។'
        elif self.language == 'zh':
            return '让我们一步一步思考。'
        elif self.language == 'ma':
            return 'Sila fikir langkah demi langkah.'
        elif self.language == 'id':
            return 'Silakan berpikir langkah demi langkah.'
        elif self.language == 'ta':
            return 'யோசித்து கேள்விக்கு பதிலளிப்போம்.'
        elif self.language == 'si':
            return 'අපි ඒ ගැන පියවරෙන් පියවර සිතමු.'
        elif self.language == 'th':
            return 'กรุณาคิดทีละขั้นตอน'
        elif self.language == 'hi':
            return 'चलिए, चरणबद्ध तरीके से सोचते हैं।'
        
        return 'Let\'s think step by step.'
    
    def make_model_distribution_openrouter(self, question: str, questions_dict: dict) -> pd.Series:
        """Make model distribution using OpenAI/OpenRouter Chat Completions with logprobs."""
        prompt = self.get_prompt(question, questions_dict)
        system_prompt = self.get_system_prompt(steering=False)
        reasoning_start_prompt = self.get_reasoning_start_prompt()

        messages = [
            {'role': "system", 'content': system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reasoning_start_prompt}
        ]

        try:
            token_kwargs_reasoning = {"max_completion_tokens": self.num_reasoning_tokens} if USE_MAX_COMPLETION_TOKENS else {"max_tokens": self.num_reasoning_tokens}
            reasoning_extra = {}
            if self.top_p is not None:
                reasoning_extra["top_p"] = self.top_p
            reasoning_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                **token_kwargs_reasoning,
                **reasoning_extra
            )
            reasoning = reasoning_response.choices[0].message.content.strip()
            messages.append({"role": "assistant", "content": reasoning})
            messages.append({"role": "user", "content": "Selected Option (reply with only the option key, e.g., 1):"})

            token_kwargs_answer = {"max_completion_tokens": 1} if USE_MAX_COMPLETION_TOKENS else {"max_tokens": 1}
            answer_extra = {}
            if self.top_p is not None:
                answer_extra["top_p"] = self.top_p
            tlp = self.top_logprobs if self.top_logprobs is not None else (5 if USE_MAX_COMPLETION_TOKENS else 10)
            answer_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                logprobs=True,
                top_logprobs=tlp,
                **token_kwargs_answer,
                **answer_extra
            )

            # Parse top_logprobs
            if (answer_response.choices[0].logprobs is None or
                not answer_response.choices[0].logprobs.content or
                not answer_response.choices[0].logprobs.content[0].top_logprobs):
                # Uniform fallback
                options = list(questions_dict[question]['options'].keys())
                uniform_prob = 1.0 / len(options)
                return pd.Series({opt: uniform_prob for opt in options})

            logprobs = answer_response.choices[0].logprobs.content[0].top_logprobs

            model_distribution = {}
            for option in questions_dict[question]['options']:
                prob = 0.0
                for item in logprobs:
                    if item.token.strip().lower() == option.strip().lower():
                        prob = float(math.exp(item.logprob))
                        break
                model_distribution[option] = prob

            total = sum(model_distribution.values())
            if total > 0:
                for k in model_distribution:
                    model_distribution[k] /= total
            else:
                options = list(questions_dict[question]['options'].keys())
                uniform_prob = 1.0 / len(options)
                return pd.Series({opt: uniform_prob for opt in options})

            return pd.Series(model_distribution)

        except Exception as e:
            print(f"Error getting OpenRouter distribution: {e}")
            options = list(questions_dict[question]['options'].keys())
            uniform_prob = 1.0 / len(options)
            return pd.Series({opt: uniform_prob for opt in options})
    
    def make_model_distribution_huggingface(self, question: str, questions_dict: dict, chat_model: bool = True) -> pd.Series:
        """Make model distribution using HuggingFace model."""
        prompt = self.get_prompt(question, questions_dict)
        system_prompt = self.get_system_prompt(steering=False)
        reasoning_start_prompt = self.get_reasoning_start_prompt()
        
        if chat_model:
            messages = [
                {'role': "system", 'content': system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reasoning_start_prompt}
            ]
            inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", return_dict=True)
        else:
            inputs = self.tokenizer(f"{system_prompt}\n{prompt}\n{reasoning_start_prompt}\n", return_tensors="pt", return_dict=True)
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Generate reasoning tokens
            outputs = self.model.generate(**inputs, max_new_tokens=self.num_reasoning_tokens, do_sample=False)
            
            # Decode the output
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated += '\nAnswer:'
            
            # Get logits for final answer
            inputs = self.tokenizer(generated, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            logits = outputs.logits
            last_token_logits = logits[:, -1, :]
            probs = last_token_logits.softmax(dim=-1)
        
        return self._make_model_distribution_from_logits(probs, question, questions_dict)
    
    def _make_model_distribution_from_logits(self, probs: torch.Tensor, question: str, questions_dict: dict) -> pd.Series:
        """Convert logits to model distribution."""
        model_distribution = {}
        
        for option in questions_dict[question]['options']:
            option_token_id = self.tokenizer.encode(option, add_special_tokens=False)[0]
            model_distribution[option] = probs[0, option_token_id].item()
        
        return pd.Series(model_distribution)
    
    def get_model_distribution(self, df: pd.DataFrame, question: str, questions: dict, chat_model: bool = True) -> pd.Series:
        """Get model distribution for a question."""
        if self.use_openrouter:
            # Average over repeated generations for stochastic settings
            agg = None
            for _ in range(self.repeats):
                dist = self.make_model_distribution_openrouter(question, questions)
                agg = dist if agg is None else (agg + dist)
            return agg / float(self.repeats)
        else:
            return self.make_model_distribution_huggingface(question, questions, chat_model)
    
    def compare_distributions(self, d1: pd.Series, d2: pd.Series, num_options: int) -> float:
        """Compute the Wasserstein distance between two distributions."""
        if not d1.index.equals(d2.index):
            d1 = d1.reindex(d2.index, fill_value=0)
        
        wd = wasserstein_distance(d1, d2)
        if num_options == 1:
            return 1  # There is no diversity if there is only one option.
        return 1 - (wd / (num_options - 1))


def main():
    parser = argparse.ArgumentParser(description="Evaluate models with multilingual support")
    parser.add_argument('--country', type=str, required=True, help='Country code')
    parser.add_argument('--language', type=str, required=True, help='Language code')
    parser.add_argument('--model', type=str, help='Model identifier (optional for HuggingFace)')
    parser.add_argument('--use-openrouter', action='store_true', help='Use OpenRouter API')
    parser.add_argument('--openrouter-api-key', type=str, help='OpenRouter API key')
    parser.add_argument('--responses-file', type=str, default='responses.csv', help='Responses CSV file')
    parser.add_argument('--secondary-filter-var', type=str, default=None)
    parser.add_argument('--secondary-filter-value', type=str, default=None)
    # Generation/rigor controls
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--top_logprobs', type=int, default=None)
    parser.add_argument('--num_reasoning_tokens', type=int, default=100)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--repeats', type=int, default=1, help='Repeats to average per question')
    args = parser.parse_args()
    SECONDARY_FILTER_VAR = args.secondary_filter_var
    SECONDARY_FILTER_VALUE = args.secondary_filter_value
    
    # Map country codes for different regions
    country_ids = {
        # SEA region
        'ca': 1, 'id': 2, 'ms': 3, 'sg': 5, 'sl': 6, 'th': 7,
        # India region
        'ind': None,  # India doesn't use country filtering
        # EA region - using SurveyPublic column
        'hk': 1, 'jp': 2, 'ko': 4, 'tw': 5, 'vi': 6
    }
    
    # Load questions file
    questions_file = f'{args.country}_{args.language}.json'
    if not os.path.exists(questions_file):
        print(f"Questions file {questions_file} not found!")
        return
    
    with open(questions_file) as f:
        questions = json.load(f)
    
    # Load responses
    responses = pd.read_csv(args.responses_file)
    
    # Filter by country if needed
    if args.country in country_ids and country_ids[args.country] is not None:
        country_id = country_ids[args.country]
        # EA region uses SurveyPublic column, others use COUNTRY column
        if args.country in ['hk', 'jp', 'ko', 'tw', 'vi']:
            responses = responses[responses['SurveyPublic'] == country_id]
        else:
            responses = responses[responses['COUNTRY'] == country_id]
    
    if SECONDARY_FILTER_VAR is not None and SECONDARY_FILTER_VALUE is not None:
        responses = responses[responses[SECONDARY_FILTER_VAR] == SECONDARY_FILTER_VALUE]
    
    # Initialize evaluator
    # Auto-enable OpenAI/OpenRouter if API key is present (prioritize over HuggingFace)
    env_api_key = os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENROUTER_API_KEY')
    use_or = bool(env_api_key) or args.use_openrouter  # Default to API if key exists

    evaluator = TranslateModelEvaluator(
        model_name=args.model,
        use_openrouter=use_or,
        openrouter_api_key=(args.openrouter_api_key or env_api_key),
        language=args.language,
        country=args.country,
        temperature=args.temperature,
        top_p=args.top_p,
        top_logprobs=args.top_logprobs,
        num_reasoning_tokens=args.num_reasoning_tokens,
        seed=args.seed,
        repeats=args.repeats
    )
    
    scores = []
    print(f"Evaluating {args.country}_{args.language} with {'OpenRouter' if use_or else 'HuggingFace'} model...")
    
    for question in tqdm(questions):
        if question in ['COUNTRY', 'QRID', 'weight', 'QMLangRec']:
            continue
        if 'question' in questions[question] and 'options' in questions[question]:
            # Get human distribution
            qd1 = evaluator.get_question_distribution(responses, question)
            
            # Get model distribution
            qd2 = evaluator.get_model_distribution(responses, question, questions)
            
            # Compare distributions
            score = evaluator.compare_distributions(qd1, qd2, num_options=len(responses[question].unique()))
            scores.append(score)
            print(f"Question {question}: Score = {score:.4f}")
    scores = [s for s in scores if s != 1]
    print('=' * 20)
    average_score = sum(scores) / len(scores) if scores else 0
    print(f'Average Representativeness: {average_score:.4f}')
    
    # Save results
    results = {
        'country': args.country,
        'language': args.language,
        'model': args.model if args.model else 'default',
        'use_openrouter': args.use_openrouter,
        'average_representativeness': average_score,
        'individual_scores': scores,
        'generation_params': {
            'temperature': args.temperature,
            'top_p': args.top_p,
            'top_logprobs': args.top_logprobs,
            'num_reasoning_tokens': args.num_reasoning_tokens,
            'seed': args.seed,
            'repeats': args.repeats
        }
    }
    
    output_prefix = "openrouter" if args.use_openrouter else "huggingface"
    model_name = (args.model or "default").replace('/', '-').replace('\\', '-')
    output_file = f"{output_prefix}_{model_name}_{args.country}_{args.language}_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
