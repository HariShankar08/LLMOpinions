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

# Add parent directory to path to import OpenRouter client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Translate.SEA.openrouter_client import OpenRouterClient, OPENROUTER_MODELS

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
                 openrouter_api_key: str = None, language: str = 'en', country: str = 'ca'):
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
        self.num_reasoning_tokens = 100
        
        if use_openrouter:
            self._setup_openrouter_model(model_name, openrouter_api_key)
        else:
            self._setup_huggingface_model(model_name)
    
    def _setup_openrouter_model(self, model_name: str, api_key: str = None):
        """Setup OpenRouter model."""
        # Convert short name to full OpenRouter model name if needed
        if model_name in OPENROUTER_MODELS:
            openrouter_model_name = OPENROUTER_MODELS[model_name]
        else:
            openrouter_model_name = model_name
        
        self.client = OpenRouterClient(api_key)
        self.model_name = openrouter_model_name
        
        # Set up tokenizer for compatibility
        if HF_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        else:
            self.tokenizer = None
        
        self.device = torch.device("cpu")  # API models don't use local device
        print(f"Initialized OpenRouter model: {openrouter_model_name}")
    
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
    
    def get_question_distribution(self, df: pd.DataFrame, question: str) -> pd.Series:
        """Get the distribution of answers for a specific question."""
        question_data = df[question]
        question_data = question_data.astype(str)
        question_data = question_data[question_data.notna() & (question_data != "") & (question_data.str.strip() != "")]
        return question_data.value_counts(normalize=True)
    
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
        """Make model distribution using OpenRouter API."""
        prompt = self.get_prompt(question, questions_dict)
        system_prompt = self.get_system_prompt(steering=False)
        reasoning_start_prompt = self.get_reasoning_start_prompt()
        
        # Create messages for chat completion
        messages = [
            {'role': "system", 'content': system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reasoning_start_prompt}
        ]
        
        try:
            # First, generate reasoning tokens
            reasoning_response = self.client.get_text_response(
                messages, 
                self.model_name,
                max_tokens=self.num_reasoning_tokens,
                temperature=0.1
            )
            
            # Add reasoning to messages and ask for final answer
            messages.append({"role": "assistant", "content": reasoning_start_prompt + " " + reasoning_response})
            messages.append({"role": "user", "content": "Answer:"})
            
            # Get probabilities for each option
            options = list(questions_dict[question]['options'].keys())
            option_probs = self.client.get_logits_for_options(messages, self.model_name, options)
            
            return pd.Series(option_probs)
            
        except Exception as e:
            print(f"Error getting OpenRouter distribution: {e}")
            # Fallback to uniform distribution
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
            return self.make_model_distribution_openrouter(question, questions)
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
    args = parser.parse_args()
    SECONDARY_FILTER_VAR = args.secondary_filter_var
    SECONDARY_FILTER_VALUE = args.secondary_filter_value
    
    # Map country codes for different regions
    country_ids = {
        # SEA region
        'ca': 1, 'id': 2, 'ms': 3, 'sg': 5, 'sl': 6, 'th': 7,
        # India region
        'ind': None  # India doesn't use country filtering
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
        responses = responses[responses['COUNTRY'] == country_id]
    
    if SECONDARY_FILTER_VAR is not None and SECONDARY_FILTER_VALUE is not None:
        responses = responses[responses[SECONDARY_FILTER_VAR] == SECONDARY_FILTER_VALUE]
    
    # Initialize evaluator
    evaluator = TranslateModelEvaluator(
        model_name=args.model,
        use_openrouter=args.use_openrouter,
        openrouter_api_key=args.openrouter_api_key,
        language=args.language,
        country=args.country
    )
    
    scores = []
    print(f"Evaluating {args.country}_{args.language} with {'OpenRouter' if args.use_openrouter else 'HuggingFace'} model...")
    
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
        'individual_scores': scores
    }
    
    output_prefix = "openrouter" if args.use_openrouter else "huggingface"
    model_name = (args.model or "default").replace('/', '-').replace('\\', '-')
    output_file = f"{output_prefix}_{model_name}_{args.country}_{args.language}_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
