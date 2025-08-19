# OpenRouter Integration

This document describes how to use OpenRouter models with the LLMOpinions evaluation framework.

## Overview

OpenRouter provides API access to a wide variety of language models including GPT-4, Claude, Llama, and many others. This integration allows you to evaluate these models without needing to download and run them locally.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements_openrouter.txt
```

### 2. Get OpenRouter API Key

1. Go to [OpenRouter](https://openrouter.ai/)
2. Sign up for an account
3. Get your API key from the dashboard
4. Set it as an environment variable:

```bash
export OPENROUTER_API_KEY='your-api-key-here'
```

Or create a `.env` file:
```
OPENROUTER_API_KEY=your-api-key-here
```

### 3. Run Setup Script

```bash
python setup_openrouter.py
```

This script will:
- Check dependencies
- Help you configure your API key
- Test the connection
- Show available models
- Create example scripts

## Available Models

The integration supports many popular models with convenient short names:

| Short Name | Full OpenRouter Name |
|------------|---------------------|
| `gpt-4o` | `openai/gpt-4o` |
| `gpt-4` | `openai/gpt-4` |
| `gpt-3.5-turbo` | `openai/gpt-3.5-turbo` |
| `claude-3.5-sonnet` | `anthropic/claude-3.5-sonnet` |
| `claude-3-haiku` | `anthropic/claude-3-haiku` |
| `llama-3.1-8b` | `meta-llama/llama-3.1-8b-instruct` |
| `llama-3.1-70b` | `meta-llama/llama-3.1-70b-instruct` |
| `gemini-pro` | `google/gemini-pro` |
| `mistral-7b` | `mistralai/mistral-7b-instruct` |

You can use either the short name or the full OpenRouter model name.

## Usage

### Legacy Approach (EA, SEA, IND directories)

For the original evaluation approach, use `evaluate_model_openrouter.py`:

```bash
# Evaluate GPT-4 for Hong Kong in EA region
python evaluate_model_openrouter.py --model gpt-4o --country HKG --region EA --use-openrouter

# Evaluate Claude for Indonesia in SEA region  
python evaluate_model_openrouter.py --model claude-3.5-sonnet --country IDN --region SEA --use-openrouter

# Evaluate Llama for India
python evaluate_model_openrouter.py --model llama-3.1-8b --country IND --region IND --use-openrouter
```

### Translate Approach (Multilingual)

For the newer multilingual approach, use `Translate/evaluate_model_openrouter.py`:

```bash
cd Translate

# Evaluate GPT-4 for Singapore in English
python evaluate_model_openrouter.py --country sg --language en --model gpt-4o --use-openrouter

# Evaluate Claude for Indonesia in Indonesian
python evaluate_model_openrouter.py --country id --language id --model claude-3.5-sonnet --use-openrouter

# Evaluate Gemini for India in Hindi
python evaluate_model_openrouter.py --country ind --language hi --model gemini-pro --use-openrouter
```

## Command Line Arguments

### Legacy Approach (`evaluate_model_openrouter.py`)

- `--model`: Model identifier (required)
- `--country`: Country code (required)
- `--region`: Region (EA, SEA, or IND)
- `--use-openrouter`: Use OpenRouter API instead of HuggingFace
- `--openrouter-api-key`: OpenRouter API key (if not set as env var)
- `--questions-file`: Questions JSON file (default: questions.json)

### Translate Approach (`Translate/evaluate_model_openrouter.py`)

- `--country`: Country code (required)
- `--language`: Language code (required)  
- `--model`: Model identifier (optional for HuggingFace)
- `--use-openrouter`: Use OpenRouter API
- `--openrouter-api-key`: OpenRouter API key (if not set as env var)
- `--responses-file`: Responses CSV file (default: responses.csv)

## Supported Countries and Languages

### EA Region
Countries: `HKG` (Hong Kong), `JPN` (Japan), `KOR` (South Korea), `TWN` (Taiwan), `VNM` (Vietnam)

### SEA Region  
Countries: `KHM` (Cambodia), `IDN` (Indonesia), `MYS` (Malaysia), `SGP` (Singapore), `LKA` (Sri Lanka), `THA` (Thailand)

Languages: `en` (English), `km` (Khmer), `id` (Indonesian), `ma` (Malay), `zh` (Chinese), `ta` (Tamil), `si` (Sinhala), `th` (Thai)

### IND Region
Country: `IND` (India)
Languages: `en` (English), `hi` (Hindi)

## Output Files

Results are saved with descriptive filenames:

- Legacy: `openrouter_{model}_{country}_scores.csv`
- Translate: `openrouter_{model}_{country}_{language}_results.json`

## Comparison with HuggingFace Models

You can run the same evaluations with both OpenRouter and HuggingFace models to compare:

```bash
# HuggingFace model (runs locally)
python evaluate_model_openrouter.py --model meta-llama/Llama-3.2-1B-Instruct --country HKG --region EA

# OpenRouter model (API call)
python evaluate_model_openrouter.py --model llama-3.2-1b --country HKG --region EA --use-openrouter
```

## Cost Considerations

OpenRouter models have different pricing. Check [OpenRouter pricing](https://openrouter.ai/pricing) for current rates. The evaluation scripts are designed to minimize token usage while maintaining accuracy.

## Troubleshooting

### API Key Issues
```bash
# Check if API key is set
echo $OPENROUTER_API_KEY

# Test connection
python -c "from openrouter_client import OpenRouterClient; print('✓ API key works')"
```

### Rate Limits
If you hit rate limits, the scripts will retry with exponential backoff. You can also:
- Use smaller model variants
- Add delays between requests
- Use batch processing

### Model Not Available
If a model is not available:
- Check [OpenRouter models](https://openrouter.ai/models) for current availability
- Try an alternative model
- Check your account credits/limits

## Examples

See the `examples/` directory for complete example scripts:

- `examples/legacy_example.py`: Legacy approach example
- `examples/translate_example.py`: Translate approach example

## Integration Notes

The OpenRouter integration is designed to be a drop-in replacement for HuggingFace models:

1. **Same evaluation metrics**: Uses identical Hamming distance and Wasserstein distance calculations
2. **Same question formats**: Works with existing question JSON files
3. **Same output formats**: Produces comparable results files
4. **Backward compatible**: Can be used alongside existing HuggingFace evaluations

## Performance Tips

1. **Use appropriate models**: Larger models are more expensive but may be more accurate
2. **Batch similar evaluations**: Group evaluations by region/language to minimize setup overhead
3. **Monitor usage**: Check your OpenRouter dashboard for usage and costs
4. **Cache results**: Save evaluation results to avoid re-running expensive evaluations

## Contributing

When adding new OpenRouter models or features:

1. Update `OPENROUTER_MODELS` dictionary in `openrouter_client.py`
2. Add tests for new functionality
3. Update this documentation
4. Ensure backward compatibility with existing scripts
