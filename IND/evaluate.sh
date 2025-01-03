#!/bin/bash

python evaluate_model.py --model google/gemma-2-9b > gemma2.txt
python evaluate_model.py --model google/gemma-2-9b-it > gemma2it.txt
python evaluate_model.py --model mistralai/Mistral-7B-v0.1 > mistralv0.1.txt
python evaluate_model.py --model mistralai/Mistral-7B-Instruct-v0.1 > mistralv0.1it.txt
python evaluate_model.py --model mistralai/Mistral-7B-v0.3 > mistralv0.3.txt
python evaluate_model.py --model mistralai/Mistral-7B-Instruct-v0.3 > mistralv0.3it.txt
python evaluate_model.py --model meta-llama/Llama-3.1-8B > llama3.1.txt
python evaluate_model.py --model meta-llama/Llama-3.1-8B-Instruct > llama3.1it.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B > llama3.2.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B-Instruct > llama3.2it.txt