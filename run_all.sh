#!/bin/bash

python run_all_evaluations.py --model meta-llama/Llama-3.2-1B-Instruct
rm Translate/*/cache*
python run_all_evaluations.py --model meta-llama/Llama-3.2-1B-Instruct --steering
rm Translate/*/cache*

python run_all_evaluations.py --model mistralai/Mistral-7B-Instruct-v0.3
rm Translate/*/cache*
python run_all_evaluations.py --model mistralai/Mistral-7B-Instruct-v0.3 --steering

python run_all_evaluations.py --model google/gemma-3-12b-it
rm Translate/*/cache*
python run_all_evaluations.py --model google/gemma-3-12b-it --steering
rm Translate/*/cache*
