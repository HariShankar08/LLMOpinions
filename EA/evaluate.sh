#!/bin/bash

# Evaluate google/gemma-2-9b for the new set of countries
python evaluate_model.py --model google/gemma-2-9b --country HKG > HKG.txt
python evaluate_model.py --model google/gemma-2-9b --country JPN > JPN.txt
python evaluate_model.py --model google/gemma-2-9b --country KOR > KOR.txt
python evaluate_model.py --model google/gemma-2-9b --country TWN > TWN.txt
python evaluate_model.py --model google/gemma-2-9b --country VNM > VNM.txt

mkdir gemma-2-9b
mv HKG.txt gemma-2-9b
mv JPN.txt gemma-2-9b
mv KOR.txt gemma-2-9b
mv TWN.txt gemma-2-9b
mv VNM.txt gemma-2-9b

# Evaluate google/gemma-2-9b-it for the new set of countries
python evaluate_model.py --model google/gemma-2-9b-it --country HKG > HKG.txt
python evaluate_model.py --model google/gemma-2-9b-it --country JPN > JPN.txt
python evaluate_model.py --model google/gemma-2-9b-it --country KOR > KOR.txt
python evaluate_model.py --model google/gemma-2-9b-it --country TWN > TWN.txt
python evaluate_model.py --model google/gemma-2-9b-it --country VNM > VNM.txt

mkdir gemma-2-9b-it
mv HKG.txt gemma-2-9b-it
mv JPN.txt gemma-2-9b-it
mv KOR.txt gemma-2-9b-it
mv TWN.txt gemma-2-9b-it
mv VNM.txt gemma-2-9b-it


# Evaluate mistralai/Mistral-7B-v0.1 for the new set of countries
python evaluate_model.py --model mistralai/Mistral-7B-v0.1 --country HKG > HKG.txt
python evaluate_model.py --model mistralai/Mistral-7B-v0.1 --country JPN > JPN.txt
python evaluate_model.py --model mistralai/Mistral-7B-v0.1 --country KOR > KOR.txt
python evaluate_model.py --model mistralai/Mistral-7B-v0.1 --country TWN > TWN.txt
python evaluate_model.py --model mistralai/Mistral-7B-v0.1 --country VNM > VNM.txt

mkdir Mistral-7B-v0.1
mv HKG.txt Mistral-7B-v0.1
mv JPN.txt Mistral-7B-v0.1
mv KOR.txt Mistral-7B-v0.1
mv TWN.txt Mistral-7B-v0.1
mv VNM.txt Mistral-7B-v0.1

# Evaluate mistralai/Mistral-7B-Instruct-v0.1 for the new set of countries
python evaluate_model.py --model mistralai/Mistral-7B-Instruct-v0.1 --country HKG > HKG.txt
python evaluate_model.py --model mistralai/Mistral-7B-Instruct-v0.1 --country JPN > JPN.txt
python evaluate_model.py --model mistralai/Mistral-7B-Instruct-v0.1 --country KOR > KOR.txt
python evaluate_model.py --model mistralai/Mistral-7B-Instruct-v0.1 --country TWN > TWN.txt
python evaluate_model.py --model mistralai/Mistral-7B-Instruct-v0.1 --country VNM > VNM.txt

mkdir Mistral-7B-Instruct-v0.1
mv HKG.txt Mistral-7B-Instruct-v0.1
mv JPN.txt Mistral-7B-Instruct-v0.1
mv KOR.txt Mistral-7B-Instruct-v0.1
mv TWN.txt Mistral-7B-Instruct-v0.1
mv VNM.txt Mistral-7B-Instruct-v0.1

# Evaluate mistralai/Mistral-7B-v0.3 for the new set of countries
python evaluate_model.py --model mistralai/Mistral-7B-v0.3 --country HKG > HKG.txt
python evaluate_model.py --model mistralai/Mistral-7B-v0.3 --country JPN > JPN.txt
python evaluate_model.py --model mistralai/Mistral-7B-v0.3 --country KOR > KOR.txt
python evaluate_model.py --model mistralai/Mistral-7B-v0.3 --country TWN > TWN.txt
python evaluate_model.py --model mistralai/Mistral-7B-v0.3 --country VNM > VNM.txt

mkdir Mistral-7B-v0.3
mv HKG.txt Mistral-7B-v0.3
mv JPN.txt Mistral-7B-v0.3
mv KOR.txt Mistral-7B-v0.3
mv TWN.txt Mistral-7B-v0.3
mv VNM.txt Mistral-7B-v0.3

# Evaluate mistralai/Mistral-7B-Instruct-v0.3 for the new set of countries
python evaluate_model.py --model mistralai/Mistral-7B-Instruct-v0.3 --country HKG > HKG.txt
python evaluate_model.py --model mistralai/Mistral-7B-Instruct-v0.3 --country JPN > JPN.txt
python evaluate_model.py --model mistralai/Mistral-7B-Instruct-v0.3 --country KOR > KOR.txt
python evaluate_model.py --model mistralai/Mistral-7B-Instruct-v0.3 --country TWN > TWN.txt
python evaluate_model.py --model mistralai/Mistral-7B-Instruct-v0.3 --country VNM > VNM.txt

mkdir Mistral-7B-Instruct-v0.3
mv HKG.txt Mistral-7B-Instruct-v0.3
mv JPN.txt Mistral-7B-Instruct-v0.3
mv KOR.txt Mistral-7B-Instruct-v0.3
mv TWN.txt Mistral-7B-Instruct-v0.3
mv VNM.txt Mistral-7B-Instruct-v0.3

# Evaluate meta-llama/Llama-3.1-8B for the new set of countries
python evaluate_model.py --model meta-llama/Llama-3.1-8B --country HKG > HKG.txt
python evaluate_model.py --model meta-llama/Llama-3.1-8B --country JPN > JPN.txt
python evaluate_model.py --model meta-llama/Llama-3.1-8B --country KOR > KOR.txt
python evaluate_model.py --model meta-llama/Llama-3.1-8B --country TWN > TWN.txt
python evaluate_model.py --model meta-llama/Llama-3.1-8B --country VNM > VNM.txt

mkdir Llama-3.1-8B
mv HKG.txt Llama-3.1-8B
mv JPN.txt Llama-3.1-8B
mv KOR.txt Llama-3.1-8B
mv TWN.txt Llama-3.1-8B
mv VNM.txt Llama-3.1-8B

# Evaluate meta-llama/Llama-3.1-8B-Instruct for the new set of countries
python evaluate_model.py --model meta-llama/Llama-3.1-8B-Instruct --country HKG > HKG.txt
python evaluate_model.py --model meta-llama/Llama-3.1-8B-Instruct --country JPN > JPN.txt
python evaluate_model.py --model meta-llama/Llama-3.1-8B-Instruct --country KOR > KOR.txt
python evaluate_model.py --model meta-llama/Llama-3.1-8B-Instruct --country TWN > TWN.txt
python evaluate_model.py --model meta-llama/Llama-3.1-8B-Instruct --country VNM > VNM.txt

mkdir Llama-3.1-8B-Instruct
mv HKG.txt Llama-3.1-8B-Instruct
mv JPN.txt Llama-3.1-8B-Instruct
mv KOR.txt Llama-3.1-8B-Instruct
mv TWN.txt Llama-3.1-8B-Instruct
mv VNM.txt Llama-3.1-8B-Instruct

# Evaluate meta-llama/Llama-3.2-3B for the new set of countries
python evaluate_model.py --model meta-llama/Llama-3.2-3B --country HKG > HKG.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B --country JPN > JPN.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B --country KOR > KOR.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B --country TWN > TWN.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B --country VNM > VNM.txt

mkdir Llama-3.2-3B
mv HKG.txt Llama-3.2-3B
mv JPN.txt Llama-3.2-3B
mv KOR.txt Llama-3.2-3B
mv TWN.txt Llama-3.2-3B
mv VNM.txt Llama-3.2-3B

# Evaluate meta-llama/Llama-3.2-3B-Instruct for the new set of countries
python evaluate_model.py --model meta-llama/Llama-3.2-3B-Instruct --country HKG > HKG.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B-Instruct --country JPN > JPN.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B-Instruct --country KOR > KOR.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B-Instruct --country TWN > TWN.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B-Instruct --country VNM > VNM.txt

mkdir Llama-3.2-3B-Instruct
mv HKG.txt Llama-3.2-3B-Instruct
mv JPN.txt Llama-3.2-3B-Instruct
mv KOR.txt Llama-3.2-3B-Instruct
mv TWN.txt Llama-3.2-3B-Instruct
mv VNM.txt Llama-3.2-3B-Instruct
