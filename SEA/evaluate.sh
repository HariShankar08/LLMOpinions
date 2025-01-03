#!/bin/bash

python evaluate_model.py --model google/gemma-2-9b --country MYS > MYS.txt
python evaluate_model.py --model google/gemma-2-9b --country SGP > SGP.txt
python evaluate_model.py --model google/gemma-2-9b --country LKA > LKA.txt
python evaluate_model.py --model google/gemma-2-9b --country THA > THA.txt

mkdir gemma-2-9b
mv KHM.txt gemma-2-9b
mv IDN.txt gemma-2-9b
mv MYS.txt gemma-2-9b
mv SGP.txt gemma-2-9b
mv LKA.txt gemma-2-9b
mv THA.txt gemma-2-9b

python evaluate_model.py --model google/gemma-2-9b-it --country KHM > KHM.txt
python evaluate_model.py --model google/gemma-2-9b-it --country IDN > IDN.txt
python evaluate_model.py --model google/gemma-2-9b-it --country MYS > MYS.txt
python evaluate_model.py --model google/gemma-2-9b-it --country SGP > SGP.txt
python evaluate_model.py --model google/gemma-2-9b-it --country LKA > LKA.txt
python evaluate_model.py --model google/gemma-2-9b-it --country THA > THA.txt

mkdir gemma-2-9b-it
mv KHM.txt gemma-2-9b-it
mv IDN.txt gemma-2-9b-it
mv MYS.txt gemma-2-9b-it
mv SGP.txt gemma-2-9b-it
mv LKA.txt gemma-2-9b-it
mv THA.txt gemma-2-9b-it

python evaluate_model.py --model mistralai/Mistral-7B-v0.1 --country KHM > KHM.txt
python evaluate_model.py --model mistralai/Mistral-7B-v0.1 --country IDN > IDN.txt
python evaluate_model.py --model mistralai/Mistral-7B-v0.1 --country MYS > MYS.txt
python evaluate_model.py --model mistralai/Mistral-7B-v0.1 --country SGP > SGP.txt
python evaluate_model.py --model mistralai/Mistral-7B-v0.1 --country LKA > LKA.txt
python evaluate_model.py --model mistralai/Mistral-7B-v0.1 --country THA > THA.txt

mkdir Mistral-7B-v0.1
mv KHM.txt Mistral-7B-v0.1
mv IDN.txt Mistral-7B-v0.1
mv MYS.txt Mistral-7B-v0.1
mv SGP.txt Mistral-7B-v0.1
mv LKA.txt Mistral-7B-v0.1
mv THA.txt Mistral-7B-v0.1

python evaluate_model.py --model mistralai/Mistral-7B-Instruct-v0.1 --country KHM > KHM.txt
python evaluate_model.py --model mistralai/Mistral-7B-Instruct-v0.1 --country IDN > IDN.txt
python evaluate_model.py --model mistralai/Mistral-7B-Instruct-v0.1 --country MYS > MYS.txt
python evaluate_model.py --model mistralai/Mistral-7B-Instruct-v0.1 --country SGP > SGP.txt
python evaluate_model.py --model mistralai/Mistral-7B-Instruct-v0.1 --country LKA > LKA.txt
python evaluate_model.py --model mistralai/Mistral-7B-Instruct-v0.1 --country THA > THA.txt

mkdir Mistral-7B-Instruct-v0.1
mv KHM.txt Mistral-7B-Instruct-v0.1
mv IDN.txt Mistral-7B-Instruct-v0.1
mv MYS.txt Mistral-7B-Instruct-v0.1
mv SGP.txt Mistral-7B-Instruct-v0.1
mv LKA.txt Mistral-7B-Instruct-v0.1
mv THA.txt Mistral-7B-Instruct-v0.1

python evaluate_model.py --model mistralai/Mistral-7B-v0.3 --country KHM > KHM.txt
python evaluate_model.py --model mistralai/Mistral-7B-v0.3 --country IDN > IDN.txt
python evaluate_model.py --model mistralai/Mistral-7B-v0.3 --country MYS > MYS.txt
python evaluate_model.py --model mistralai/Mistral-7B-v0.3 --country SGP > SGP.txt
python evaluate_model.py --model mistralai/Mistral-7B-v0.3 --country LKA > LKA.txt
python evaluate_model.py --model mistralai/Mistral-7B-v0.3 --country THA > THA.txt

mkdir Mistral-7B-v0.3
mv KHM.txt Mistral-7B-v0.3
mv IDN.txt Mistral-7B-v0.3
mv MYS.txt Mistral-7B-v0.3
mv SGP.txt Mistral-7B-v0.3
mv LKA.txt Mistral-7B-v0.3
mv THA.txt Mistral-7B-v0.3

python evaluate_model.py --model mistralai/Mistral-7B-Instruct-v0.3 --country KHM > KHM.txt
python evaluate_model.py --model mistralai/Mistral-7B-Instruct-v0.3 --country IDN > IDN.txt
python evaluate_model.py --model mistralai/Mistral-7B-Instruct-v0.3 --country MYS > MYS.txt
python evaluate_model.py --model mistralai/Mistral-7B-Instruct-v0.3 --country SGP > SGP.txt
python evaluate_model.py --model mistralai/Mistral-7B-Instruct-v0.3 --country LKA > LKA.txt
python evaluate_model.py --model mistralai/Mistral-7B-Instruct-v0.3 --country THA > THA.txt

mkdir Mistral-7B-Instruct-v0.3
mv KHM.txt Mistral-7B-Instruct-v0.3
mv IDN.txt Mistral-7B-Instruct-v0.3
mv MYS.txt Mistral-7B-Instruct-v0.3
mv SGP.txt Mistral-7B-Instruct-v0.3
mv LKA.txt Mistral-7B-Instruct-v0.3
mv THA.txt Mistral-7B-Instruct-v0.3

python evaluate_model.py --model meta-llama/Llama-3.1-8B --country KHM > KHM.txt
python evaluate_model.py --model meta-llama/Llama-3.1-8B --country IDN > IDN.txt
python evaluate_model.py --model meta-llama/Llama-3.1-8B --country MYS > MYS.txt
python evaluate_model.py --model meta-llama/Llama-3.1-8B --country SGP > SGP.txt
python evaluate_model.py --model meta-llama/Llama-3.1-8B --country LKA > LKA.txt
python evaluate_model.py --model meta-llama/Llama-3.1-8B --country THA > THA.txt

mkdir Llama-3.1-8B
mv KHM.txt Llama-3.1-8B
mv IDN.txt Llama-3.1-8B
mv MYS.txt Llama-3.1-8B
mv SGP.txt Llama-3.1-8B
mv LKA.txt Llama-3.1-8B
mv THA.txt Llama-3.1-8B

python evaluate_model.py --model meta-llama/Llama-3.1-8B-Instruct --country KHM > KHM.txt
python evaluate_model.py --model meta-llama/Llama-3.1-8B-Instruct --country IDN > IDN.txt
python evaluate_model.py --model meta-llama/Llama-3.1-8B-Instruct --country MYS > MYS.txt
python evaluate_model.py --model meta-llama/Llama-3.1-8B-Instruct --country SGP > SGP.txt
python evaluate_model.py --model meta-llama/Llama-3.1-8B-Instruct --country LKA > LKA.txt
python evaluate_model.py --model meta-llama/Llama-3.1-8B-Instruct --country THA > THA.txt

mkdir Llama-3.1-8B-Instruct
mv KHM.txt Llama-3.1-8B-Instruct
mv IDN.txt Llama-3.1-8B-Instruct
mv MYS.txt Llama-3.1-8B-Instruct
mv SGP.txt Llama-3.1-8B-Instruct
mv LKA.txt Llama-3.1-8B-Instruct
mv THA.txt Llama-3.1-8B-Instruct

python evaluate_model.py --model meta-llama/Llama-3.2-3B --country KHM > KHM.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B --country IDN > IDN.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B --country MYS > MYS.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B --country SGP > SGP.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B --country LKA > LKA.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B --country THA > THA.txt

mkdir Llama-3.2-3B
mv KHM.txt Llama-3.2-3B
mv IDN.txt Llama-3.2-3B
mv MYS.txt Llama-3.2-3B
mv SGP.txt Llama-3.2-3B
mv LKA.txt Llama-3.2-3B
mv THA.txt Llama-3.2-3B

python evaluate_model.py --model meta-llama/Llama-3.2-3B-Instruct --country KHM > KHM.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B-Instruct --country IDN > IDN.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B-Instruct --country MYS > MYS.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B-Instruct --country SGP > SGP.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B-Instruct --country LKA > LKA.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B-Instruct --country THA > THA.txt

mkdir Llama-3.2-3B-Instruct
mv KHM.txt Llama-3.2-3B-Instruct
mv IDN.txt Llama-3.2-3B-Instruct
mv MYS.txt Llama-3.2-3B-Instruct
mv SGP.txt Llama-3.2-3B-Instruct
mv LKA.txt Llama-3.2-3B-Instruct
mv THA.txt Llama-3.2-3B-Instruct