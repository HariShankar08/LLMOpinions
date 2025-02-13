#!/bin/bash

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