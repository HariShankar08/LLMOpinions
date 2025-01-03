#!/bin/bash

python evaluate_model.py --model meta-llama/Llama-3.1-8B --country VNM > VNM.txt
python evaluate_model.py --model meta-llama/Llama-3.1-8B --country HKG > HKG.txt
python evaluate_model.py --model meta-llama/Llama-3.1-8B --country TWN > TWN.txt

mkdir Llama-3.1-8B
mv JPN.txt Llama-3.1-8B
mv KOR.txt Llama-3.1-8B
mv VNM.txt Llama-3.1-8B
mv TWN.txt Llama-3.1-8B
mv HKG.txt Llama-3.1-8B

python evaluate_model.py --model meta-llama/Llama-3.1-8B-Instruct --country JPN > JPN.txt
python evaluate_model.py --model meta-llama/Llama-3.1-8B-Instruct --country KOR > KOR.txt
python evaluate_model.py --model meta-llama/Llama-3.1-8B-Instruct --country HKG > HKG.txt
python evaluate_model.py --model meta-llama/Llama-3.1-8B-Instruct --country TWN > TWN.txt
python evaluate_model.py --model meta-llama/Llama-3.1-8B-Instruct --country VNM > VNM.txt

mkdir Llama-3.1-8B-Instruct
mv JPN.txt Llama-3.1-8B-Instruct
mv KOR.txt Llama-3.1-8B-Instruct
mv HKG.txt Llama-3.1-8B-Instruct
mv TWN.txt Llama-3.1-8B-Instruct
mv VNM.txt Llama-3.1-8B-Instruct

python evaluate_model.py --model meta-llama/Llama-3.2-3B --country JPN > JPN.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B --country KOR > KOR.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B --country HKG > HKG.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B --country TWN > TWN.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B --country VNM > VNM.txt

mkdir Llama-3.2-3B
mv JPN.txt Llama-3.2-3B
mv KOR.txt Llama-3.2-3B
mv HKG.txt Llama-3.2-3B
mv TWN.txt Llama-3.2-3B
mv VNM.txt Llama-3.2-3B

python evaluate_model.py --model meta-llama/Llama-3.2-3B-Instruct --country JPN > JPN.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B-Instruct --country KOR > KOR.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B-Instruct --country HKG > HKG.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B-Instruct --country TWN > TWN.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B-Instruct --country VNM > VNM.txt

mkdir Llama-3.2-3B-Instruct
mv JPN.txt Llama-3.2-3B-Instruct
mv KOR.txt Llama-3.2-3B-Instruct
mv HKG.txt Llama-3.2-3B-Instruct
mv TWN.txt Llama-3.2-3B-Instruct
mv VNM.txt Llama-3.2-3B-Instruct
