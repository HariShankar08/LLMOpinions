#!/bin/bash

python evaluate_model.py --model google/gemma-2-9b --country JPN > JPN.txt
python evaluate_model.py --model google/gemma-2-9b --country HKG > HKG.txt
python evaluate_model.py --model google/gemma-2-9b --country KOR > KOR.txt
python evaluate_model.py --model google/gemma-2-9b --country TWN > TWN.txt
python evaluate_model.py --model google/gemma-2-9b --country VNM > VNM.txt

