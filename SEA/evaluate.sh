#!/bin/bash

python evaluate_model.py --model google/gemma-2-9b --country KHM > KHM.txt
python evaluate_model.py --model google/gemma-2-9b --country IDN > IDN.txt
python evaluate_model.py --model google/gemma-2-9b --country MYS > MYS.txt
python evaluate_model.py --model google/gemma-2-9b --country SGP > SGP.txt
python evaluate_model.py --model google/gemma-2-9b --country LKA > LKA.txt
python evaluate_model.py --model google/gemma-2-9b --country THA > THA.txt
