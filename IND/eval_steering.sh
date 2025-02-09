#!/usr/bin/bash
mkdir -p steering

python evaluate_model.py --model meta-llama/Llama-3.2-3B-Instruct --steer hindu > steering/hindu-it.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B-Instruct --steer muslim > steering/muslim-it.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B-Instruct --steer christian > steering/christian-it.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B-Instruct --steer sikh > steering/sikh-it.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B-Instruct --steer buddhist > steering/buddhist-it.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B-Instruct --steer jain > steering/jain-it.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B-Instruct --steer parsi > steering/parsi-it.txt
