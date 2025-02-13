#!/usr/bin/bash
mkdir -p steering

python evaluate_model.py --model meta-llama/Llama-3.2-3B --steer hindu > steering/hindu.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B --steer muslim > steering/muslim.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B --steer christian > steering/christian.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B --steer sikh > steering/sikh.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B --steer buddhist > steering/buddhist.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B --steer jain > steering/jain.txt
python evaluate_model.py --model meta-llama/Llama-3.2-3B --steer parsi > steering/parsi.txt
