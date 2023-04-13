import os
import subprocess
import multiprocessing


def evaluate(dataset, gpu):
    print('*******dataset:', dataset)

    command = f"CUDA_VISIBLE_DEVICES={gpu} python evaluate.py \
               --model LLaMA-7B \
               --adapter LoRA \
               --dataset {dataset} \
               --base_model '../LLM/models/llama-7b-hf' \
               --lora_weights './trained_models/llama-lora'"

    result = subprocess.run(command, shell=True, text=True, capture_output=False)
    print(f"Evaluation results for dataset {dataset} on GPU {gpu}:\n{result.stdout}")


datasets = ['AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'AQuA', 'SVAMP']
gpus = [1, 2, 3]

num_processes = min(len(datasets), len(gpus))  # number of processes to run in parallel
pool = multiprocessing.Pool(processes=num_processes)
for i in range(num_processes):
    pool.apply_async(evaluate, args=(datasets[i], gpus[i]))
pool.close()
pool.join()
