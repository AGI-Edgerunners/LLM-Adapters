<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<h1 align="center"> <p> LLM-Adapter</p></h1>
<center><img src="picture.jpg" width="73"></center>
<h3 align="center">
    <p>LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models</p>
</h3>
LLM-Adapters is an easy-to-use framework that integrates various adapters into LLMs and can execute adapter-based PEFT methods of LLMs for different tasks. 

The framework includes state-of-the-art open-access LLMs: LLaMa-7B, OPT-6.7B, BLOOM-7.1B, and GPT-J, as well as widely used adapters such as Bottleneck adapters, Parallel adapters, and LoRA.

Supported Adapters:

1. LoRA: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685.pdf)
2. AdapterH: [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/pdf/1902.00751.pdf)
3. AdapterP: [GMAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer](https://arxiv.org/pdf/2005.00052.pdf)
4. Parallel/Scaled Parallel: [TOWARDS A UNIFIED VIEW OF PARAMETER-EFFICIENT TRANSFER LEARNING](https://arxiv.org/pdf/2110.04366.pdf)

## Setup

1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Set environment variables, or modify the files referencing `BASE_MODEL`:

```bash
# Files referencing `BASE_MODEL`
# export_hf_checkpoint.py
# export_state_dict_checkpoint.py

export BASE_MODEL=decapoda-research/llama-7b-hf
```

Both `finetune.py` and `generate.py` use `--base_model` flag as shown further below.

3. If bitsandbytes doesn't work, [install it from source.](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md) Windows users can follow [these instructions](https://github.com/tloen/alpaca-lora/issues/17).


## Training(finetune.py)
This file contains some code related to prompt construction and tokenization.In this file, specify different adapters and different sets of data, so that different models can be trained. 

Example usage:

```bash
WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=3192 finetune.py \
  --base_model 'decapoda-research/llama-7b-hf' \
  --data_path 'math_data.json' \
  --output_dir './trained_models/llama-lora' \
  --batch_size 128 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lora
```

The ```math_data.json``` file contains preprocessed instruction data from the addsub, SingleEQ and MultiArith dataset.

```decapoda-research/llama-7b-hf``` is a pretrained model, LLaMa-7B. 

Add ```lora``` adapter to this model

## Evaluate (evaluate.py)
This file reads the foundation model from the Hugging Face model hub and the LoRA weights from tloen/alpaca-lora-7b, and runs a Gradio interface for inference on a specified input. Users should treat this as example code for the use of the model, and modify it as needed.
Example usage:

```bash
CUDA_VISIBLE_DEVICES=12 nohup python evaluate.py 
--model LLaMA-7B 
--adapter lora 
--dataset gsm8k
```

## Performance of different model

There is a matrix of performance of different model, which contains Trainable Parameters, GPU RAM Usage, Fine-tuning Time with different model with Alpaca dataset

Hardware: Eight V100 GPUs

| Model                 | Trainable Parameters | GPU RAM Usage | Fine-tuning Time |
|-----------------------|----------------------|---------------|------------------|
| LLaMA-LoRA            | 4.2M                 | 18GB          | 15mins           | 
| LLaMA-AdapterH        | 200M                 | 22GB          | 15mins           | 
| LLaMA-AdapterP        | 200M                 | 22GB            | 14mins           | 
| LLaMA-Parallel        | 200M                 | 22GB            | 14mins           | 


## Finetune Result
There is a finetune result in different model with six dataset, which contains MultiArith, GSM8K, AddSub, AQuA, SingleEq, SVAMP

| Model                 | Params | MultiArith | GSM8K | AddSub | AQuA | SingleEq | SVAMP | Average |
|-----------------------|--------|------------|-------|--------|------|----------|-------|---------|
| GPT-3.5               | -      | 83.8       | 56.4  | 85.3   | 38.9 | 88.1     | 69.9  | 70.4    |
| LLaMA-LoRA            | 4.2M   | 88.3       | 21.9  | 78.5   | 27.5 | 83.3     | 54.5  | 59.0    |
| LLaMA-AdapterH        | 200M   | 88.3       | 18.5  | 69.6   | 27.4 | 85.2     | 52.5  | 56.9    |
| LLaMA-AdapterP        | 200M   | 88.3       | 18.5  | 69.6   | 15.6 | 79.4     | 52.0  | 53.9    |
| LLaMA-Parallel        | 200M   | 83.3       | 22.7  | 77.2   | 9.8  | 81.3     | 57.0  | 55.2    |
| BLOOM-LoRA            | 4M     | 46.7       | 4.2   | 32.9   | 11.7 | 41.2     | 22.5  | 26.5    |
| BLOOM-AdapterH        | 125M   | 60.8       | 6.4   | 43     | 23.5 | 52       | 37.5  | 37.2    |
| BLOOM-AdapterP        | 188M   | 70.6       | 8.3   | 50.6   | 13.7 | 50       | 35.5  | 38.1    |
| BLOOM-Parallel        | 125M   | 55         | 5.7   | 35.4   | 27.5 | 49       | 28    | 33.4    |
| GPT-j-LoRA            | 3.7M   | 79.2       | 10.6  | 69.6   | 2.0  | 71.6     | 45.0  | 46.3    |
| GPT-j-AdapterH        | 117M   | 82.5       | 4.5   | 55.7   | 3.9  | 67.6     | 39.5  | 42.3    |
| GPT-j-AdapterP        | 176M   | 79.2       | 9.8   | 54.4   | 19.6 | 63.7     | 37.5  | 44.0    |
| GPT-j-Parallel        | 176M   | 79.2         | 11.0  | 65.8   | 11.8 | 69.6     | 44.5  | 47.0    |


### Adapter support matrix
This metrix shows whether different models can use LoRA,AdapterH,AdapterP,Parallel and Scaled Parallel adapters.

| Adapter      | LoRA | AdapterH | AdapterP | Parallel| Prefix Tuning	|P-Tuning|Prompt Tuning|
|--------------|-------|-------|----------|-------|-------|-------|-------|
| LLaMA        | ✅     | ✅     | ✅        |✅     | ✅     | ✅     | ✅     |
| BLOOM        | ✅     | ✅     | ✅        |✅     | ✅     | ✅     | ✅     | 
| GPT-J        | ✅     | ✅     | ✅        |✅     | ✅     | ✅     | ✅     |
| OPT          | ✅ | ✅     | ✅  |✅ | ✅     | ✅     | ✅     |
| GPT-2        | ✅ | 🔧Developing | 🔧Developing |🔧Developing | ✅     | ✅     | ✅     | 
| GPT-Neo      | ✅ | 🔧Developing | 🔧Developing |🔧Developing | ✅     | ✅     | ✅     | 
| GPT-NeoX-20B | ✅ | 🔧Developing | 🔧Developing |🔧Developing | ✅     | ✅     | ✅     |
| ChatGLM      | ✅ | 🔧Developing | 🔧Developing |🔧Developing | ✅     | ✅     | ✅     | 



## Citing LLM-Adapter

If you use LLM-Adapter in your publication, please cite it by using the following BibTeX entry.

```bibtex
@Misc{LLM-Adapter,
  title =        {LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models},
  author =       {Zhiqiang Hu, Yihuai Lan, Lei Wang, Wanyu Xu, Ee-Peng Lim, Roy Ka-Wei Lee, Lidong Bing, Soujanya Poria},
  howpublished = {\url{https://github.com/AGI-Edgerunners/LLM-Adapters}},
  year =         {2023}
}
```

## Acknowledgement
This repo benefits from PEFT,Adapter-Transformer, Alpaca-lora. Thanks for their wonderful works.