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

<h1 align="center"> 
<img src="picture.jpg" width="73" height="114">
<p> LLM-Adapters</p>
</h1>

<h3 align="center">
    <p>LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models </p>
</h3>
LLM-Adapters is an easy-to-use framework that integrates various adapters into LLMs and can execute adapter-based PEFT methods of LLMs for different tasks. LLM-Adapter is an extension of HuggingFace's PEFT library, many thanks for their amazing work! Please find our paper at this link: https://arxiv.org/abs/2304.01933.

The framework includes state-of-the-art open-access LLMs: LLaMa, OPT, BLOOM, and GPT-J, as well as widely used adapters such as Bottleneck adapters, Parallel adapters, and LoRA.

Supported Adapters:

1. LoRA: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685.pdf)
2. AdapterH: [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/pdf/1902.00751.pdf)
3. AdapterP: [GMAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer](https://arxiv.org/pdf/2005.00052.pdf)
4. Parallel: [TOWARDS A UNIFIED VIEW OF PARAMETER-EFFICIENT TRANSFER LEARNING](https://arxiv.org/pdf/2110.04366.pdf)
5. Prefix Tuning: [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353/), [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/pdf/2110.07602.pdf)
6. P-Tuning: [GPT Understands, Too](https://arxiv.org/pdf/2103.10385.pdf)
7. Prompt Tuning: [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691.pdf) 

## Setup

1. Install dependencies
```bash
pip install -r requirements.txt
cd peft/
pip install -e .
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

Example usage for multiple GPUs:

```bash
WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=3192 finetune.py \
  --base_model 'decapoda-research/llama-7b-hf' \
  --data_path 'math_data.json' \
  --output_dir './trained_models/llama-lora' \
  --batch_size 16 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lora
```

The `math_data.json` file contains preprocessed instruction data from the addsub, SingleEQ, MultiArith, AQuA, SVAMP and GSM8K dataset. `decapoda-research/llama-7b-hf` is a base model, LLaMa-7B. Add `lora` adapter to this model.

Example usage for Single GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py \
  --base_model 'decapoda-research/llama-7b-hf' \
  --data_path 'math_data.json' \
  --output_dir './trained_models/llama-lora' \
  --batch_size 16 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lora
```

Moreover, you can use `--use_gradient_checkpointing` to save more GPU memory, but it will increase the training time.

To use the bottleneck adapters (AdapterH and AdapterP), just add the following arguments:

```bash
--adapter_name bottleneck # use the bottleneck adapter, refers to AdapterH in the result table
--target_modules '[down_proj]' # add the last mlp layer name first to place the adapter only after MLP modoule, refers to AdapterP in the result table
```

To use parallel adapter, just add the following arguments:

```bash
--adapter_name bottleneck
--use_parallel_adapter
```

Note that, In order to facilitate INT8 training of large models with parallel adapters, we have adopted a technique whereby the parallel adapter layers are incorporated into multi-head attention layers and MLP layers, in parallel with Linear layers. It is different from [Hu et al. (2021)](https://arxiv.org/pdf/2106.09685.pdf). 

## Inference (generate.py)

This file reads the foundation model from the Hugging Face model hub and the LoRA weights from `'./trained_models/llama-lora'` , and runs a Gradio interface for inference on a specified input. Users should treat this as example code for the use of the model, and modify it as needed.
Example usage:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun generate.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora'
```

## Evaluation (evaluate.py)

To evaluate the performance of the finetuned model on the Arithmetic Reasoning tasks, you can use the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py 
    --model LLaMA-7B \ #specify the base model
    --adapter LoRA \   #specify the adapter name ["LoRA", "AdapterH", "AdapterP", "Parallel"， "Scaled_Parallel""]
    --dataset SVAMP \  #specify the test dataset
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora'
```

## Resource Consumption

There is a table of resouce needed for different adapters, which contains Trainable Parameters, GPU RAM Usage, and Fine-tuning Time on the Arithmetic Reasoning dataset `math_data.json`

Hyper-parameter setting: num_epochs=3, lora_r=8, lora_alpha=16, bottleneck_size=256 (768 for Parallel Adapter)

Models: LLaMA-7B, BLOOM-6.7B, GPT-j-6B
Dataset: 3.2K math word problems

Hardware: 2*3090 GPUs

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


### TODO List
- [x] Add AdapterH
- [x] Add AdapterP
- [x] Add Parallel Adapter
- [ ] Support More LLMs
- [ ] Support Multiple Adapter
- [ ] Support Adapter Composition
- [ ] Support Adapter Fusion


## Citing <img src="picture.jpg" width="14px" height="14px"> LLM-Adapter

If you use <img src="picture.jpg" width="14px" height="14px"> LLM-Adapters in your publication, please cite it by using the following BibTeX entry.

```bibtex
@misc{hu2023llmadapters,
      title={LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models}, 
      author={Zhiqiang Hu and Yihuai Lan and Lei Wang and Wanyu Xu and Ee-Peng Lim and Roy Ka-Wei Lee and Lidong Bing and Soujanya Poria},
      year={2023},
      eprint={2304.01933},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Acknowledgement

This repo benefits from [PEFT](https://github.com/huggingface/peft), [Adapter-Transformer](https://github.com/adapter-hub/adapter-transformers), [Alpaca-lora](https://github.com/tloen/alpaca-lora). Thanks for their wonderful works. Additionally, we thank DONG Shan and [dream.ai](https://dream.ai/create) for the exceptional logo design, which has added immense value to our project.
