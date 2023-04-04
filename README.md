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
<!-- <p align="center">
  <img src="picture.jpg" width="73" alt="image_description">
</p> -->
<!-- <center><img src="picture.jpg" width="73"></center> -->
<h1 align="center">LLM-Adapters <img src="picture.jpg" width="50" alt="LLM-Adapters"></h1>
<!-- <h1 align="center"> <p> LLM-Adapter</p></h1> -->
<h3 align="center">
    <p>LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models</p>
</h3>
LLM-Adapters is an easy-to-use framework that integrates various adapters into LLMs and can execute adapter-based PEFT methods of LLMs for different tasks. LLM-Adapter is an extension of HuggingFace's PEFT library, many thanks for their amazing work!

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
    --dataset SVAMP    #specify the test dataset
```



## Resource Consumption

There is a table of resouce needed for different adapters, which contains Trainable Parameters, GPU RAM Usage, and Fine-tuning Time on the Arithmetic Reasoning dataset `math_data.json`

Hyper-parameter setting: num_epochs=3, lora_r=8, lora_alpha=16, bottleneck_size=256 

Models: LLaMA-7B, BLOOM-6.7B, GPT-j-6B

Hardware: 8*V100 GPUs

| Model          | Trainable Parameters | GPU RAM Usage | Fine-tuning Time |
| -------------- | -------------------- | ------------- | ---------------- |
| LLaMA-LoRA     | xx                   | xx            | xx               |
| LLaMA-AdapterH | xx                   | xx            | xx               |
| LLaMA-AdapterP | xx                   | xx            | xx               |
| LLaMA-Parallel | xx                   | xx            | xx               |
| BLOOM-LoRA     | xx                   | xx            | xx               |
| BLOOM-AdapterH | xx                   | xx            | xx               |
| BLOOM-AdapterP | xx                   | xx            | xx               |
| BLOOM-Parallel | xx                   | xx            | xx               |
| GPT-j-LoRA     | xx                   | xx            | xx               |
| GPT-j-AdapterH | xx                   | xx            | xx               |
| GPT-j-AdapterP | xx                   | xx            | xx               |
| GPT-j-Parallel | xx                   | xx            | xx               |

## Performance on



| Model          | Addsub | SingleEQ | MultiArith |     |     |     |
| -------------- | ------ | -------- | ---------- | --- | --- | --- |
| LLaMA-LoRA     | xx     | xx       | xx         |     |     |     |
| LLaMA-AdapterH | xx     | xx       | xx         |     |     |     |
| LLaMA-AdapterP | xx     | xx       | xx         |     |     |     |
| LLaMA-Parallel | xx     | xx       | xx         |     |     |     |
| BLOOM-LoRA     | xx     | xx       | xx         |     |     |     |

### Adapter support matrix

This metrix shows whether different models can use LoRA,AdapterH,AdapterP,Parallel and Scaled Parallel adapters.

| Adapter         | LLaMA-7B | BLOOM-7B | GPT-j-6B |
| --------------- | -------- | -------- | -------- |
| LoRA            | ✅        | ✅        | ✅        |
| AdapterH        | ✅        | ✅        | ✅        |
| AdapterP        | ✅        | ✅        | ✅        |
| Parallel        | ✅        | ✅        | ✅        |
| Scaled Parallel | ✅        | ✅        | ✅        |

## Citing <img src="piture"> LLM-Adapter

If you use <img src="piture"> LLM-Adapter in your publication, please cite it by using the following BibTeX entry.

```bibtex
@Misc{peft,
  title =        {LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models},
  author =       {Zhiqiang Hu, Yihuai Lan, Lei Wang, Wanyu Xu, Ee-Peng Lim, Roy Ka-Wei Lee, Lidong Bing, Soujanya Poria},
  howpublished = {\url{https://github.com/AGI-Edgerunners/LLM-Adapters}},
  year =         {2023}
}
```

## Acknowledgement

This repo benefits from PEFT,Adapter-Transformer, LLaMa-Adapter. Thanks for their wonderful works.
