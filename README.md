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

## Latest News 🔥🔥

* [2023-07-16] we released commonsense170k dataset and the  The LLaMA-13B-Parallel model outformances ChatGPT on 8 commonsense benchmarks.
* [2023-04-21] We released math10k dataset and the [LLaMA-13B adapter checkpoints](https://drive.google.com/file/d/1NqUv-Hn_mAkGXsUOqpJKmPKW5Gp8mRlO/view?usp=sharing). The LLaMA-13B-Parallel model achieves **91%** of GPT-3.5 performance!
* [2023-04-10] We can support GPT-Neo and ChatGLM now!
* [2023-04-04] [Release code and dataset](https://github.com/AGI-Edgerunners/LLM-Adapters)

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

export BASE_MODEL=yahma/llama-7b-hf
```

Both `finetune.py` and `generate.py` use `--base_model` flag as shown further below.

3. If bitsandbytes doesn't work, [install it from source.](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md) Windows users can follow [these instructions](https://github.com/tloen/alpaca-lora/issues/17).

## Training(finetune.py)

This file contains some code related to prompt construction and tokenization.In this file, specify different adapters and different sets of data, so that different models can be trained. 

Example usage for multiple GPUs:

```bash
WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=3192 finetune.py \
  --base_model 'yahma/llama-7b-hf' \
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

The `math_data.json` file contains preprocessed instruction data from the addsub, SingleEQ, MultiArith, AQuA, SVAMP and GSM8K dataset. `yahma/llama-7b-hf` is a base model, LLaMa-7B. Add `lora` adapter to this model.

Example usage for Single GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py \
  --base_model 'yahma/llama-7b-hf' \
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

To use the AdapterH, just add the following arguments:

```bash
--adapter_name bottleneck # use the bottleneck adapter, refers to AdapterH in the result table
```

To use the AdapterP, just add the following arguments:

```bash
--adapter_name bottleneck 
--use_adapterp  # use the AdapterP, refers to AdapterP in the result table
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
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora'
```

## Evaluation (evaluate.py)

To evaluate the performance of the finetuned model on the Arithmetic Reasoning tasks, you can use the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py 
    --model LLaMA-7B \ #specify the base model
    --adapter LoRA \   #specify the adapter name ["LoRA", "AdapterH", "AdapterP", "Parallel"， "Scaled_Parallel""]
    --dataset SVAMP \  #specify the test dataset
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora'
```

<!-- ## Resource Consumption

There is a table of resouce needed for different adapters, which contains Trainable Parameters, GPU RAM Usage, and Fine-tuning Time on the Arithmetic Reasoning dataset `math_data.json`

Hyper-parameter setting: num_epochs=3, lora_r=8, lora_alpha=16, bottleneck_size=256

Models: LLaMA-13B, LLaMA-7B, BLOOM-6.7B, GPT-j-6B
Dataset: 3.2K math word problems

Hardware: 2*3090 GPUs

| Model                 | Trainable Parameters | GPU RAM Usage | Fine-tuning Time |
|-----------------------|----------------------|---------------|------------------|
| LLaMA-7B-LoRA         | 4.2M                 | 18GB          |     1h           | 
| LLaMA-7B-AdapterH     | 200M                 | 22GB          |     1h           | 
| LLaMA-7B-AdapterP     | 200M                 | 22GB          |     1h           | 
| LLaMA-7B-Parallel     | 200M                 | 22GB          |     1h           |  -->


## Finetune Result
There is the finetune results in different model with six math reasoning datasets, which contains MultiArith, GSM8K, AddSub, AQuA, SingleEq, SVAMP. In this tabel, as AdapterH and AdapterP are Series adapters, and AdapterP outperformances AdapterH, we use AdapterP with bottleneck size 256 as Series Adapter.

| Model                 | MultiArith | GSM8K  | AddSub | AQuA   | SingleEq |  SVAMP | Average |
|-----------------------|------------|--------|--------|--------|----------|--------|---------|
| GPT-3.5               |    83.8    |**56.4**|  85.3  |**38.9**|   88.1   |**69.9**|**70.4** |
| BLOOMz-7B-Prefix	    |    68.8    | 13.8   |  47.1  |  12.5  |   49.4   |  24.1  |  36.0   |
| BLOOMz-7B-Series	    |    80.7    | 14.3   |  72.6  |  20.5  |   69.3   |  38.1  |  49.3   |
| BLOOMz-7B-Parallel	  |    85.8    | 18.5   |  77.7  |  18.9  |   74.8   |  36.4  |  52.0   |
| BLOOMz-7B-LoRA	      |    82.8	   | 17.4	  |  72.4	 |  21.3	|   69.9	 |  41.0	|  50.8   |
| GPT-j-6B-Prefix	      |    74.5	   | 16.0	  |  65.6	 |  14.7	|   61.4	 |  31.0	|  43.9   |
| GPT-j-6B-Series	      |    91.7	   | 19.5	  |  85.8	 |  15.0	|   81.7	 |  43.6	|  56.2   |
| GPT-j-6B-Parallel	    |    92.2	   | 18.9	  |  83.8	 |  17.9	|   80.7	 |  41.1	|  55.8   |
| GPT-j-6B-LoRA	        |    90.7	   | 23.0	  |  84.1	 |  16.1	|   84.1	 |  46.0	|  57.3   |
| LLaMA-7B-Prefix	      |    63.2	   | 24.4	  |  57.0	 |  14.2	|   55.3	 |  38.1	|  42.0   |
| LLaMA-7B-Series	      |    92.8	   | 33.3	  |  80.0	 |  15.0	|   83.5	 |  52.3	|  59.5   |
| LLaMA-7B-Parallel	    |    94.5	   | 35.3	  |  86.6	 |  18.1	|   86.0	 |  49.6	|  61.7   |
| LLaMA-7B-LoRA	        |  **95.0**	 | 37.5	  |  83.3	 |  18.9	|   84.4	 |  52.1	|  61.9   |
| LLaMA-13B-Prefix	    |    72.2	   | 31.1	  |  56.0	 |  15.7	|   62.8	 |  41.4	|  46.5   |
| LLaMA-13B-Series	    |    93.0	   | 44.0	  |  80.5	 |  22.0	|   87.6	 |  50.8	|  63.0   |
| LLaMA-13B-Parallel	  |    94.3	   | 43.3	  |  83.0	 |  20.5	|   89.6	 |  55.7	|  64.4   |
| LLaMA-13B-LoRA	      |    94.8	   | 47.5	  |**87.3**|	18.5	| **89.8** |  54.6	|  65.4   |


There is the finetune results in different model with eight commonsense reasoning datasets.

| Model                 |  BoolQ  |  PIQA  |  SIQA  |  HellaSwag  |  WinoGrande  |  ARC-e  |  ARC-c  |  OBQA  |  Average  |
|-----------------------|---------|--------|--------|-------------|--------------|---------|---------|--------|-----------|
| ChatGPT               | **73.1**|**85.4**|  68.5  |  78.5       |  66.1        |**89.8** |**79.9** |  74.8  |  77.0     |
| BLOOMz-7B-Prefix	    |   45.6  |  53.7  |  46.3  |  26.7       |  49.5        |  52.1   |  39.7   |  44.3  |  44.7     |
| BLOOMz-7B-Series	    |   65.4  |  70.4  |  73.6  |  53.4       |  69.3        |  72.3   |  55.9   |  68.0  |  66.0     |
| BLOOMz-7B-Parallel	  |   64.1  |  71.5  |  72.1  |  52.9       |  67.0        |  70.5   |  54.7   |  69.6  |  65.3     |
| BLOOMz-7B-LoRA	      |   65.9  |  75.3  |  74.5  |  57.3       |  72.5        |  74.6   |  57.8   |  73.4  |  68.9     |
| GPT-j-6B-Prefix	      |   63.1  |  66.9  |  68.7  |  34.4       |  64.5        |  64.4   |  46.8   |  59.0  |  58.5     |
| GPT-j-6B-Series	      |   62.1  |  63.5  |  72.3  |  30.6       |  68.0        |  63.9   |  48.1   |  63.8  |  59.0     |
| GPT-j-6B-Parallel	    |   62.2  |  69.7  |  70.0  |  41.7       |  65.0        |  60.2   |  44.6   |  58.2  |  59.0     |
| GPT-j-6B-LoRA	        |   62.4  |  68.6  |  49.5  |  43.1       |  57.3        |  43.4   |  31.0   |  46.6  |  50.2     |
| LLaMA-7B-Prefix	      |   64.3  |  76.8  |  73.9  |  42.1       |  72.1        |  72.9   |  54.0   |  60.6  |  64.6     |
| LLaMA-7B-Series	      |   63.0  |  79.2  |  76.3  |  67.9       |  75.7        |  74.5   |  57.1   |  72.4  |  70.8     |
| LLaMA-7B-Parallel	    |   67.9  |  76.4  |  78.8  |  69.8       |  78.9        |  73.7   |  57.3   |  75.2  |  72.3     |
| LLaMA-7B-LoRA	        |   68.9  |  80.7  |  77.4  |  78.1       |  78.8        |  77.8   |  61.3   |  74.8  |  74.7     |
| LLaMA-13B-Prefix	    |   65.3  |  75.4  |  72.1  |  55.2       |  68.6        |  79.5   |  62.9   |  68.0  |  68.4     |
| LLaMA-13B-Series	    |   71.8  |  83.0  |  79.2  |  88.1       |  82.4        |  82.5   |  67.3   |  81.8  |  79.5     |
| LLaMA-13B-Parallel	  |   72.5  |  84.8  |  79.8  |**92.1**     |**84.7**      |  84.2   |  71.2   |**82.4**|**81.5**   |
| LLaMA-13B-LoRA	      |   72.1  |  83.5  |**80.5**|  90.5       |  83.7        |  82.8   |  68.3   |**82.4**|  80.5     |


### Adapter support matrix
This metrix shows whether different models can use LoRA,AdapterH,AdapterP,Parallel and Scaled Parallel adapters.

| Adapter      | LoRA | AdapterH | AdapterP | Parallel| Prefix Tuning	|P-Tuning|Prompt Tuning|
|--------------|-------|-------|----------|-------|-------|-------|-------|
| LLaMA        | ✅     | ✅     | ✅        |✅     | ✅     | ✅     | ✅     |
| BLOOM        | ✅     | ✅     | ✅        |✅     | ✅     | ✅     | ✅     | 
| GPT-J        | ✅     | ✅     | ✅        |✅     | ✅     | ✅     | ✅     |
| OPT          | ✅     | ✅     | ✅        |✅     | ✅     | ✅     | ✅     |
| GPT-2        | ✅     | 🔧Developing | 🔧Developing|🔧Developing | ✅     | ✅     | ✅     | 
| GPT-Neo      | ✅     | ✅     | ✅        | ✅    | ✅     | ✅     | ✅     | 
| GPT-NeoX-20B | ✅     | 🔧Developing | 🔧Developing|🔧Developing | ✅     | ✅     | ✅     |
| ChatGLM      | ✅     | ✅     | ✅        |✅     | ✅     | ✅     | ✅     | 


### TODO List
- [x] Add AdapterH
- [x] Add AdapterP
- [x] Add Parallel Adapter
- [ ] Support More LLMs
- [ ] Support Multiple Adapter
- [ ] Support Adapter Composition
- [ ] Support Adapter Fusion


## :star: Star History

[![Star History Chart](https://api.star-history.com/svg?repos=AGI-Edgerunners/LLM-Adapters&type=Date)](https://star-history.com/#AGI-Edgerunners/LLM-Adapters&Date)

## Citing <img src="picture.jpg" width="14px" height="14px"> LLM-Adapter

If you use <img src="picture.jpg" width="14px" height="14px"> LLM-Adapters in your publication, please cite it by using the following BibTeX entry.

```bibtex
@article{hu2023llm,
  title={LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models},
  author={Hu, Zhiqiang and Lan, Yihuai and Wang, Lei and Xu, Wanyu and Lim, Ee-Peng and Lee, Roy Ka-Wei and Bing, Lidong and Poria, Soujanya},
  journal={arXiv preprint arXiv:2304.01933},
  year={2023}
}
```

## Acknowledgement

This repo benefits from [PEFT](https://github.com/huggingface/peft), [Adapter-Transformer](https://github.com/adapter-hub/adapter-transformers), [Alpaca-lora](https://github.com/tloen/alpaca-lora). Thanks for their wonderful works. Additionally, we thank DONG Shan and [dream.ai](https://dream.ai/create) for the exceptional logo design, which has added immense value to our project.
