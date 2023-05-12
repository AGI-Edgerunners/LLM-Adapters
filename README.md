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

## Resource Consumption

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
| LLaMA-7B-Parallel     | 200M                 | 22GB          |     1h           | 


## Finetune Result
There is a finetune result in different model with six dataset, which contains MultiArith, GSM8K, AddSub, AQuA, SingleEq, SVAMP

| Model                 | Params | MultiArith | GSM8K  | AddSub | AQuA   | SingleEq |  SVAMP | Average |
|-----------------------|--------|------------|--------|--------|--------|----------|--------|---------|
| GPT-3.5               | -      |    83.8    | 56.4   |  85.3  |  38.9  |   88.1   |  69.9  |  70.4   |
| LLaMA-13B-LoRA        | 6.5M   |    93.3    | 43.3   |  80.0  |  20.5  |   84.6   |  52.9  |  62.4   |
| LLaMA-13B-AdapterH    | 314M   |    94.0    | 36.1   |  82.3  |  19.7  |   84.8   |  52.9  |  61.6   |
| LLaMA-13B-AdapterP    | 104M   |    94.8    | 41.0   |  81.3  |  19.3  |   87.0   |  51.1  |  62.4   |
| LLaMA-13B-Parallel    | 314M   |  **95.0**  |**43.8**|**84.6**|**20.9**| **88.0** |**53.5**|**64.3** |
| LLaMA-7B-LoRA         | 4.2M   |    88.3    | 30.9   |  78.5  |  14.2  |   74.8   |  47.2  |  55.7   |
| LLaMA-7B-AdapterH     | 200M   |    93.8    | 29.8   |  70.6  |  16.1  |   71.1   |  37.7  |  53.2   |
| LLaMA-7B-AdapterP     | 66M    |    91.0    | 30.2   |  75.7  |  14.9  |   75.4   |  43.3  |  55.1   |
| LLaMA-7B-Parallel     | 200M   |    93.7    | 33.3   |  80.5  |  16.5  |   81.7   |  46.5  |  58.7   |
| BLOOM-7B-LoRA         | 4M     |    73.0    | 9.9    |  41.8  |  16.9  |   40.7   |  25.1  |  34.6   |
| BLOOM-7B-AdapterH     | 125M   |    81.8    | 16.5   |  76.5  |  18.9  |   71.3   |  37.8  |  50.5   |
| BLOOM-7B-AdapterP     | 62M    |    87.7    | 18.0   |  69.6  |**20.9**|   68.3   |  32.1  |  49.4   |
| BLOOM-7B-Parallel     | 125M   |    78.2    | 15.7   |  65.4  |  20.5  |   64.2   |  35.1  |  46.5   |
| GPT-j-6B-LoRA         | 3.7M   |    80.5    | 17.4   |  74.9  |  18.1  |   72.2   |  43.8  |  51.2   |
| GPT-j-6B-AdapterH     | 117M   |    82.5    | 17.9   |  83.8  |  21.3  |   76.8   |  40.0  |  53.7   |
| GPT-j-6B-AdapterP     | 58M    |    90.3    | 19.1   |  80.7  |  18.5  |   81.3   |  41.3  |  55.2   |
| GPT-j-6B-Parallel     | 176M   |    77.8    | 17.5   |  77.2  |  20.5  |   74.8   |  39.8  |  51.3   | 


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
