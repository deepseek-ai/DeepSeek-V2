<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/logo.svg?raw=true" width="60%" alt="DeepSeek-V2" />
</div>
<hr>
<div align="center" style="line-height: 1;">
  <a href="https://www.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/badge.svg?raw=true" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://chat.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/🤖%20Chat-DeepSeek%20V2-536af5?color=536af5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/deepseek-ai" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20AI-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://discord.gg/Tc7c45Zzu5" target="_blank" style="margin: 2px;">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-DeepSeek%20AI-7289da?logo=discord&logoColor=white&color=7289da" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/qr.jpeg?raw=true" target="_blank" style="margin: 2px;">
    <img alt="Wechat" src="https://img.shields.io/badge/WeChat-DeepSeek%20AI-brightgreen?logo=wechat&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://twitter.com/deepseek_ai" target="_blank" style="margin: 2px;">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-deepseek_ai-white?logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/LICENSE-CODE" style="margin: 2px;">
    <img alt="Code License" src="https://img.shields.io/badge/Code_License-MIT-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/LICENSE-MODEL" style="margin: 2px;">
    <img alt="Model License" src="https://img.shields.io/badge/Model_License-Model_Agreement-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<p align="center">
  <a href="#2-model-downloads">Model Download</a> |
  <a href="#3-evaluation-results">Evaluation Results</a> |
  <a href="#4-model-architecture">Model Architecture</a> |
  <a href="#6-api-platform">API Platform</a> |
  <a href="#8-license">License</a> |
  <a href="#9-citation">Citation</a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2405.04434"><b>Paper Link</b>👁️</a>
</p>

# DeepSeek-V2:  A Strong, Economical, and Efficient Mixture-of-Experts Language Model

## 1. Introduction
Today, we’re introducing DeepSeek-V2, a strong Mixture-of-Experts (MoE) language model characterized by economical training and efficient inference. It comprises 236B total parameters, of which 21B are activated for each token. Compared with DeepSeek 67B, DeepSeek-V2 achieves stronger performance, and meanwhile saves 42.5% of training costs, reduces the KV cache by 93.3%, and boosts the maximum generation throughput to 5.76 times. 

<p align="center">
<div style="display: flex; justify-content: center;">
    <img src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/activationparameters.png?raw=true" style="height:300px; width:auto; margin-right:10px">
    <img src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/trainingcost.png?raw=true" style="height:300px; width:auto; margin-left:10px">
</div>
</p>

We pretrained DeepSeek-V2 on a diverse and high-quality corpus comprising 8.1 trillion tokens. This comprehensive pretraining was followed by a process of Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) to fully unleash the model's capabilities. The evaluation results validate the effectiveness of our approach as DeepSeek-V2 achieves remarkable performance on both standard benchmarks and open-ended generation evaluation.

## 2. News

- 2024.05.16: We released the DeepSeek-V2-Lite.
- 2024.05.06: We released the DeepSeek-V2.

## 3. Model Downloads

<div align="center">

| **Model** | **#Total Params** | **#Activated Params** | **Context Length** | **Download** |
| :------------: | :------------: | :------------: | :------------: | :------------: |
| DeepSeek-V2-Lite | 16B | 2.4B | 32k   | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)   |
| DeepSeek-V2-Lite-Chat (SFT)   | 16B | 2.4B | 32k   | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat)   |
| DeepSeek-V2   | 236B | 21B |  128k   | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V2)   |
| DeepSeek-V2-Chat (RL)   | 236B | 21B |  128k   | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat)   |

</div>

Due to the constraints of HuggingFace, the open-source code currently experiences slower performance than our internal codebase when running on GPUs with Huggingface. To facilitate the efficient execution of our model, we offer a dedicated vllm solution that optimizes performance for running our model effectively.

## 4. Evaluation Results
### Base Model
#### Standard Benchmark (Models larger than 67B)

<div align="center">

| **Benchmark** | **Domain** | **LLaMA3 70B** | **Mixtral 8x22B** | **DeepSeek-V1 (Dense-67B)** | **DeepSeek-V2 (MoE-236B)** |
|:-----------:|:--------:|:------------:|:---------------:|:-------------------------:|:------------------------:|
| **MMLU** | English | 78.9 | 77.6 | 71.3 | 78.5 |
| **BBH** | English | 81.0 | 78.9 | 68.7 | 78.9 |
| **C-Eval** | Chinese | 67.5 | 58.6 | 66.1 | 81.7 |
| **CMMLU** | Chinese | 69.3 | 60.0 | 70.8 | 84.0 |
| **HumanEval** | Code | 48.2	| 53.1 | 45.1 | 48.8 |
| **MBPP** | Code | 68.6 | 64.2 | 57.4 | 66.6 |
| **GSM8K** | Math | 83.0 | 80.3 | 63.4 | 79.2 |
| **Math** | Math | 42.2 | 42.5 | 18.7 | 43.6 |

</div>

#### Standard Benchmark (Models smaller than 16B)
<div align="center">

| **Benchmark** | **Domain** | **DeepSeek 7B (Dense)** | **DeepSeekMoE 16B** | **DeepSeek-V2-Lite (MoE-16B)** |
|:-------------:|:----------:|:--------------:|:-----------------:|:--------------------------:|
| **Architecture**      | -    | MHA+Dense           | MHA+MoE              | MLA+MoE                       |
| **MMLU**      | English    | 48.2           | 45.0              | 58.3                       |
| **BBH**       | English    | 39.5           | 38.9              | 44.1                       |
| **C-Eval**    | Chinese    | 45.0           | 40.6              | 60.3                       |
| **CMMLU**     | Chinese    | 47.2           | 42.5              | 64.3                       |
| **HumanEval** | Code       | 26.2           | 26.8              | 29.9                       |
| **MBPP**      | Code       | 39.0           | 39.2              | 43.2                       |
| **GSM8K**     | Math       | 17.4           | 18.8              | 41.1                       |
| **Math**      | Math       | 3.3            | 4.3               | 17.1                       |

</div>
For more evaluation details, such as few-shot settings and prompts, please check our paper. 

#### Context Window
<p align="center">
  <img width="80%" src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/niah.png?raw=true">
</p>

Evaluation results on the ``Needle In A Haystack`` (NIAH) tests.  DeepSeek-V2 performs well across all context window lengths up to **128K**. 

### Chat Model
#### Standard Benchmark (Models larger than 67B)
<div align="center">

| Benchmark | Domain         | QWen1.5 72B Chat | Mixtral 8x22B | LLaMA3 70B Instruct | DeepSeek-V1 Chat (SFT) | DeepSeek-V2 Chat (SFT) | DeepSeek-V2 Chat (RL) |
|:-----------:|:----------------:|:------------------:|:---------------:|:---------------------:|:-------------:|:-----------------------:|:----------------------:|
| **MMLU**      | English        | 76.2             | 77.8          | 80.3                | 71.1        | 78.4                 | 77.8                 |
| **BBH**       | English        | 65.9             | 78.4          | 80.1                | 71.7        | 81.3                 | 79.7                 |
| **C-Eval**    | Chinese        | 82.2             | 60.0          | 67.9                | 65.2        | 80.9                 | 78.0                 |
| **CMMLU**     | Chinese        | 82.9             | 61.0          | 70.7                | 67.8        | 82.4                 | 81.6                 |
| **HumanEval** | Code           | 68.9             | 75.0          | 76.2                | 73.8        | 76.8                 | 81.1                 |
| **MBPP**      | Code           | 52.2             | 64.4          | 69.8                | 61.4        | 70.4                 | 72.0                 |
|   **LiveCodeBench  (0901-0401)**     | Code       | 18.8          | 25.0                | 30.5        | 18.3                 | 28.7                 | 32.5                 |
| **GSM8K**     | Math           | 81.9             | 87.9          | 93.2                | 84.1        | 90.8                 | 92.2                 |
| **Math**      | Math           | 40.6             | 49.8          | 48.5                | 32.6        | 52.7                 | 53.9                 |

</div>

#### Standard Benchmark (Models smaller than 16B)

<div align="center">

| Benchmark | Domain         | DeepSeek 7B Chat (SFT) | DeepSeekMoE 16B Chat (SFT) | DeepSeek-V2-Lite 16B Chat (SFT) |
|:-----------:|:----------------:|:------------------:|:---------------:|:---------------------:|
| **MMLU**      | English        | 49.7             | 47.2          | 55.7                |
| **BBH**       | English        | 43.1             | 42.2          | 48.1                |
| **C-Eval**    | Chinese        | 44.7             | 40.0          | 60.1                |
| **CMMLU**     | Chinese        | 51.2             | 49.3          | 62.5                |
| **HumanEval** | Code           | 45.1             | 45.7          | 57.3                |
| **MBPP**      | Code           | 39.0             | 46.2          | 45.8                |
| **GSM8K**     | Math           | 62.6             | 62.2          | 72.0                |
| **Math**      | Math           | 14.7             | 15.2          | 27.9                |

</div>

#### English Open Ended Generation Evaluation
We evaluate our model on AlpacaEval 2.0 and MTBench, showing the competitive performance of DeepSeek-V2-Chat-RL on English conversation generation. 
<p align="center">
  <img width="50%" src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/mtbench.png?raw=true" />
</p>

#### Chinese Open Ended Generation Evaluation
**Alignbench** (https://arxiv.org/abs/2311.18743)
<div align="center">

| **模型** | **开源/闭源** | **总分** | **中文推理** | **中文语言** |
| :---: | :---: | :---: | :---: | :---: |
| gpt-4-1106-preview | 闭源 | 8.01 | 7.73 | 8.29 |
| DeepSeek-V2 Chat (RL) | 开源 | 7.91 | 7.45 | 8.36 |
| erniebot-4.0-202404 (文心一言) | 闭源 | 7.89 | 7.61 | 8.17 |
| DeepSeek-V2 Chat (SFT) | 开源 | 7.74 | 7.30 | 8.17 |
| gpt-4-0613 | 闭源 | 7.53 | 7.47 | 7.59 |
| erniebot-4.0-202312 (文心一言) | 闭源 | 7.36 | 6.84 | 7.88 |
| moonshot-v1-32k-202404 (月之暗面) | 闭源 | 7.22 | 6.42 | 8.02 |
| Qwen1.5-72B-Chat (通义千问) | 开源 | 7.19 | 6.45 | 7.93 |
| DeepSeek-67B-Chat | 开源 | 6.43 | 5.75 | 7.11 |
| Yi-34B-Chat (零一万物) | 开源 | 6.12 | 4.86 | 7.38 |
| gpt-3.5-turbo-0613 | 闭源 | 6.08 | 5.35 | 6.71 |
| DeepSeek-V2-Lite 16B Chat | 开源 | 6.01 | 4.71 | 7.32 |

</div>

#### Coding Benchmarks
We evaluate our model on LiveCodeBench (0901-0401), a benchmark designed for live coding challenges. As illustrated, DeepSeek-V2 demonstrates considerable proficiency in LiveCodeBench, achieving a Pass@1 score that surpasses several other sophisticated models. This performance highlights the model's effectiveness in tackling live coding tasks.

<p align="center">
  <img width="50%" src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/code_benchmarks.png?raw=true">
</p>

## 5. Model Architecture
DeepSeek-V2 adopts innovative architectures to guarantee economical training and efficient inference： 
- For attention, we design MLA (Multi-head Latent Attention), which utilizes low-rank key-value union compression to eliminate the bottleneck of inference-time key-value cache, thus supporting efficient inference. 
- For Feed-Forward Networks (FFNs), we adopt DeepSeekMoE architecture, a high-performance MoE architecture that enables training stronger models at lower costs. 

<p align="center">
  <img width="90%" src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/architecture.png?raw=true" />
</p>

## 6. Chat Website
You can chat with the DeepSeek-V2 on DeepSeek's official website: [chat.deepseek.com](https://chat.deepseek.com/sign_in)

## 7. API Platform
We also provide OpenAI-Compatible API at DeepSeek Platform: [platform.deepseek.com](https://platform.deepseek.com/). Sign up for over millions of free tokens. And you can also pay-as-you-go at an unbeatable price.


<p align="center">
  <img width="40%" src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/model_price.png?raw=true">
</p>

## 8. How to run locally
**To utilize DeepSeek-V2 in BF16 format for inference, 80GB*8 GPUs are required.**
### Inference with Huggingface's Transformers
You can directly employ [Huggingface's Transformers](https://github.com/huggingface/transformers) for model inference.

#### Text Completion
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name = "deepseek-ai/DeepSeek-V2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# `max_memory` should be set based on your devices
max_memory = {i: "75GB" for i in range(8)}
# `device_map` cannot be set to `auto`
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="sequential", torch_dtype=torch.bfloat16, max_memory=max_memory, attn_implementation="eager")
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

#### Chat Completion
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name = "deepseek-ai/DeepSeek-V2-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# `max_memory` should be set based on your devices
max_memory = {i: "75GB" for i in range(8)}
# `device_map` cannot be set to `auto`
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="sequential", torch_dtype=torch.bfloat16, max_memory=max_memory, attn_implementation="eager")
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

messages = [
    {"role": "user", "content": "Write a piece of quicksort code in C++"}
]
input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
outputs = model.generate(input_tensor.to(model.device), max_new_tokens=100)

result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
print(result)
```

The complete chat template can be found within `tokenizer_config.json` located in the huggingface model repository.

An example of chat template is as belows:

```bash
<｜begin▁of▁sentence｜>User: {user_message_1}

Assistant: {assistant_message_1}<｜end▁of▁sentence｜>User: {user_message_2}

Assistant:
```

You can also add an optional system message:

```bash
<｜begin▁of▁sentence｜>{system_message}

User: {user_message_1}

Assistant: {assistant_message_1}<｜end▁of▁sentence｜>User: {user_message_2}

Assistant:
```
### Inference with SGLang (recommended)

[SGLang](https://github.com/sgl-project/sglang) currently supports MLA optimizations, FP8 (W8A8), FP8 KV Cache, and Torch Compile, offering the best latency and throughput among open-source frameworks. Here are some example commands to launch an OpenAI API-compatible server:

```bash
# BF16, tensor parallelism = 8
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V2-Chat --tp 8 --trust-remote-code

# BF16, w/ torch.compile (The compilation can take several minutes)
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V2-Lite-Chat --trust-remote-code --enable-torch-compile

# FP8, tensor parallelism = 8, FP8 KV cache
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V2-Chat --tp 8 --trust-remote-code --quant fp8 --kv-cache-dtype fp8_e5m2
```

After launching the server, you can query it with OpenAI API

```
import openai
client = openai.Client(
    base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

# Chat completion
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)
print(response)
```

### Inference with vLLM (recommended)
To utilize [vLLM](https://github.com/vllm-project/vllm) for model inference, please merge this Pull Request into your vLLM codebase: https://github.com/vllm-project/vllm/pull/4650.

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

max_model_len, tp_size = 8192, 8
model_name = "deepseek-ai/DeepSeek-V2-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, tensor_parallel_size=tp_size, max_model_len=max_model_len, trust_remote_code=True, enforce_eager=True)
sampling_params = SamplingParams(temperature=0.3, max_tokens=256, stop_token_ids=[tokenizer.eos_token_id])

messages_list = [
    [{"role": "user", "content": "Who are you?"}],
    [{"role": "user", "content": "Translate the following content into Chinese directly: DeepSeek-V2 adopts innovative architectures to guarantee economical training and efficient inference."}],
    [{"role": "user", "content": "Write a piece of quicksort code in C++."}],
]

prompt_token_ids = [tokenizer.apply_chat_template(messages, add_generation_prompt=True) for messages in messages_list]

outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)

generated_text = [output.outputs[0].text for output in outputs]
print(generated_text)
```

### LangChain Support
Since our API is compatible with OpenAI, you can easily use it in [langchain](https://www.langchain.com/).
Here is an example:

```
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model='deepseek-chat',
    openai_api_key=<your-deepseek-api-key>,
    openai_api_base='https://api.deepseek.com/v1',
    temperature=0.85,
    max_tokens=8000)
``` 
## 9. License
This code repository is licensed under [the MIT License](LICENSE-CODE). The use of DeepSeek-V2 Base/Chat models is subject to [the Model License](LICENSE-MODEL). DeepSeek-V2 series (including Base and Chat) supports commercial use.

## 10. Citation
```
@misc{deepseekv2,
      title={DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model}, 
      author={DeepSeek-AI},
      year={2024},
      eprint={2405.04434},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## 11. Contact
If you have any questions, please raise an issue or contact us at [service@deepseek.com](service@deepseek.com).
