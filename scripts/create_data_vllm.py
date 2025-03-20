
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import json

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"

# 获取外部传入的参数
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-72B-Instruct")
parser.add_argument("--file_id", type=str, default="3")
args = parser.parse_args()




online_models = ["DeepSeek-V3", "yi-linghting",]
small_models = ['phi-4','llama-3.2-3b-instruct',  'internlm3-8b-instruct', 'gemma-2-9b-it', 'Llama-3.2-1B-Instruct']


tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

df = []
with open(f"./extra_dataset/lmsys-chat-1m/v{args.file_id}.jsonl", "r") as f:
    data = json.load(f)
    for item in data:
        if item['model'] == args.model_path:
            df.append(item)

df = pd.DataFrame(df)
# df = df.sample(24)
tp = 8
if args.model_path in small_models:
    tp = 2

llm = LLM(model=args.model_path, tensor_parallel_size=tp, gpu_memory_utilization=0.8, trust_remote_code=True, enforce_eager=True,)

def batch_infer(data):
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=4096)
    prompt_list = []
    doc_ids = []
    for i in range(len(data)):

        messages = [
            # {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": data[i]}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_list.append(text)
        

    outputs = llm.generate(prompt_list, sampling_params)

    new_questions = []
    for idx, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        new_questions.append(generated_text)
    return new_questions


result = batch_infer(df['prompt'].values)

df['response'] = result

save_path = args.model_path.replace("Qwen/", "")

os.makedirs(f'./extra_dataset/lmsys-chat-1m/v{args.file_id}-data/', exist_ok=True)
df.to_parquet(f'./extra_dataset/lmsys-chat-1m/v{args.file_id}-data/{save_path}.parquet', index=False)
