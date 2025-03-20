
#!/bin/bash


# model list
models=(
    'Qwen/Qwen2.5-72B-Instruct'
    'Llama-3.3-70B-Instruct'
    'gemma-2-27b-it'
    'phi-4'
    'DeepSeek-R1-Distill-Qwen-32B'
    'Mistral-7B-Instruct-v0.3'
    'internlm2_5-20b-chat'
    'llama-3.2-3b-instruct'
    'Hermes-3-Llama-3.1-8B'
    'internlm3-8b-instruct'
    'gemma-2-9b-it'
    'DeepSeek-R1-Distill-Qwen-14B'
    'Llama-3.2-1B-Instruct'
)


# file list
files=(
    '1'
    '2'
    '3'
)

for model in "${models[@]}"
do
    for file in "${files[@]}"
    do
        echo "Processing model: $model with file: $file"
        python create_data_vllm.py --model_path "$model" --file_id "$file"
        sleep 10  
    done
done