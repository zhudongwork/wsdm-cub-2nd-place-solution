# WSDM Cup - Multilingual Chatbot Arena

[Competition](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/leaderboard)

## Requirements

### Hardware

A100 SXM 80G x4

### Software

Base Image

```
nvcr.io/nvidia/pytorch:24.04-py3
```

Packages

```
detectron2==0.6
transformers==4.43.3
datasets==2.19.0
flash-attn==2.6.2
pip install flash-attn==2.6.2 --no-build-isolation
pip install torch-optimi==0.2.1
```

## base model

Download the well-trained models that ranked top 2 in the previous competition as the model initialization. 

(gemma)https://www.kaggle.com/datasets/tascj0/lmsys-checkpoints-0-0805/data

(llama)https://www.kaggle.com/datasets/tascj0/lmsys-checkpoints-3-0805

Convert the model and initialize it.
```
python scripts/prepare_gemma_base.py
python scripts/prepare_llama_base.py
```

## data

Process the dataset of the WSDM Cup competition. 
```
python scripts/prepare_dataset.py
```

## Fine-Tuning Base Models

```
torchrun --nproc_per_node=4 main.py configs/stage1/s1_m0.py
torchrun --nproc_per_node=4 main.py configs/stage1/s1_m1.py
torchrun --nproc_per_node=4 main.py configs/stage1/s1_m3.py
```

## Generate pseudo labels


```
torchrun --nproc_per_node=4 main.py configs/stage1_generate_pseudo_labels/m0_hf_21k.py --load-from ./outputs/stage1/s1_m0/update_last.pth --eval-only --out ./outputs/stage1/s1_m0/pseudo_labels_hf_21k.pth
torchrun --nproc_per_node=4 main.py configs/stage1_generate_pseudo_labels/m1_hf_21k.py --load-from ./outputs/stage1/s1_m1/update_last.pth --eval-only --out ./outputs/stage1/s1_m1/pseudo_labels_hf_21k.pth
torchrun --nproc_per_node=4 main.py configs/stage1_generate_pseudo_labels/m3_hf_21k.py --load-from ./outputs/stage1/s1_m3/update_last.pth --eval-only --out ./outputs/stage1/s1_m3/pseudo_labels_hf_21k.pth
```
merge output results
```
python scripts/prepare_pseudo_label_hf_21k.py
```

## Fine-Tuning with Pseudo-Labels and WSDM data

```
torchrun --nproc_per_node=4 main.py configs/stage3/s3_m0_hf_21k.py
torchrun --nproc_per_node=4 main.py configs/stage3/s3_m3_hf_21k.py
```

## Online Inference (submission 1) 

prepare model forsubmission.
```
python scripts/prepare_gemma2_for_submission.py --save_path uploads/lmsys-pretrain-mid-pseudo-v3  --checkpoint_path outputs/stage3/s3_m0_hf_21k/update_last.pth
python scripts/prepare_llama3_for_submission.py --save_path uploads/llama-lmsys-pretrain-mid-pseudo-4096  --checkpoint_path outputs/stage3/s3_m3_hf_21k/update_last.pth
```

kaggle notebook link: https://www.kaggle.com/code/zhudong1949/lmsys-0130?scriptVersionId=219956474
local file: inference/lmsys-0130.ipynb



## Expanded Pseudo-Labels

download data: https://www.kaggle.com/datasets/gmhost/lmsys-chat-1m

### get prompt from lmsys-1m
```
python scripts/lmsys_1m.py
```
### Response Pair Generation
```
python scripts/sample_prompt_vllm_v1.py
python scripts/sample_prompt_vllm_v2.py
python scripts/sample_prompt_vllm_v3.py

bash scripts/run_vllm.sh

python scripts/merge_data.py
```

### Pseudo-Label Generation
```
torchrun --nproc_per_node=4 main.py configs/stage1_generate_pseudo_labels/m0_vllm_v1.py --load-from ./outputs/stage3/s3_m0_hf_21k/update_last.pth --eval-only --out ./outputs/stage3/s3_m0_hf_21k/pseudo_labels_vllm_v1.pth
torchrun --nproc_per_node=4 main.py configs/stage1_generate_pseudo_labels/m3_vllm_v1.py --load-from ./outputs/stage3/s3_m3_hf_21k/update_last.pth --eval-only --out ./outputs/stage3/s3_m3_hf_21k/pseudo_labels_vllm_v1.pth

torchrun --nproc_per_node=4 main.py configs/stage1_generate_pseudo_labels/m0_vllm_v2.py --load-from ./outputs/stage3/s3_m0_hf_21k/update_last.pth --eval-only --out ./outputs/stage3/s3_m0_hf_21k/pseudo_labels_vllm_v2.pth
torchrun --nproc_per_node=4 main.py configs/stage1_generate_pseudo_labels/m3_vllm_v2.py --load-from ./outputs/stage3/s3_m3_hf_21k/update_last.pth --eval-only --out ./outputs/stage3/s3_m3_hf_21k/pseudo_labels_vllm_v2.pth

torchrun --nproc_per_node=4 main.py configs/stage1_generate_pseudo_labels/m0_vllm_v3.py --load-from ./outputs/stage3/s3_m0_hf_21k/update_last.pth --eval-only --out ./outputs/stage3/s3_m0_hf_21k/pseudo_labels_vllm_v3.pth
torchrun --nproc_per_node=4 main.py configs/stage1_generate_pseudo_labels/m3_vllm_v3.py --load-from ./outputs/stage3/s3_m3_hf_21k/update_last.pth --eval-only --out ./outputs/stage3/s3_m3_hf_21k/pseudo_labels_vllm_v3.pth

python scripts/prepare_pseudo_label_vllm_74k.py
```
## Expanded Pretraining
```
torchrun --nproc_per_node=4 main.py configs/stage2/m0_lmsys_mid2_pseudo_final.py
torchrun --nproc_per_node=4 main.py configs/stage2/m3_lmsys_mid2_pseudo_final.py
```

## Final Training
```
torchrun --nproc_per_node=4 main_step.py configs/stage3/m0_lmsys_mid2_final_step.py
torchrun --nproc_per_node=4 main_step.py configs/stage3/m3_lmsys_mid2_final_step.py
```
## Inference (submission 2, best score)
```
python scripts/prepare_gemma2_for_submission.py --save_path uploads/m0-lmsys-mid2-final-step  --checkpoint_path outputs/stage3/m0_lmsys_mid2_final_step/update_last.pth
python scripts/prepare_llama3_for_submission.py --save_path uploads/m3-lmsys-mid2-final-step  --checkpoint_path outputs/stage3/m3_lmsys_mid2_final_step/update_last.pth
```
kaggle notebook link: https://www.kaggle.com/code/zhudong1949/lmsys-0201
local file: inference/lmsys-0201.ipynb
