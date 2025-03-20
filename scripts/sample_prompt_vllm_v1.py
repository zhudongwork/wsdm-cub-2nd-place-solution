# %%
import pandas as pd
import random

# %%
df_en = pd.read_parquet("./extra_dataset/lmsys-chat-1m/lmsys_1m_unuse_df2_en_208k.parquet")
df_en.head()

# %%
df_other = pd.read_parquet("./extra_dataset/lmsys-chat-1m/lmsys_1m_unuse_df2_other_57k.parquet")
df_other = df_other.loc[:7000]
df_other.shape

# %%
df_en = df_en.loc[:3000]
df_en.shape

# %%
df = pd.concat([df_en, df_other])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.shape

# %%
df['language'].value_counts()

# %%
model_list = ['qwen_72b', 'llama3_72b', 'gemma2_27b', 'phi-4', 'QwQ-32B-Preview', 'DeepSeek-V3', 'internlm2_5-20b-chat', 'llama-3.2-3b-instruct', 'MiniMax', 'internlm3-8b-instruct', 'gemma-2-9b-it', 'Qwen2.5-7B-Instruct','other']

# %%
len(model_list)

# %%
df[['model_a', 'model_b']] = [random.sample(model_list, 2) for _ in range(len(df))]

# %%
df.head()

# %%
from tqdm import tqdm

# %%
data = []
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    data.append({
        "prompt": row['prompt2'],
        "model": row['model_a'],
        "type": "model_a"
    })
    data.append({
        "prompt": row['prompt2'],
        "model": row['model_b'],
        "type": "model_b"
    })

# %%
import json
with open("./extra_dataset/lmsys-chat-1m/v1.jsonl", 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)


