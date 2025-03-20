# %%
import pandas as pd
import random

# %%
df_en = pd.read_parquet("./extra_dataset/lmsys-chat-1m/lmsys_1m_unuse_df2_en_208k.parquet")
df_en.head()

# %%
df_other = pd.read_parquet("./extra_dataset/lmsys-chat-1m/lmsys_1m_unuse_df2_other_57k.parquet")
df_other2 = df_other.loc[:7000]
df_other = df_other.loc[7000*4:]
df_other.shape

# %%
df_en = df_en.loc[3000*3:3000*3+2000]
df_en.shape

# %%
df = pd.concat([df_en, df_other, df_other2])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.shape

# %%
df['language'].value_counts()

# %%
model_list = ['Qwen/Qwen2.5-72B-Instruct', 'Llama-3.3-70B-Instruct', 'gemma-2-27b-it', 'phi-4', 'DeepSeek-R1-Distill-Qwen-32B', 'Mistral-7B-Instruct-v0.3', 'internlm2_5-20b-chat', 'llama-3.2-3b-instruct', 'Hermes-3-Llama-3.1-8B', 'internlm3-8b-instruct', 'gemma-2-9b-it', 'DeepSeek-R1-Distill-Qwen-14B','Llama-3.2-1B-Instruct']

# %%
len(model_list)

# %%
df[['model_a', 'model_b']] = [random.sample(model_list, 2) for _ in range(len(df))]

# %%
df.head()

# %%
df['pid'] = df['prompt2'].map(hash)

# %%
# 根据pid去重
df = df.drop_duplicates(subset='pid')
df = df.reset_index(drop=True)
df.shape

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
with open("./extra_dataset/lmsys-chat-1m/v3.jsonl", 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)


