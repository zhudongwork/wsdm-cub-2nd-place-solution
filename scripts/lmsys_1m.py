# %%
import pandas as pd

# %%
df = []
for i in range(0, 6):
    temp = pd.read_parquet(f"./extra_dataset/lmsys-chat-1m/000{i}.parquet")
    temp = temp.drop(['openai_moderation', 'redacted'],axis=1)
    df.append(temp)
df = pd.concat(df)
df.head()

# %%
df.shape

# %%
import json
from tqdm import tqdm
from collections import defaultdict
import pandas as pd


def separate_user_assistant(conversation):
    res = defaultdict(list)
    for c in conversation:
        assert c["role"] in {"user", "assistant"}
        assert type(c["content"]) == str
        res[c["role"]].append(c["content"])
    assert len(res["user"]) > 0
    assert len(res["user"]) == len(res["assistant"])
    return res


sep = df["conversation"].map(separate_user_assistant)
df["prompt"] = sep.map(lambda x: x["user"])
df["response"] = sep.map(lambda x: x["assistant"])

# Generate a unique id for each prompt
df["pid"] = df["prompt"].map(str).map(hash)
pid_to_count = dict(df["pid"].value_counts())
df["n_pid"] = df["pid"].map(pid_to_count)

# remove
print("BEFORE removal", len(df))
df["prompt_response_hash"] = (df["prompt"].map(str) + df["response"].map(str)).map(hash)
df = df.drop_duplicates(["prompt_response_hash"])
print("AFTER removal", len(df))

df = df.drop(columns=["prompt_response_hash"])

# %%
df.to_parquet("./extra_dataset/lmsys-chat-1m/lmsys-1m-duplicates.parquet", index=False)

# %%
df.head()

# %%
df = df.sample(len(df), random_state=42).reset_index(drop=True)

# The result dictionary
res = dict(id=[], model_a=[], model_b=[], prompt=[], response_a=[], response_b=[])
cand_df = df[df["n_pid"] > 1]
cand_pids = cand_df["pid"].unique()

# %%
cand_df.shape[0]

# %%
cand_df['language'].value_counts()

# %%
unuse_df = df[df["n_pid"] == 1]

# %%
unuse_df['language'].value_counts()

# %%
unuse_df['first_prompt'] = unuse_df['prompt'].apply(lambda x: x[0])

# %%
unuse_df

# %%
unuse_df['first_prompt_id'] = unuse_df['first_prompt'].map(hash)

# %%
unuse_df

# %%
hf1 = pd.read_parquet("./extra_dataset/hf-open/hf-open-models-v1.parquet")
hf2 = pd.read_parquet("./extra_dataset/hf-open/hf-open-models-v2.parquet")
hf = pd.concat([hf1, hf2])
hf.head()

# %%
hf['prompt_id'] = hf['prompt'].map(hash)

# %%
hf

# %%
hf_use_id = set(unuse_df['first_prompt_id']).intersection(set(hf['prompt_id']))
len(hf_use_id)

# %%
unuse_df2 = unuse_df[~unuse_df['first_prompt_id'].isin(hf_use_id)]

# %%
unuse_df2

# %%
unuse_df2 = unuse_df2[['model', "prompt", "language"]]

# %%
unuse_df2_en = unuse_df2[unuse_df2['language'] == "English"]
unuse_df2_en = unuse_df2_en.reset_index(drop=True)
unuse_df2_en.shape

# %%
unuse_df2_other = unuse_df2[unuse_df2['language'] != "English"]
unuse_df2_other = unuse_df2_other.reset_index(drop=True)
unuse_df2_other.shape

# %%
unuse_df2_en.to_parquet("./extra_dataset/lmsys-chat-1m/lmsys_1m_unuse_df2_en_424k.parquet", index=False)
unuse_df2_other.to_parquet("./extra_dataset/lmsys-chat-1m/lmsys_1m_unuse_df2_other_103k.parquet", index=False)


