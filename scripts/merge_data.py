# %%
import os
import pandas as pd
import json

# %%
def merge(id):
    df = []
    for path in os.listdir(f"extra_dataset/lmsys-chat-1m/v{id}-data"):
        # print(path)
        temp = pd.read_parquet(f"extra_dataset/lmsys-chat-1m/v{id}-data/{path}")
        df.append(temp)
    df = pd.concat(df)

    df['pid'] = df['prompt'].map(hash)
    pid_to_count = dict(df["pid"].value_counts())
    df["n_pid"] = df["pid"].map(pid_to_count)

    pids = df['pid'].unique()
    df_train = []
    for pid in pids:
        df_pid = df[df['pid'] == pid]
        a = df_pid[df_pid['type'] == "model_a"].iloc[0]
        b = df_pid[df_pid['type'] == "model_b"].iloc[0]
        df_train.append({
            "prompt": a.prompt,
            "response_a": a.response,
            "response_b": b.response,
            "model_a": a.model,
            "model_b": b.model,
            })
    
    df_train = pd.DataFrame(df_train)
    df_train['winner_model_a'] = 1
    df_train['winner_model_b'] = 0
    df_train.to_parquet(f"artifacts/vllm-v{id}.parquet", index=False)

    df_train['response_a'], df_train['response_b'] = df_train['response_b'], df_train['response_a']
    df_train.to_parquet(f"artifacts/vllm-v{id}-swap.parquet", index=False)


# %%
merge("1")
merge("2")
merge("3")