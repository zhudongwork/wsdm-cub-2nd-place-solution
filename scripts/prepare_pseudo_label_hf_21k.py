import torch
import pandas as pd
import numpy as np


def load_pred(path):
    pred = torch.from_numpy(torch.load(path))
    return pred

pred = np.average(
    [
        load_pred(f"./outputs/stage1/s1_m0/pseudo_labels_hf_21k.pth"),
        load_pred(f"./outputs/stage1/s1_m1/pseudo_labels_hf_21k.pth"),
        load_pred(f"./outputs/stage1/s1_m3/pseudo_labels_hf_21k.pth"),
    ],
    axis=0,
    weights=[1, 1, 1]
)

df = pd.read_parquet("./artifacts/hf-21k.parquet")
df["winner_model_a"] = pred[:, 0]
df["winner_model_b"] = pred[:, 1]

df['abs'] = abs(df['winner_model_a'] - df['winner_model_b'])
df = df[df['abs'] > 0.02]
df = df.drop(columns=['abs'])  

df.to_parquet("./artifacts/pseudo_hf_21k.parquet", index=False)

