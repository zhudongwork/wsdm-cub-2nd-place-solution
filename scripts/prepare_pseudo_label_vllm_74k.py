import torch
import pandas as pd
import numpy as np


def load_pred(path):
    pred = torch.from_numpy(torch.load(path))
    return pred

def run(id):
    pred_110k = np.average(
        [
            load_pred(f"./artifacts/stage3/s3_m0_hf_21k/pseudo_labels_vllm_v{id}.pth"),
            load_pred(f"./artifacts/stage3/s3_m3_hf_21k/pseudo_labels_vllm_v{id}.pth").numpy()[:, [1, 0]],
        ],
        axis=0,
        weights=[3.2,1]
    )

    df = pd.read_parquet(f"./artifacts/vllm-v{id}.parquet")
    df["winner_model_a"] = pred_110k[:, 0]
    df["winner_model_b"] = pred_110k[:, 1]

    df['abs'] = abs(df['winner_model_a'] - df['winner_model_b'])
    df = df[df['abs'] > 0.03]
    df = df.drop(columns=['abs'])  

    df.to_parquet("./artifacts/pseudo_vllm_v{id}.parquet", index=False)
    print(df.head())

run("1")
run("2")
run("3")