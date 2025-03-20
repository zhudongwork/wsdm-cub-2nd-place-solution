import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

df = pd.read_parquet("data/train.parquet")
df['winner_model_a'] = df['winner'].apply(lambda x: 1 if x == "model_a" else 0)
df['winner_model_b'] = df['winner'].apply(lambda x: 1 if x == "model_b" else 0)
sgkf = StratifiedGroupKFold(n_splits=5, random_state=0, shuffle=True)
group_id = df["prompt"]
label_id = df["winner_model_a winner_model_b".split()].values.argmax(1)
splits = list(sgkf.split(df, label_id, group_id))

df["fold"] = -1
for fold, (_, valid_idx) in enumerate(splits):
    df.loc[valid_idx, "fold"] = fold

print(df["fold"].value_counts())
df.to_parquet("artifacts/dtrainval.parquet", index=False)


# download https://www.kaggle.com/datasets/nbroad/wsdm-open-models-nbroad/data
hf1 = pd.read_parquet("extra_dataset/hf-open/hf-open-models-v1.parquet")
hf2 = pd.read_parquet("extra_dataset/hf-open/hf-open-models-v2.parquet")
df = pd.concat([hf1, hf2])
df.to_parquet("artifacts/hf-21k.parquet", index=False)