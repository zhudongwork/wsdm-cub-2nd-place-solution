# %%
import pandas as pd
import random

# %%
df_en = pd.read_parquet("./extra_dataset/lmsys-chat-1m/lmsys_1m_unuse_df2_en_424k.parquet")

# %%
df_en.head()

# %%
df_en['prompt2'] = df_en['prompt'].apply(lambda x: x[0])

# %%
df_en['prompt_len'] = df_en['prompt2'].apply(lambda x: len(x.split()))

# %%
df_en['prompt_len'].median()

# %%
df_en = df_en[df_en['prompt_len'] > 18]

# %%
df_en.shape

# %%
df_en = df_en.sample(frac=1).reset_index(drop=True)

# %%
df_en.to_parquet("./extra_dataset/lmsys-chat-1m/lmsys_1m_unuse_df2_en_208k.parquet", index=False)

# %%
df_other = pd.read_parquet("./extra_dataset/lmsys-chat-1m/lmsys_1m_unuse_df2_other_103k.parquet")
df_other.shape

# %%
df_other['prompt2'] = df_other['prompt'].apply(lambda x: x[0])
df_other['prompt_len'] = df_other['prompt2'].apply(lambda x: len(x))

# %%
df_other['prompt_len'].median()

# %%
df_other = df_other[df_other['prompt_len'] >= 45]

# %%
df_other.shape

# %%
df_other = df_other.sample(frac=1, random_state=42).reset_index(drop=True)

# %%
df_other.to_parquet("./extra_dataset/lmsys-chat-1m/lmsys_1m_unuse_df2_other_57k.parquet", index=False)


