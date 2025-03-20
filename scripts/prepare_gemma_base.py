import torch

from transformers import AutoTokenizer

from human_pref.inference.modeling_gemma2_local import Gemma2ForSequenceClassification

# download https://www.kaggle.com/datasets/tascj0/lmsys-checkpoints-0-0805/data
model_name_or_path = "models/lmsys-checkpoints-0-0805"
model_name_or_path = "gemma-2-9b-it"

save_path = "uploads/lmsys-gemma-pretrain"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = Gemma2ForSequenceClassification.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    num_labels=3,
)
# state_dict = torch.load(checkpoint_path, "cpu")["model"]
state_dict = model.state_dict()
for idx, layer in enumerate(model.model.layers):
    gate_up_proj = state_dict[f"model.layers.{idx}.mlp.gate_up_proj.weight"]
    out_features, in_features = gate_up_proj.shape

    half_out_features = out_features // 2

    
    gate_proj_weight = gate_up_proj[:half_out_features, :]
    up_proj_weight = gate_up_proj[half_out_features:, :]

    state_dict[f"model.layers.{idx}.mlp.gate_proj.weight"] = gate_proj_weight
    state_dict[f"model.layers.{idx}.mlp.up_proj.weight"] = up_proj_weight
    
    del state_dict[f"model.layers.{idx}.mlp.gate_up_proj.weight"]

del state_dict['score.4.weight']
del state_dict['score.4.bias']


from human_pref.models.modeling_gemma2_fast import Gemma2ForSequenceClassification



tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = Gemma2ForSequenceClassification.from_pretrained(
    model_name_or_path,
    num_labels=2,
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2",
)

print(model.load_state_dict(state_dict, strict=False))
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)