import torch

from transformers import AutoTokenizer

from human_pref.inference.modeling_llama import LlamaForSequenceClassification
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str)
parser.add_argument("--checkpoint_path", type=str)
args = parser.parse_args()


model_name_or_path = "uploads/lmsys-llama-pretrain"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = LlamaForSequenceClassification.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
)
state_dict = torch.load(args.checkpoint_path, "cpu")["model"]
for idx, layer in enumerate(model.model.layers):
    state_dict[f"model.layers.{idx}.mlp.gate_up_proj.weight"] = torch.cat(
        [
            state_dict[f"model.layers.{idx}.mlp.gate_proj.weight"],
            state_dict[f"model.layers.{idx}.mlp.up_proj.weight"],
        ],
        dim=0,
    )
print(model.load_state_dict(state_dict, strict=False))
tokenizer.save_pretrained(args.save_path)
model.save_pretrained(args.save_path)
