import json
import re
import sys
from pathlib import Path
from typing import Optional

import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from models.model import ModelArgs

from training_utils.checkpoint_hook import merge_tp_checkpoint
from transformers import AutoModelForCausalLM

@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf"),
    model_name: Optional[str] = None,
) -> None:
    if model_name is None:
        model_name = checkpoint_dir.name

    config = ModelArgs.from_name(model_name)
    print(model_name, config)
    print(f"Model config {config.__dict__}")

    from safetensors import safe_open

    # Load the json file containing weight mapping
    model_map_json = checkpoint_dir / "model.safetensors.index.json"

    assert model_map_json.is_file()

    with open(model_map_json) as json_map:
        bin_index = json.load(json_map)

    weight_map = {
        "tok_embeddings.weight": "model.tok_embeddings.weight",
        "layers.{}.attention.wqkv.weight": "model.layers.{}.attention.wqkv.weight",
        "layers.{}.attention.wo.weight": "model.layers.{}.attention.wo.weight",
        'layers.{}.feed_forward.w1.weight': 'model.layers.{}.feed_forward.w1.weight',
        "layers.{}.feed_forward.w3.weight": "model.layers.{}.feed_forward.w3.weight",
        "layers.{}.feed_forward.w2.weight": "model.layers.{}.feed_forward.w2.weight",
        "layers.{}.attention_norm.weight": "model.layers.{}.attention_norm.weight",
        "layers.{}.ffn_norm.weight": "model.layers.{}.ffn_norm.weight",
        "norm.weight": "model.norm.weight",
        "output.weight": "output.weight",
    }
    #bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}

    def permute(w, n_head):
        dim = config.dim
        return (
            w.view(n_head, config.head_dim // 2, 2, dim)
            .transpose(1, 2)
            .reshape(config.head_dim * n_head, dim)
        )

    all_state_dict = []
    checkpoint_path = "/data/user_data/shengyuf/Lean/checkpoints/internlm2-7b_stars_epoch-3_lr-3e-5-plus/epoch_1_step_497_rank_0.pt"
    pattern = checkpoint_path.replace(f"_rank_{0}", "_rank_*")
    for i in range(8):
        ckpt_file_path = pattern.replace("*", str(i))
        model_state_dict = torch.load(
            ckpt_file_path, map_location="cpu", mmap=True
        )["model"]
        for key in model_state_dict:
            model_state_dict[key] = model_state_dict[key].cpu()
        all_state_dict.append(model_state_dict)

    merged_result = merge_tp_checkpoint(all_state_dict)
    final_result = {}
    for key, value in merged_result.items():
        print(key)
        if "layers" in key:
            abstract_key = re.sub(r'\.(\d+)', '.{}', key)
            layer_num = re.search(r'\d+', key).group(0)
            new_key = weight_map[abstract_key]
            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = weight_map[key]

        final_result[new_key] = value

    for key in tuple(final_result.keys()):
        if "wqkv" in key:
            qkv = final_result[key]
            head_dim = config.dim // config.n_head
            num_key_value_groups = config.n_head // config.n_local_heads
            q = qkv[: config.head_dim * config.n_head, :]
            k = qkv[config.head_dim * config.n_head: config.head_dim * (config.n_head + config.n_local_heads), :]
            v = qkv[-config.head_dim * config.n_local_heads :, :]
            q = permute(q, config.n_head)
            k = permute(k, config.n_local_heads)
            q = q.view(-1, num_key_value_groups, head_dim ,config.dim)
            k = k.view(-1, 1, head_dim ,config.dim)
            v = v.view(-1, 1, head_dim ,config.dim)
            qkv = torch.cat([q, k, v], dim=1)
            final_result[key] = qkv.reshape(-1 ,config.dim)
    #if "output.weight" not in final_result:
    #    final_result["output.weight"] = final_result["tok_embeddings.weight"]

    #print(f"Saving checkpoint to {checkpoint_dir / 'model_intern.pth'}")
    #torch.save(final_result, checkpoint_dir / "model_intern.pth")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )
    model.load_state_dict(final_result, strict=False)
    model.save_pretrained(checkpoint_dir / "stars")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert HuggingFace checkpoint.')
    parser.add_argument('--checkpoint_dir', type=Path, default=Path("checkpoints/meta-llama/llama-2-7b-chat-hf"))
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument("--target_precision", type=str, default="fp32")

    args = parser.parse_args()
    convert_hf_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
    )
