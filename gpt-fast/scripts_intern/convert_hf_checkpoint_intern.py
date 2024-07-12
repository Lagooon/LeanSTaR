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
        "model.tok_embeddings.weight": "tok_embeddings.weight",
        "model.layers.{}.attention.wqkv.weight": "layers.{}.attention.wqkv.weight",
        "model.layers.{}.attention.wo.weight": "layers.{}.attention.wo.weight",
        'model.layers.{}.feed_forward.w1.weight': 'layers.{}.feed_forward.w1.weight',
        "model.layers.{}.feed_forward.w3.weight": "layers.{}.feed_forward.w3.weight",
        "model.layers.{}.feed_forward.w2.weight": "layers.{}.feed_forward.w2.weight",
        "model.layers.{}.attention_norm.weight": "layers.{}.attention_norm.weight",
        "model.layers.{}.ffn_norm.weight": "layers.{}.ffn_norm.weight",
        "model.norm.weight": "norm.weight",
        "output.weight": "output.weight",
    }
    bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}

    def permute(w, n_head):
        dim = config.dim
        return (
            w.view(n_head, 2, config.head_dim // 2, dim)
            .transpose(1, 2)
            .reshape(config.head_dim * n_head, dim)
        )

    merged_result = {}
    for file in sorted(bin_files):
        state_dict = safe_open(str(file), framework="pt", device='cpu')
        state_dict = {k: state_dict.get_tensor(k) for k in state_dict.keys()}
        merged_result.update(state_dict)
    final_result = {}
    for key, value in merged_result.items():
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
            print(config.n_local_heads)
            qkv = qkv.view(-1, 2+num_key_value_groups, head_dim ,config.dim)
            q = qkv[:, : num_key_value_groups, :, :].reshape(-1 ,config.dim)
            k = qkv[:, -2, :, :].reshape(-1 ,config.dim)
            v = qkv[:, -1, :, :].reshape(-1 ,config.dim)
            q = permute(q, config.n_head)
            k = permute(k, config.n_local_heads)
            final_result[key] = torch.cat([q, k, v])
    if "output.weight" not in final_result:
        final_result["output.weight"] = final_result["tok_embeddings.weight"]

    print(f"Saving checkpoint to {checkpoint_dir / 'model.pth'}")
    torch.save(final_result, checkpoint_dir / "model.pth")

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
