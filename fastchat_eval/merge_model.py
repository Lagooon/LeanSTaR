# Load the model.
# Note: It can take a while to download LLaMA and add the adapter modules.
# You can also use the 13B model by loading in 4bits.

import torch
import fire
from peft import PeftModel
from transformers import AutoModelForCausalLM, LlamaTokenizer


def main(
    adapters_name: str,
    output_dir: str,
    model_name: str = "EleutherAI/llemma_34b",
    tokenizer_name: str = "hf-internal-testing/llama-tokenizer",
):
    tok = LlamaTokenizer.from_pretrained(tokenizer_name)
    tok.save_pretrained(output_dir)

    print(f"Starting to load the model {model_name} into memory")

    m = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    m = PeftModel.from_pretrained(m, adapters_name)
    m = m.merge_and_unload()

    print("Model loaded. Starting to save the merged model.")

    m.save_pretrained(output_dir)

    print("Model saved.")


if __name__ == "__main__":
    fire.Fire(main)
