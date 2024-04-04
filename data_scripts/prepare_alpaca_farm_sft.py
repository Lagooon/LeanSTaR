import json
from typing import List

import random
import tqdm
import fire

from datasets import load_dataset

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")


def main(
    output_path: str = "/nobackup/users/yikangs/zhiqings/dpo/alpaca_sft_10k.json",
):
    dataset = load_dataset(
        "tatsu-lab/alpaca_farm",
        "alpaca_instructions",
        split="sft",
    )

    outputs = []

    for example in tqdm.tqdm(dataset):
        instruction = example["instruction"]
        input_ = example["input"]
        output = example["output"]

        outputs.append(
            {
                "instruction": instruction,
                "input": input_,
                "output": output,
            }
        )

    random.shuffle(outputs)

    print(f"Number of alpaca_farm SFT: {len(outputs)}")

    # Check max input (instruction + input + output_prefix + 32) length and max output length

    max_total_length = 768

    filtered_outputs = []

    for ex in tqdm.tqdm(outputs):
        input_length = len(tokenizer(ex["instruction"] + ex["input"])["input_ids"]) + 32
        output_length = len(tokenizer(ex["output"])["input_ids"])

        if input_length + output_length <= max_total_length:
            filtered_outputs.append(ex)

    print(f"Number of filtered alpaca_farm SFT: {len(filtered_outputs)}")

    with open(output_path, "w") as f:
        json.dump(filtered_outputs, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
