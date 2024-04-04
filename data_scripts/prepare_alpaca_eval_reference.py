import json
from typing import List

import tqdm
import fire

from datasets import load_dataset

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")


def main(
    output_path: str = "/nobackup/users/yikangs/zhiqings/dpo/alpaca_eval_reference.json",
):
    dataset = load_dataset("tatsu-lab/alpaca_farm", "alpaca_farm_evaluation")["eval"]

    outputs = []

    for example in tqdm.tqdm(dataset):
        outputs.append(
            {
                "instruction": example["instruction"] + "\n\n" + example["input"]
                if len(example["input"]) > 0
                else example["instruction"],
                "output": example["output"],
                "dataset": example["dataset"],
                "datasplit": example["datasplit"],
                "generator": example["generator"],
            }
        )

    # outputs = sorted(outputs, key=lambda x: len(x["prompt"]))
    print(f"Number of alpaca_eval: {len(outputs)}")

    max_prompt_length = -1

    for ex in tqdm.tqdm(outputs):
        input_length = len(tokenizer(ex["output"])["input_ids"])
        max_prompt_length = max(max_prompt_length, input_length)

    print(f"Max prompt length: {max_prompt_length}")
    # Max prompt length: 610

    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
