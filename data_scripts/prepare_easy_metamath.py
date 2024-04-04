import glob
import json
import os
import random
import re
import tqdm
from typing import List

import fire

from datasets import load_dataset

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llemma_34b")


def main(
    levels: List[str] = ["Level 1", "Level 2", "Level 3"],
    output_path: str = "/nobackup/users/yikangs/zhiqings/math/train_1_2_3_metamath.json",
    splitter: str = "\n\n",
):
    metamath_data_pattern = "/nobackup/users/yikangs/zhiqings/math/metamath_oriquestion_348309_level1-3.json"

    outputs = []

    with open(metamath_data_pattern, "r") as f:
        data = json.load(f)

        print("Number of MetaMath examples: ", len(data))

        for ex in data:
            if "MATH" in ex["type"] and ex["level"] in levels:
                outputs.append(
                    {
                        "input": ex["query"]
                        + " Please answer the question in the MetaMath format.",
                        "output": ex["response"].split("The answer is:")[0].strip(),
                        "output_prefix": "",
                        "is_eos": False,
                    }
                )

    print("Number of MetaMath annotations: ", len(outputs))

    # Remove repeated inputs >= 5 times
    unique_inputs = {}
    filtered_outputs = []

    for ex in tqdm.tqdm(outputs):
        if ex["input"] not in unique_inputs:
            unique_inputs[ex["input"]] = 1
            filtered_outputs.append(ex)
        elif unique_inputs[ex["input"]] < 5:
            unique_inputs[ex["input"]] += 1
            filtered_outputs.append(ex)

    outputs = filtered_outputs

    print("Number of unique MetaMath annotations: ", len(outputs))

    random.shuffle(outputs)

    # Check max input (input + output_prefix) length and max output length

    # max_input_length = 768
    # max_output_length = 256
    max_total_length = 768

    filtered_output = []

    for ex in tqdm.tqdm(outputs):
        input_length = len(tokenizer(ex["input"] + ex["output_prefix"])["input_ids"])
        output_length = len(tokenizer(ex["output"])["input_ids"])

        # if input_length <= max_input_length and output_length <= max_output_length:
        #     filtered_output.append(ex)

        if input_length + output_length <= max_total_length:
            filtered_output.append(ex)

    print(f"Number of filtered MetaMath annotations: {len(filtered_output)}")

    with open(output_path, "w") as f:
        json.dump(filtered_output, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
