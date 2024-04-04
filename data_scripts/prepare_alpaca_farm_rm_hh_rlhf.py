import json
from typing import List

import random
import tqdm
import fire

from datasets import load_dataset

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")


def find_shared_prefix(str1, str2):
    """
    Find the shared prefix between two strings.

    Args:
        str1 (str): The first string.
        str2 (str): The second string.

    Returns:
        int: The index position where the shared prefix ends in the first string.
    """
    # Find the minimum length of both strings
    min_length = min(len(str1), len(str2))

    max_common_prefix = 0
    # Loop through characters to find where they start to differ
    for i in range(min_length):
        if str1[i] != str2[i]:
            max_common_prefix = i
            break

    while (
        not str1[:max_common_prefix].endswith("Assistant:")
        # Force the prefix to end with "Assistant:"
    ) and max_common_prefix > 0:
        max_common_prefix -= 1

    return max_common_prefix


def split_example(example):
    """
    Split an example into a shared prefix and the remaining parts of the strings.

    Args:
        example (dict): A dictionary containing 'chosen' and 'rejected' keys with strings as values.

    Returns:
        dict: A dictionary with keys 'query', 'output_1', and 'output_2'.
    """
    chosen = example["chosen"]
    rejected = example["rejected"]

    # Find the index where the shared prefix ends
    shared_index = find_shared_prefix(chosen, rejected)

    # Split the strings
    query = chosen[:shared_index].strip()
    output_1 = chosen[shared_index:].strip()
    output_2 = rejected[shared_index:].strip()

    # Return the result as a dictionary
    return {"query": query, "output_1": output_1, "output_2": output_2}


def main(
    # output_path: str = "/nobackup/users/yikangs/zhiqings/dpo/alpaca_rm_10k.json",
    output_path: str = "/nobackup/users/yikangs/zhiqings/dpo/hh_rlhf_pm_10k.json",
    preference_model_augmentation: bool = True,
):
    dataset = load_dataset(
        "Anthropic/hh-rlhf",
        split="train",
    )

    outputs = []

    for example in tqdm.tqdm(dataset):
        example = split_example(example)
        instruction = example["query"]
        input_ = ""
        output_1 = example["output_1"]
        output_2 = example["output_2"]
        preference = 1

        shuffle_0_1 = random.random() < 0.5

        outputs.append(
            {
                "instruction": instruction,
                "input": input_,
                "output_1": output_1 if shuffle_0_1 else output_2,
                "output_2": output_2 if shuffle_0_1 else output_1,
                "preference": preference if shuffle_0_1 else 3 - preference,
            }
        )

    random.shuffle(outputs)

    if preference_model_augmentation:
        new_outputs = outputs[:]

        for ex in outputs:
            new_outputs.append(
                {
                    "instruction": ex["instruction"],
                    "input": ex["input"],
                    "output_1": ex["output_2"],
                    "output_2": ex["output_1"],
                    "preference": 3 - ex["preference"],
                }
            )

        outputs = new_outputs

    print(f"Number of alpaca_farm RM: {len(outputs)}")

    # Check max input (instruction + input + output_prefix + 32) length and max output length

    max_total_length = 768

    filtered_outputs = []

    for ex in tqdm.tqdm(outputs):
        input_length = len(tokenizer(ex["instruction"] + ex["input"])["input_ids"]) + 32
        output_length = len(tokenizer(ex["output_1"])["input_ids"]) + len(
            tokenizer(ex["output_2"])["input_ids"]
        )

        if input_length + output_length <= max_total_length:
            filtered_outputs.append(ex)

    print(f"Number of filtered alpaca_farm RM: {len(filtered_outputs)}")

    with open(output_path, "w") as f:
        json.dump(filtered_outputs, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
