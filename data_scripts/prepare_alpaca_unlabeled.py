import json
from typing import List

import tqdm
import fire

from datasets import load_dataset

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")

alpaca_prompt_dict = {
    "prompt_noinputs": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "prompt_inputs": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
}


def format_prompt(example: dict, prompt_dict: dict) -> str:
    """Formats a prompt with a prompt_dict formatter.

    Args:
        example: A dict-like object with required keys "instruction" and "input"
        prompt_dict: Dictionary containing the keys "prompt_noinputs" and "prompt_inputs" which have
            placeholders corresponding to the keys from `example`. E.g. "{instruction}".

    Returns:
        A formatted prompt string.

    Examples
    --------
    >>> format_prompt(dict(instruction="test", input=""), prompt_dict=dict(prompt_noinputs="prompt {instruction} "))
    "prompt test"
    """
    assert (
        "instruction" in example and "input" in example
    ), "Internal error: example missing required keys."

    if example["input"] is None or len(example["input"]) == 0:
        formatted_prompt = prompt_dict["prompt_noinputs"].format_map(example)
    else:
        formatted_prompt = prompt_dict["prompt_inputs"].format_map(example)

    return formatted_prompt


def main(
    output_path: str = "/nobackup/users/yikangs/zhiqings/dpo/alpaca_unlabeled_20k.json",
):
    dataset = load_dataset("tatsu-lab/alpaca_farm", "alpaca_instructions")["unlabeled"]

    outputs = []

    for example in tqdm.tqdm(dataset):
        prompt = format_prompt(example, alpaca_prompt_dict)

        outputs.append(
            {
                "prompt": prompt,
                "instruction": example["instruction"],
                "input": example["input"],
            }
        )

    # outputs = sorted(outputs, key=lambda x: len(x["prompt"]))
    print(f"Number of alpaca unlabeled: {len(outputs)}")

    max_prompt_length = -1

    for ex in tqdm.tqdm(outputs):
        input_length = len(tokenizer(ex["prompt"])["input_ids"])
        max_prompt_length = max(max_prompt_length, input_length)

    print(f"Max prompt length: {max_prompt_length}")
    # Max prompt length: 610

    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
