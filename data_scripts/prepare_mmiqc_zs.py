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

import re

MAX_QUERY_LENGTH = 512
HINT_QUERY_PREFIX = "Please give me a hint for the following problem.\n\n"
METAMATH_QUERY_PREFIX = 'Please solve the following problem and put your answer at the end with "# Answer\\n\\n{answer}".\n\n'


def reformat_output(output: str) -> str:
    item = {"output": output}
    for _ in range(10):
        item["output"] = item["output"].replace("  ", " ")

    item["output"] = item["output"].replace("\\[ ", "\\[")
    item["output"] = item["output"].replace(" \\]", "\\]")

    item["output"] = item["output"].replace("\n", "\n\n")
    item["output"] = item["output"].replace(".\n", ".\n\n")
    item["output"] = item["output"].replace(". ", ".\n\n")
    item["output"] = item["output"].replace(".$", ".$\n\n")
    item["output"] = item["output"].replace("\\]", "\\]\n\n")

    item["output"] = item["output"].replace("\n\n\\\\\n\n", " \\\\\n\n")
    item["output"] = item["output"].replace("\n\n&=\n\n", "\n\n&= ")

    for _ in range(10):
        item["output"] = item["output"].replace("\n\n\n", "\n\n")
        item["output"] = item["output"].replace("\n ", "\n")

    item["output"] = item["output"].replace(",\n\n", ", ")
    item["output"] = item["output"].replace("\n\n$", " $")
    item["output"] = item["output"].replace("\n\n\\]", " \\]")
    item["output"] = item["output"].replace("\\[\n\n", "\\[ ")
    item["output"] = item["output"].replace("\n*\n", "")
    item["output"] = item["output"].replace("\n.\n", "")
    item["output"] = item["output"].replace("$ $", "$$")
    item["output"] = item["output"].strip()

    # for i in "abcdefghijklmnopqrstuvwxyz:":
    #     item["output"] = item["output"].replace(f"{i}\n\n\\[", f"{i} \\[")

    for common_abbreviation in [
        "Mr.",
        "Dr.",
        "Ms.",
        "Mrs.",
        "St.",
        "Prof.",
    ]:
        item["output"] = item["output"].replace(
            common_abbreviation + "\n\n", common_abbreviation + " "
        )
    return item["output"]


def main(
    levels: str = "Level 1, Level 2, Level 3, Level 4, Level 5",
    output_path: str = "/home/wliu/longhui/llms-all/ScalableMath_sun-main/data/train_1_2_3_4_5_metamath_v5.json",
    pruned_output_path: str = "/home/wliu/longhui/llms-all/ScalableMath_sun-main/data/train_1_2_3_4_5_metamath_v5_pruned.json",
    pruned_numbers: int = 8,
    metamath_path: str = "/home/wliu/longhui/llms-all/ScalableMath_sun-main/data/MetaMathQA-395K.json",
    math_path: str = (
        "/home/wliu/longhui/llms-all/ScalableMath_sun-main/data/MATH_train-cleaned_processed.json"
    ),
    epoch: int = 1,
    print_examples: bool = False,
):
    levels = levels.split(", ")
    math_dataset = {}

    mmiqc_dataset = load_dataset("Vivacem/MMIQC")["train"]

    # use map to speed up with multi-threading
    mmiqc_stex_dataset = mmiqc_dataset.filter(
        lambda x: x["source"] == "stackexchange-math",
        keep_in_memory=True,
        num_proc=64,
    )
    mmiqc_stex_dataset = mmiqc_stex_dataset.map(
        lambda x: {
            "query": (
                HINT_QUERY_PREFIX + x["instruction"]
                if x["output"].startswith("Hint")
                else x["instruction"]
            ),
            "output": reformat_output(x["output"]),
            "type": x["source"],
        },
        remove_columns=["source"],
        keep_in_memory=True,
        num_proc=64,
    )
    mmiqc_stex_dataset = mmiqc_stex_dataset.filter(
        lambda x: len(tokenizer.encode(x["query"])) < MAX_QUERY_LENGTH,
        keep_in_memory=True,
        num_proc=64,
    )
    mmiqc_stex_dataset = list(mmiqc_stex_dataset)

    print("MMIQC StackExchange: ", len(mmiqc_stex_dataset))

    math_dataset["train"] = []
    math_dataset["test"] = []

    with open(metamath_path, "r") as f:
        data = json.load(f)

    with open(math_path, "r") as f:
        ori_math = json.load(f)

    select_list = []
    question_set = set()
    for item in ori_math:
        if item["level"] in levels:
            question_set.add(item["question"])
    for item in data:
        if "MATH" in item["type"]:
            if item["original_question"] in question_set:
                select_list.append(item)
        else:
            select_list.append(item)

    print("Original: ", len(data))
    print("Before pruning: ", len(select_list))

    random.shuffle(select_list)

    for item in select_list:
        split_ans = item["response"].split("The answer is: ")
        item["output"] = (
            split_ans[0].split("####")[0].strip() + "\n\n# Answer\n\n" + split_ans[1]
        )
        item["output"] = reformat_output(item["output"])
        item["query"] = METAMATH_QUERY_PREFIX + item["query"]

    pruned_metamath = []
    num_dict = {}
    for item in select_list:
        num_dict[item["query"]] = []

    for item in select_list:
        num_dict[item["query"]].append(item)

    for key in num_dict:
        training_examples = []
        for _ in range(epoch):
            training_examples.extend(num_dict[key])
        training_examples = training_examples[: pruned_numbers * epoch]
        pruned_metamath.extend(training_examples)

    random.shuffle(pruned_metamath)

    print("After pruning", len(pruned_metamath))

    merged_dataset = mmiqc_stex_dataset + pruned_metamath
    random.shuffle(merged_dataset)
    print("Merged: ", len(merged_dataset))

    with open(output_path, "w") as f:
        json.dump(select_list, f, indent=2)

    with open(pruned_output_path, "w") as f:
        json.dump(merged_dataset, f, indent=2)

    print("Saving to {}".format(output_path))
    print("Saving pruned to {}".format(pruned_output_path))

    if print_examples:
        cnt = 10
        for ex in merged_dataset:
            if "stackexchange-math" in ex["type"]:
                if "\\begin" in ex["output"] or "$$" in ex["output"]:
                    print(ex["query"])
                    print("-" * 40)
                    print(ex["output"])
                    print("=" * 80)
                    cnt -= 1
                    if cnt == 0:
                        break


if __name__ == "__main__":
    fire.Fire(main)
