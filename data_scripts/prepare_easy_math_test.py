from typing import List
import json
import re

import fire

from datasets import load_dataset

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llemma_34b")

METAMATH_QUERY_PREFIX = 'Please solve the following problem and put your answer at the end with "# Answer\\n\\n{answer}".\n\n'


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def main(
    levels: List[str] = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"],
    output_path: str = "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3_mmiqc.json",
    test_math_path: str = "/nobackup/users/yikangs/zhiqings/math/prm800k/math_splits/test.jsonl",
    max_len: int = 1024,
):
    test_dataset = []

    with open(test_math_path) as f:
        for line in f:
            ex = json.loads(line)
            if f'Level {ex["level"]}' in levels:
                test_dataset.append(ex)

    output = []

    for ex in test_dataset:
        answer = ex["answer"]
        # answer = last_boxed_only_string(ex["solution"])
        # transform \boxed{xx} to xx
        # if answer is not None:
        #     answer = answer[7:-1]

        # if answer is None:
        #     print(ex["solution"])
        #     answer = "ParseError"
        #     # raise ValueError("No solution found")

        output.append(
            {
                "prompt": "# Question\n\n"
                + METAMATH_QUERY_PREFIX
                + ex["problem"]
                + "\n\n# Solution\n\n",
                # "prompt": (
                #     "# Question\n\n"
                #     + ex["problem"]
                #     + " Please answer the question in the MetaMath format."
                #     + "\n\n# Solution\n\n"
                # ),
                "level": f'Level {ex["level"]}',
                "type": ex["subject"],
                "gt_solution": ex["solution"],
                "gt_answer": answer,
            }
        )

    # print the longest length of the prompt
    new_output = []
    for ex in output:
        if len(tokenizer(ex["prompt"])["input_ids"]) <= max_len:
            new_output.append(ex)
    output = new_output

    print(f"Max prompt length: {max_len}")

    print(f"Number of test problems: {len(output)}")

    print(output[0])

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    fire.Fire(main)
