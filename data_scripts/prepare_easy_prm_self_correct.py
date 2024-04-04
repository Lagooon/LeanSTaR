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
    output_path: str = "/nobackup/users/yikangs/zhiqings/math/train_1to3_prm_self_correct_v5_epoch_3_v3_max_2.json",
    debug_path: str = "/nobackup/users/yikangs/zhiqings/math/debug.json",
    prm_data_pattern: str = "/nobackup/users/yikangs/zhiqings/math/prm800k/data/*train.jsonl",
    train_math_path: str = (
        "/nobackup/users/yikangs/zhiqings/math/prm800k/math_splits/train.jsonl"
    ),
    test_math_path: str = (
        "/nobackup/users/yikangs/zhiqings/math/prm800k/math_splits/test.jsonl"
    ),
    splitter: str = "\n\n",
    epoch: int = 3,
    max_error_per_step: int = 2,
    max_inserted_errors: int = 5,
):
    # math_dataset = load_dataset("hendrycks/competition_math")
    math_dataset = {}

    math_dataset["train"] = []
    math_dataset["test"] = []

    with open(train_math_path) as f:
        for line in f:
            data = json.loads(line)
            math_dataset["train"].append(data)

    with open(test_math_path) as f:
        for line in f:
            data = json.loads(line)
            math_dataset["test"].append(data)

    all_promblems = set()
    train_promblems = set()
    asy_promblems = set()

    for split in ["train", "test"]:
        for ex in math_dataset[split]:
            problem = ex["problem"]
            all_promblems.add(problem)
            if "[asy]" in problem:
                asy_promblems.add(problem)

    for ex in math_dataset["train"]:
        if f'Level {ex["level"]}' in levels:
            problem = ex["problem"]
            train_promblems.add(problem)

    print(f"Number of MATH-easy train problems: {len(train_promblems)}")
    print(f"Number of [asy] problems: {len(asy_promblems)}")

    additional_promblems = set()

    prm_data_files = glob.glob(prm_data_pattern)

    outputs = []
    skipped = 0

    prm_train_promblems = set()

    print("Training from:", prm_data_files)

    for data_file in prm_data_files:
        with open(data_file) as f:
            for line in f:
                data = json.loads(line)

                if data["question"]["problem"] in train_promblems:
                    outputs.append(data)
                    prm_train_promblems.add(data["question"]["problem"])
                elif data["question"]["problem"] not in all_promblems:
                    outputs.append(data)
                    prm_train_promblems.add(data["question"]["problem"])
                else:
                    skipped += 1

                # # What is the smallest positive integer such that the product of its digits is $2016$?
                if data["question"]["problem"] not in all_promblems:
                    additional_promblems.add(data["question"]["problem"])

    print(f"Number of PRM train problems: {len(prm_train_promblems)}")
    print(f"Number of additional problems in PRM800k: {len(additional_promblems)}")

    # for ex in train_promblems:
    #     if ex not in prm_train_promblems:
    #         print(ex)
    #         print("=" * 20)

    print(f"Skipped {skipped} problems")
    print(f"Number of PRM annotations: {len(outputs)}")

    train_errors = 0
    inserted_errors = 0
    # SELF_CORRECT_PROMPT = splitter + "Oops! I made a mistake. Correction: "
    # splitter = "(good)\n\n"
    # SELF_CORRECT_PROMPT = "(bad)\n\nThis seems to be wrong. Let's try again: "
    SELF_CORRECT_PROMPT = splitter + "This seems to be wrong. Let's try again: "

    extended_output = []

    for epoch_iter in range(epoch):
        generated_answer_set = set()

        pre_generated_cnt = 0

        num_errors_dict = {}

        for ex in outputs:
            error_flag = False

            question = ex["question"]["problem"]

            chosen_prefix = ""

            for step_idx, step in enumerate(ex["label"]["steps"]):
                if step["completions"] is None:
                    print(step)
                    break

                error_step = None
                error_steps = []
                for completion in step["completions"]:
                    if completion["rating"] is not None and completion["rating"] < 0:
                        # The expression is a telescoping series
                        error_steps.append(completion["text"])

                num_errors_dict[len(error_steps)] = (
                    num_errors_dict.get(len(error_steps), 0) + 1
                )

                for completion in step["completions"]:
                    text = completion["text"]

                    rating = completion["rating"]

                    if rating is not None and rating < 0:
                        continue

                    if "# Answer\n\n" in text and "\n# Answer\n\n" not in text:
                        text = text.replace("# Answer\n\n", "\n# Answer\n\n")
                    if "\n# Answer\n\n" in text and "\n\n# Answer\n\n" not in text:
                        text = text.replace("\n# Answer\n\n", "\n\n# Answer\n\n")

                    if chosen_prefix == "":
                        full_text = text
                    else:
                        full_text = chosen_prefix + splitter + text

                    if full_text not in generated_answer_set:
                        is_eos = "# Answer\n\n" in text
                        add_splitter = not text.startswith("\n\n# Answer\n\n")
                        new_splitter = splitter
                        if chosen_prefix.endswith("Mr.") or chosen_prefix.endswith(
                            "Mrs."
                        ):
                            new_splitter = " "
                            add_splitter = True

                        erronous_chosen_prefix = chosen_prefix

                        filtered_error_steps = []
                        for error_step in error_steps:
                            if error_step.split()[0] != text.split()[0]:
                                # to avoid starting with the same word
                                filtered_error_steps.append(error_step)

                        if len(filtered_error_steps) > 0:
                            # sample from 1 to len(filtered_error_steps) to construct the errors
                            num_error_this_step = random.randint(
                                0, min(len(filtered_error_steps), max_error_per_step)
                            )
                            if num_error_this_step > 0:
                                train_errors += 1
                                random.shuffle(filtered_error_steps)
                                aggregated_error_step = SELF_CORRECT_PROMPT.join(
                                    filtered_error_steps[:num_error_this_step]
                                )

                                if chosen_prefix == "":
                                    erronous_chosen_prefix = aggregated_error_step
                                else:
                                    erronous_chosen_prefix = (
                                        chosen_prefix + splitter + aggregated_error_step
                                    )
                                new_splitter = SELF_CORRECT_PROMPT

                        if rating is not None and (
                            rating > 0 or len(filtered_error_steps) > 0 or error_flag
                        ):
                            extended_output.append(
                                {
                                    "input": question,
                                    "output_prefix": erronous_chosen_prefix,
                                    "output": (
                                        new_splitter + text
                                        if (
                                            erronous_chosen_prefix != ""
                                            and add_splitter
                                        )
                                        else text
                                    ),
                                    "is_eos": is_eos,
                                }
                            )
                            generated_answer_set.add(full_text)
                            if error_flag:
                                assert SELF_CORRECT_PROMPT in erronous_chosen_prefix
                                inserted_errors += 1

                if step["chosen_completion"] is None:
                    if step["human_completion"] is not None:
                        next_sentence = step["human_completion"]["text"]
                    elif len(step["completions"]) == 1:
                        next_sentence = step["completions"][0]["text"]
                    else:
                        # find the completion with the highest rating
                        max_rating = -1
                        for completion in step["completions"]:
                            if (
                                completion["rating"] is not None
                                and completion["rating"] > max_rating
                            ):
                                next_sentence = completion["text"]
                                max_rating = completion["rating"]

                        # check only one completion has the highest rating
                        num_max_rating = 0
                        for completion in step["completions"]:
                            if completion["rating"] == max_rating:
                                num_max_rating += 1

                        if num_max_rating > 1:
                            # select the first completion with the highest rating
                            for completion in step["completions"]:
                                if completion["rating"] == max_rating:
                                    next_sentence = completion["text"]
                                    break
                else:
                    next_sentence = step["completions"][step["chosen_completion"]][
                        "text"
                    ]

                if len(error_steps) > 0:
                    # sample from 0 to len(error_steps) to construct the errors
                    num_error_this_step = random.randint(
                        1, min(len(error_steps), max_inserted_errors)
                    )
                    random.shuffle(error_steps)
                    if num_error_this_step > 0:
                        error_step = SELF_CORRECT_PROMPT.join(
                            error_steps[:num_error_this_step]
                        )
                        if chosen_prefix == "":
                            chosen_prefix = error_step
                        else:
                            chosen_prefix = chosen_prefix + splitter + error_step

                        chosen_prefix = (
                            chosen_prefix + SELF_CORRECT_PROMPT + next_sentence
                        )
                        error_flag = True
                        continue

                if chosen_prefix == "":
                    chosen_prefix = next_sentence
                else:
                    chosen_prefix = chosen_prefix + splitter + next_sentence

    print(f"Number of pre-generated PRM annotations: {pre_generated_cnt}")
    print(f"Number of extended PRM annotations: {len(extended_output)}")
    print(f"Statistics of number of errors: {num_errors_dict}")
    print(f"Number of inserted errors: {inserted_errors}")
    print(f"Number of train errors: {train_errors}")

    random.shuffle(extended_output)

    # Check max input (input + output_prefix) length and max output length

    # max_input_length = 768
    # max_output_length = 256
    max_total_length = 768

    filtered_output = []

    for ex in tqdm.tqdm(extended_output):
        input_length = len(tokenizer(ex["input"] + ex["output_prefix"])["input_ids"])
        output_length = len(tokenizer(ex["output"])["input_ids"])

        # if input_length <= max_input_length and output_length <= max_output_length:
        #     filtered_output.append(ex)

        if input_length + output_length <= max_total_length:
            filtered_output.append(ex)

    # re-account for the number of errors
    debug_output = []

    train_errors = 0
    inserted_errors = 0
    for ex in extended_output:
        if "Oops! I made a mistake." in ex["output_prefix"]:
            inserted_errors += 1
            debug_output.append(ex)
        if "Oops! I made a mistake." in ex["output"]:
            train_errors += 1
    print(f"Re-counted number of inserted errors: {inserted_errors}")
    print(f"Re-counted number of train errors: {train_errors}")
    print(f"Number of filtered PRM annotations: {len(filtered_output)}")

    with open(output_path, "w") as f:
        json.dump(filtered_output, f, indent=2)

    with open(debug_path, "w") as f:
        json.dump(debug_output, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
