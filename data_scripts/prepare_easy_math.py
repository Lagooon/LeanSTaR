import glob
import json
import os
import random
import re

import fire


INSTRUCTION = (
    "I will give you a mathematical problem, either pure math or a word problem. "
    'Your response should always start with the phrase "Let\'s think step by step." '
    "Following this, you should provide a clear and logical breakdown of the problem, detailing each step and any calculations or reasonings involved. "
    'Conclude your response with the phrase "The answer is: [ANSWER]", where "[ANSWER]" is the final solution to the problem. '
    "Any mathematical symbols, equations, or expressions should be accurately represented and clear to understand."
)


def process_math(
    metamath_math_data,
    seed,
    train_portion,
    keep_levels=None,
):
    if keep_levels is None:
        keep_levels = ["Level 1", "Level 2", "Level 3"]
    data = []

    with open(metamath_math_data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    question_map = {}

    question_id = 0
    for d in data:
        question = d["question"]

        # if not d['true_mask']:
        #     continue
        if d["level"] not in keep_levels:
            continue

        if question not in question_map:
            question_map[question] = (question_id, 1)
            question_id += 1
        else:
            question_map[question] = (
                question_map[question][0],
                question_map[question][1] + 1,
            )

    # count question frequency

    question_freq = {}

    for question_id, freq in question_map.values():
        if freq not in question_freq:
            question_freq[freq] = 1
        else:
            question_freq[freq] += 1

    print("question_freq:", question_freq)
    print("all freq:", sum(question_freq.values()))

    # split sft and rm
    random.seed(seed)

    all_question_ids = list(range(len(question_map)))
    random.shuffle(all_question_ids)

    sft_question_ids = all_question_ids[: int(len(all_question_ids) * train_portion)]
    rm_question_ids = all_question_ids[int(len(all_question_ids) * train_portion) :]

    # for sft, we only keep the true_mask == True data
    # for rm, we keep the first question and remove the rest

    sft_data = []
    rm_data = []

    rm_question_set = set()

    example_id = 0
    for d in data:
        if d["level"] not in keep_levels:
            continue

        question = d["question"]
        question_id, freq = question_map[question]

        if question_id in sft_question_ids:
            if d["true_mask"]:
                sft_data.append(
                    {
                        "instruction": INSTRUCTION,
                        "input": question,
                        "output": d["pred_answer"],
                        "short_answer": d["pred_answer_cleaned"],
                        "data_type": d["data_type"],
                        "type": d["type"],
                        "level": d["level"],
                        "example_id": example_id,
                    }
                )
                example_id += 1
        else:
            if question_id not in rm_question_set:
                rm_question_set.add(question_id)
                rm_data.append(
                    {
                        "instruction": INSTRUCTION,
                        "input": question,
                        "output": d["pred_answer"],
                        "short_answer": d["pred_answer_cleaned"],
                        "data_type": d["data_type"],
                        "type": d["type"],
                        "level": d["level"],
                        "example_id": example_id,
                    }
                )
                example_id += 1

    print("MATH SFT:", len(sft_data))
    print("MATH RM:", len(rm_data))
    return sft_data, rm_data


def process_gsm8k(
    metamath_gsm8k_data,
    seed,
    train_portion,
):
    data = []

    with open(metamath_gsm8k_data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    question_map = {}

    question_id = 0
    for d in data:
        question = d["question"]

        # if not d['true_mask']:
        #     continue
        if question not in question_map:
            question_map[question] = (question_id, 1)
            question_id += 1
        else:
            question_map[question] = (
                question_map[question][0],
                question_map[question][1] + 1,
            )

    # count question frequency

    question_freq = {}

    for question_id, freq in question_map.values():
        if freq not in question_freq:
            question_freq[freq] = 1
        else:
            question_freq[freq] += 1

    print("question_freq:", question_freq)
    print("all freq:", sum(question_freq.values()))

    # split sft and rm

    random.seed(seed)

    all_question_ids = list(range(len(question_map)))
    random.shuffle(all_question_ids)

    sft_question_ids = all_question_ids[: int(len(all_question_ids) * train_portion)]
    rm_question_ids = all_question_ids[int(len(all_question_ids) * train_portion) :]

    # for sft, we only keep the true_mask == True data
    # for rm, we keep the first question and remove the rest

    sft_data = []
    rm_data = []

    rm_question_set = set()

    example_id = 0
    for d in data:
        question = d["question"]
        question_id, freq = question_map[question]

        if question_id in sft_question_ids:
            if d["true_mask"]:
                sft_data.append(
                    {
                        "instruction": INSTRUCTION,
                        "input": question,
                        "output": d["pred_answer"],
                        "short_answer": d["pred_answer_cleaned"],
                        "data_type": d["data_type"],
                        "example_id": example_id,
                    }
                )
                example_id += 1
        else:
            if question_id not in rm_question_set:
                rm_question_set.add(question_id)
                rm_data.append(
                    {
                        "instruction": INSTRUCTION,
                        "input": question,
                        "output": d["pred_answer"],
                        "short_answer": d["pred_answer_cleaned"],
                        "data_type": d["data_type"],
                        "example_id": example_id,
                    }
                )
                example_id += 1

    print("GSM8K SFT:", len(sft_data))
    print("GSM8K RM:", len(rm_data))
    return sft_data, rm_data


def main(
    metamath_math_data,  # jsonl
    metamath_gsm8k_data,  # jsonl
    output_dir,
    seed=42,
    train_portion=0.8,
    sft_name="train_sft.json",
    reward_name="train_rm.json",
):
    # process metamath math data
    math_sft, math_rm = process_math(metamath_math_data, seed, train_portion)

    # process metamath gsm8k data
    gsm8k_sft, gsm8k_rm = process_gsm8k(metamath_gsm8k_data, seed, train_portion)

    # combine math and gsm8k
    sft_data = math_sft + gsm8k_sft
    rm_data = math_rm + gsm8k_rm

    # shuffle
    random.seed(seed)
    random.shuffle(sft_data)
    random.shuffle(rm_data)

    # write to file

    with open(os.path.join(output_dir, sft_name), "w") as f:
        f.write(json.dumps(sft_data, indent=2))

    with open(os.path.join(output_dir, reward_name), "w") as f:
        f.write(json.dumps(rm_data, indent=2))

    print("SFT path:", os.path.join(output_dir, sft_name))
    print("RM path:", os.path.join(output_dir, reward_name))


if __name__ == "__main__":
    fire.Fire(main)
