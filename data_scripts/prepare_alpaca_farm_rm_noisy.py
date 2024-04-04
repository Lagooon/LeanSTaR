import json
from typing import List

import random
import tqdm
import fire

from datasets import load_dataset

import numpy as np
from transformers import AutoTokenizer

import llm_blender

blender = llm_blender.Blender()
blender.loadranker("llm-blender/PairRM")

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")


def main(
    output_path: str = "/nobackup/users/yikangs/zhiqings/dpo/alpaca_rm_10k_v2.json",
    # output_path: str = "/nobackup/users/yikangs/zhiqings/dpo/alpaca_rm_10k.json",
    # output_path: str = "/nobackup/users/yikangs/zhiqings/dpo/alpaca_pm_10k.json",
    # preference_model_augmentation: bool = True,
    gt_data_augmentation: bool = True,
):
    dataset = load_dataset(
        "tatsu-lab/alpaca_farm",
        "alpaca_noisy_multi_preference",
        split="preference",
    )

    outputs = []

    for example in tqdm.tqdm(dataset):
        instruction = example["instruction"]
        input_ = example["input"]
        output_1 = example["output_1"]
        output_2 = example["output_2"]
        preference = example["preference"]

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

    all_inputs = [
        (
            ex["instruction"] + "\n\n" + ex["input"]
            if ex["input"] != ""
            else ex["instruction"]
        )
        for ex in outputs
    ]
    all_candidate_texts = [[ex["output_1"], ex["output_2"]] for ex in outputs]

    all_scores = blender.rank(
        all_inputs,
        all_candidate_texts,
        return_scores=True,
        batch_size=32,
        policy="max_probs",
    )

    print(np.min(all_scores), np.max(all_scores))

    all_scores = (all_scores * 2.0 - 0.5).tolist()

    cnt_correct = 0.0
    cnt_error = 0.0

    for i, ex in enumerate(outputs):
        ex["win_rate_1"] = all_scores[i][0]
        ex["win_rate_2"] = all_scores[i][1]

        if ex["preference"] == 1 and ex["win_rate_1"] > ex["win_rate_2"]:
            cnt_correct += 1
        elif ex["preference"] == 2 and ex["win_rate_2"] > ex["win_rate_1"]:
            cnt_correct += 1
        else:
            cnt_error += 1

        if gt_data_augmentation:
            gt_win_rate_1 = 1.0 if ex["preference"] == 1 else 0.0
            gt_win_rate_2 = 1.0 if ex["preference"] == 2 else 0.0
            ex["win_rate_1"] = (ex["win_rate_1"] + gt_win_rate_1) / 2.0
            ex["win_rate_2"] = (ex["win_rate_2"] + gt_win_rate_2) / 2.0

    print(f"Preference model accuracy: {cnt_correct / (cnt_correct + cnt_error)}")

    random.shuffle(outputs)

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
