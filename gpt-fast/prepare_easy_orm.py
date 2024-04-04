import glob
import json
import os
import random
import re
import tqdm
from typing import List

import fire

from grading import grader


def main(
    levels: List[str] = ["Level 1", "Level 2", "Level 3"],
    prompt_path: str = "/nobackup/users/yikangs/zhiqings/math/train_1to5_prm_v3.json",
    sample_path: str = "/nobackup/users/yikangs/zhiqings/math/outputs/train_1to5_prm_v3_7b-v4_epoch-3_lr-2e-5_seq-768_s32.jsonl",
    output_path: str = "/nobackup/users/yikangs/zhiqings/math/train_1_2_3_prm_v5_orm.json",
    max_ratio: float = 3.0,
    limiter: int = 16,
):
    with open(prompt_path) as f:
        prompts = json.load(f)

    with open(sample_path) as f:
        samples = [json.loads(l) for l in f]

    idx_set = set()
    id2sample = {}

    print("Number of prompts:", len(prompts))
    print("Number of samples:", len(samples))

    for sample in tqdm.tqdm(samples):
        idx = sample["idx"]
        sample_idx = sample["sample_idx"]

        if (idx, sample_idx) in idx_set:
            continue

        if prompts[idx]["level"] not in levels:
            continue

        idx_set.add((idx, sample_idx))

        assert sample["prompt"] == prompts[idx]["prompt"]

        input_text = sample["prompt"][
            len("# Question\n\n") : -len("\n\n# Solution\n\n")
        ]

        output_text = sample["output"]

        if len(output_text.split("\n\n# Answer\n\n")) != 2:
            continue

        pred_answer = output_text.split("\n\n# Answer\n\n")[1]

        gt_answer = prompts[idx]["gt_answer"]

        label = grader.grade_answer(pred_answer, gt_answer)

        if idx not in id2sample:
            id2sample[idx] = []

        id2sample[idx].append(
            {
                "input": input_text,
                "output_prefix": "",
                "output": output_text,
                "label": 1 if label else 0,
            }
        )

    # We do some statistics
    # number of score = 1 and score = 0

    num_score_1 = 0
    num_score_0 = 0
    for idx, samples in id2sample.items():
        for sample in samples:
            if sample["label"] == 1:
                num_score_1 += 1
            else:
                num_score_0 += 1
    print("Number of score = 1:", num_score_1)
    print("Number of score = 0:", num_score_0)

    outputs = []

    output_total_1 = 0
    output_total_0 = 0

    for idx, samples in tqdm.tqdm(id2sample.items()):
        random.shuffle(samples)

        # We take at most limiter samples
        # The ratio between score = 1 and score = 0 is at most max_ratio and at least 1/max_ratio

        num_score_1 = 0
        num_score_0 = 0
        for sample in samples:
            if sample["label"] == 1:
                num_score_1 += 1
            else:
                num_score_0 += 1

        num_score_1 = min(num_score_1, int(limiter * max_ratio))
        num_score_0 = min(num_score_0, int(limiter * max_ratio))

        while num_score_1 + num_score_0 > limiter:
            if num_score_1 > num_score_0:
                num_score_1 -= 1
            else:
                num_score_0 -= 1

        for sample in samples:
            if sample["label"] == 1 and num_score_1 > 0:
                outputs.append(sample)
                num_score_1 -= 1
                output_total_1 += 1
            elif sample["label"] == 0 and num_score_0 > 0:
                outputs.append(sample)
                num_score_0 -= 1
                output_total_0 += 1

    random.shuffle(outputs)

    with open(output_path, "w") as f:
        f.write(json.dumps(outputs, indent=2))

    print("Number of samples:", len(outputs))
    print("Number of label = 1:", output_total_1)
    print("Number of label = 0:", output_total_0)
    print("Output saved to", output_path)


if __name__ == "__main__":
    fire.Fire(main)
