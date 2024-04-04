import json
from typing import List

import fire

from datasets import load_dataset

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llemma_34b")


def main(
    levels: str = "Level 1, Level 2, Level 3",
    output_path: str = "/nobackup/users/yikangs/zhiqings/math/train_1to5_1-2-3_prm_ppo.json",
    skip_unavailable: bool = False,
    question_format: str = "# Question\n\n{question}\n\n# Solution\n\n",
    train_math_path: str = "/nobackup/users/yikangs/zhiqings/math/prm800k/math_splits/train.jsonl",
    test_math_path: str = "/nobackup/users/yikangs/zhiqings/math/prm800k/math_splits/test.jsonl",
):
    # math_dataset = load_dataset("hendrycks/competition_math")
    levels = levels.split(", ")
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

    print("Train:", len(math_dataset["train"]))
    print("Test:", len(math_dataset["test"]))

    outputs = []

    for data in math_dataset["train"]:
        gt_answer = data["answer"]
        subject = data["subject"]
        level = data["level"]

        if f"Level {level}" in levels:
            answer = gt_answer
        else:
            answer = "Unavailable"

        if skip_unavailable and answer == "Unavailable":
            continue

        outputs.append(
            {
                "input": question_format.format(question=data["problem"]),
                "answer": answer,
                "gt_answer": gt_answer,
                "subject": subject,
                "level": level,
            }
        )

    print("Train:", len(outputs))

    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
