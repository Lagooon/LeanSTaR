import os
import random
import json

if __name__ == "__main__":
    gt_path = "data/math_rm/question.jsonl"
    model_answer_dir = "data/math_rm/model_answer"
    model_label_dir = "data/math_rm/model_label"

    max_replication = 12

    with open(gt_path, "r") as f:
        gt = [json.loads(line) for line in f]

    for file in os.listdir(model_answer_dir):
        model_answer_path = os.path.join(model_answer_dir, file)
        with open(model_answer_path, "r") as f:
            model_answer = [json.loads(line) for line in f]

        model_label_path = os.path.join(
            model_label_dir, file.replace(".jsonl", f"-replic{max_replication}.json")
        )

        zeros = 0
        ones = 0
        model_labels = []

        # sort model_answer with question_id
        model_answer = sorted(model_answer, key=lambda x: x["question_id"])

        assert len(gt) == len(model_answer)

        for i in range(len(gt)):
            gt_answer = gt[i]["short_answer"]
            if "Level" in gt[i]["level"]:
                level = f"Level {gt[i]['level'][-1]}"
            else:
                level = f"Level 0"
            question_type = gt[i]["type"]
            if "turns" in model_answer[i]["choices"][0]:
                keyword = "turns"
            else:
                keyword = "output"

            answers = []

            for answer in model_answer[i]["choices"]:
                answers.append(answer[keyword][0])

            max_ones = max_replication
            max_zeros = max_replication
            max_one_neg = 1
            max_one_pos = 1
            # do best of n matching
            for answer in answers:
                short_answer = answer.split("\nThe answer is:")[-1].strip()
                if short_answer == gt_answer:
                    model_label = 1
                else:
                    model_label = 0

                if model_label == 1 and max_zeros <= 0:
                    if max_one_neg > 0:
                        max_one_neg -= 1
                        model_label = 0
                    else:
                        continue
                if model_label == 0 and max_ones <= 0:
                    if max_one_pos > 0:
                        max_one_pos -= 1
                        model_label = 1
                    else:
                        continue

                if model_label == 1:
                    ones += 1
                    max_zeros -= 1
                else:
                    zeros += 1
                    max_ones -= 1

                model_labels.append(
                    {
                        "question_id": gt[i]["question_id"],
                        "instruction": gt[i]["turns"][0],
                        "input": "",
                        "output": answer,
                        "level": level,
                        "type": question_type,
                        "label": model_label,
                    }
                )

        random.shuffle(model_labels)

        with open(model_label_path, "w") as f:
            json.dump(model_labels, f, indent=2)

        print(f"zeros: {zeros}, ones: {ones}")
        print(f"Finish {model_label_path}")
