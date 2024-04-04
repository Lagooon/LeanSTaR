import json
import sys
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from grading import grader


if __name__ == "__main__":
    file1 = "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json"

    with open(file1, "r") as f:
        file1_data = json.load(f)

    # {"prompt": str, "gt_answer": str}

    file2 = "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to3_metamath_v3_7b-dpo-prm_epoch-3_round-r5_g.jsonl"

    file2_prompt_to_output = {}
    with open(file2, "r") as f:
        for line in f:
            data = json.loads(line)
            file2_prompt_to_output[data["prompt"]] = data["output"]

    file3 = "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-metamath_1-to-3_g.jsonl"

    file3_prompt_to_output = {}
    with open(file3, "r") as f:
        for line in f:
            data = json.loads(line)
            file3_prompt_to_output[data["prompt"]] = data["output"]

    acc_1 = 0
    acc_2 = 0
    cnt = 0
    for ex in file1_data:
        prompt = ex["prompt"]
        gt_answer = ex["gt_answer"]

        answer2 = file2_prompt_to_output[prompt].split("\n\n# Answer\n\n")[-1]
        answer3 = file3_prompt_to_output[prompt].split("\n\n# Answer\n\n")[-1]

        if grader.grade_answer(answer2, gt_answer):
            acc_1 += 1
        if grader.grade_answer(answer3, gt_answer):
            acc_2 += 1

        if answer2 != answer3:
            if grader.grade_answer(answer2, gt_answer) and not grader.grade_answer(
                answer3, gt_answer
            ):
                print(prompt)
                print("=" * 20)
                print(gt_answer)
                print("=" * 20)
                print(file2_prompt_to_output[prompt])
                print("=" * 20)
                print(file3_prompt_to_output[prompt])
                print("=" * 20)
                cnt += 1

    print(acc_1 / len(file1_data))
    print(acc_2 / len(file1_data))
    print(cnt)
