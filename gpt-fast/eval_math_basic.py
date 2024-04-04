import glob
import json
from collections import Counter

from grading import grader

import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


#  prm.json
#   {
#     "prompt": "Question: ...",
#     "level": "Level ....",
#     "type": "....",
#     "gt_solution": "....",
#     "gt_answer": "...."
#   },


def print_table(results_table, type_list, level_list):
    # Header for levels
    print(f"{'Type/Level':<24}", end="")
    for level in level_list:
        print(f"{level:>16}", end="")
    print(f"{'Overall':>16}")

    # Rows for types
    for type_name in type_list:
        print(f"{type_name:<24}", end="")
        for level_name in level_list:
            correct, total = results_table[type_name].get(level_name, [0, 0])
            accuracy = correct / total if total > 0 else 0
            print(f"{accuracy:>16.3f}", end="")
        # Overall accuracy for type
        correct, total = results_table[type_name].get("Overall", [0, 0])
        accuracy = correct / total if total > 0 else 0
        print(f"{accuracy:>16.3f}")
    print()

    # Overall row at the bottom
    print(f"{'Overall':<24}", end="")
    acc_list = []
    # num_list = [437, 894, 1131, 1214, 1324]
    num_list = [43, 90, 105, 128, 134]
    for level_name, num_cnt in zip(level_list, num_list):
        correct, total = results_table["Overall"].get(level_name, [0, 0])
        # assert total == num_cnt
        if total != num_cnt:
            print(f"Warning: total {total} != num_cnt {num_cnt}")
        accuracy = correct / total if total > 0 else 0
        acc_list.append(accuracy)
        print(f"{accuracy:>16.3f}", end="")
    # Overall accuracy for all types and levels
    correct, total = results_table["Overall"]["Overall"]
    accuracy = correct / total if total > 0 else 0
    print(f"{accuracy:>16.3f}")
    print(
        "Level1-3:",
        round(
            (
                acc_list[0] * num_list[0]
                + acc_list[1] * num_list[1]
                + acc_list[2] * num_list[2]
            )
            / (num_list[0] + num_list[1] + num_list[2]),
            3,
        ),
        "Level4-5:",
        round(
            (acc_list[3] * num_list[3] + acc_list[4] * num_list[4])
            / (num_list[3] + num_list[4]),
            3,
        ),
        "Level1-5:",
        round(
            (
                acc_list[0] * num_list[0]
                + acc_list[1] * num_list[1]
                + acc_list[2] * num_list[2]
                + acc_list[3] * num_list[3]
                + acc_list[4] * num_list[4]
            )
            / (num_list[0] + num_list[1] + num_list[2] + num_list[3] + num_list[4]),
            3,
        ),
    )


if __name__ == "__main__":
    gt_files = [
        "/nobackup/users/yikangs/zhiqings/math/valid_v3_1to5_1-2-3_prm_ppo.json",
        # "/nobackup/users/yikangs/zhiqings/math/test_1_2_3_prm.json",
        # "/nobackup/users/yikangs/zhiqings/math/test_1_2_3_prm_v3.json",
        "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json",
        # "/workspace/zhiqings/output4/math/test_1to5_prm_v3.json",
        # "/workspace/zhiqings/output4/math/test_1to5_prm_v3.json",
    ]
    answer_patterns = [
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1_2_3_prm_7b*.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1_2_3_prm_self_correct*_7b_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1_2_3_prm_v3*.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b*.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_34b*.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-metamath_*g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_34b-v4_epoch-3_lr-2e-5_seq-768_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-v4_epoch-3_lr-2e-5_seq-768_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-v4_epoch-3_lr-2e-5_seq-768_g2.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_ppo_prm_v3_7b_epoch-50_lr-2e-5_seq-768_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_ppo_prm_v3_7b-v2_epoch-50_lr-2e-5_seq-768_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_ppo_prm_v3_7b-1to3-v4_epoch-50_lr-2e-5_seq-768_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_ppo_prm_v3_7b-4to5-v3_epoch-50_lr-2e-5_seq-768_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_ppo_prm_v3_7b-1to3-v8_epoch-50_lr-2e-5_seq-768_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_ppo_prm_v3_7b-1to4-v9_epoch-50_lr-2e-5_seq-768_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_ppo_prm_v3_7b-1to3-vc_epoch-50_lr-2e-5_seq-768_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_ppo_prm_v3_7b-1to3-vj_epoch-50_lr-2e-5_seq-768_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-full-v4_epoch-10_lr-1e-5_seq-1024_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-metamath_1-to-3_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-metamath_1-to-5_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-dpo-prm_epoch-5_round-r1_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-dpo-prm_epoch-3_round-r1_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-dpo-prm_epoch-3_round-r2_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-dpo-prm_epoch-3_round-r3_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-dpo-prm_epoch-3_round-r4_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-dpo-prm_epoch-3_round-r5_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-dpo-prm_epoch-2_round-r1_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-dpo-prm_epoch-2_round-r2_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-dpo-prm_epoch-2_round-r3_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-dpo-prm_epoch-2_round-r4_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-dpo-prm_epoch-2_round-r5_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to3_prm_v3_7b-dpo-prm_epoch-3_round-r1_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to3_prm_v3_7b-dpo-prm_epoch-3_round-r2_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to3_prm_v3_7b-dpo-prm_epoch-3_round-r3_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to3_prm_v3_7b-dpo-prm_epoch-3_round-r4_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to3_prm_v3_7b-dpo-prm_epoch-3_round-r5_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to3_prm_v3_7b-dpo-prm_epoch-3_round-r6_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to3_metamath_v3_7b-dpo-prm_epoch-3_round-r1_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to3_metamath_v3_7b-dpo-prm_epoch-3_round-r2_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to3_metamath_v3_7b-dpo-prm_epoch-3_round-r3_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to3_metamath_v3_7b-dpo-prm_epoch-3_round-r4_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to3_metamath_v3_7b-dpo-prm_epoch-3_round-r5_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to3_metamath_v3_7b-dpo-prm_epoch-3_round-r6_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to3_metamath_v3_7b-dpo-prm_epoch-3_round-r7_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_metamath_v3_7b-dpo-prm_epoch-3_round-r1_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_metamath_v3_7b-dpo-prm_epoch-3_round-r2_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_metamath_v3_7b-dpo-prm_epoch-3_round-r3_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to3_metamath_v3_7b-dpo-prm_epoch-4_t-1.2_round-r1_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to3_metamath_v3_7b-dpo-prm_epoch-4_t-1.2_round-r2_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-dpo-prm_epoch-2_round-r1_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-dpo-prm_epoch-2_round-r2_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-dpo-prm_epoch-2_round-r3_g.jsonl",
        # "/workspace/zhiqings/output4/math/outputs/test_1to5_prm_v3_7b-metamath_ppo-v9_g.jsonl",
        # "/workspace/zhiqings/output4/math/outputs/test_1to5_prm_v3_7b-metamath_ppo-vb_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-metamath_v5_1-to-5_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-metamath_v5_amp_1-to-5_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-metamath_v5_1-to-5_g.jsonl",p
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_ds-math-7b_metamath_v5_1-to-5_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-mmiqc_1-to-3_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_ds-math-7b_prm_v4_1-to-3_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_ds-math-7b-amp_prm_v4_1-to-3_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_ds-math-7b_mmiqc_v5_1-to-3_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_ds-math-7b_mmiqc_v6_1-to-5_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_ds-math-7b_metamath_v5_1-to-3_g.jsonl",
        #
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_34b-metamath_v6_amp_1-to-3_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_34b-metamath-5e-6_v6_amp_1-to-3_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_34b-metamath-8e-6_v6_amp_1-to-5_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_34b-metamath-5e-6_v6_amp_1-to-5_g.jsonl",
        #
        "/nobackup/users/yikangs/zhiqings/math/outputs/valid_1to5_dpo-metamath_*seed-4*_1-to-3_dup-3_lr-*e-6_beta-0.1_r*_s8_t*_g.jsonl",
        "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-metamath_v6_amp_1-to-3_g.jsonl",
        "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-metamath_v6_amp_1-to-5_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_dpo-metamath_*seed-4*_1-to-3_dup-3_lr-*e-6_beta-0.1_r*_s8_t*_g.jsonl",
        # "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_dpo-metamath_*seed-*_1-to-3_dup-3_lr-*e-6_beta-0.1_r*_s8_t*_g.jsonl",
    ]

    while len(gt_files) < len(answer_patterns):
        gt_files.append(gt_files[-1])
    assert len(gt_files) == len(answer_patterns)

    for gt_file, answer_pattern in zip(gt_files, answer_patterns):
        with open(gt_file, "r") as f:
            gt_prompts = json.load(f)
        print("Reading prompts from", answer_pattern)

        prompt_idx = list(range(len(gt_prompts)))

        idx_to_prompt = {idx: prompt for idx, prompt in zip(prompt_idx, gt_prompts)}

        for answer_file in sorted(glob.glob(answer_pattern)):
            weighted_voting = False
            idx_to_answer = {}
            idx_to_sample_idx = {}
            results_table = {}

            type_list = set()
            level_list = set()

            with open(answer_file, "r") as f:
                for line in f:
                    answer = json.loads(line)
                    if answer["idx"] not in idx_to_answer:
                        idx_to_answer[answer["idx"]] = []
                        idx_to_sample_idx[answer["idx"]] = set()

                    if "# Answer\n\n" in answer["output"]:
                        concise_answer = (
                            answer["output"]
                            .rsplit("# Answer\n\n")[1]
                            .split("\n")[0]
                            .strip()
                        )

                        sample_idx = answer["sample_idx"]
                        if sample_idx not in idx_to_sample_idx[answer["idx"]]:
                            idx_to_sample_idx[answer["idx"]].add(sample_idx)

                            if "reward" in answer:
                                weighted_voting = True

                                reward = [sigmoid(r) for r in answer["reward"]]
                                # final reward is the minimun of all rewards
                                final_reward = 1
                                for r in reward:
                                    final_reward *= r

                                idx_to_answer[answer["idx"]].append(
                                    (concise_answer, final_reward)
                                )
                            else:
                                idx_to_answer[answer["idx"]].append(concise_answer)

            # do majority voting or weighted voting
            correct_count = 0
            total_count = 0

            for idx in idx_to_answer:
                answers = idx_to_answer[idx]

                if len(answers) == 0:
                    model_short_answer = "I don't know"
                else:
                    if weighted_voting:
                        # weighted voting
                        # answers = [(answer, reward), ...]
                        # reward is a float between 0 and 1
                        # we get the answer with the highest sum of rewards
                        answer_reward = {}
                        for answer, reward in answers:
                            if answer not in answer_reward:
                                answer_reward[answer] = 0.0
                            answer_reward[answer] += reward
                            # answer_reward[answer] = max(reward, answer_reward[answer])

                        model_short_answer = max(answer_reward, key=answer_reward.get)
                    else:
                        counter = Counter(answers)
                        model_short_answer = counter.most_common(1)[0][0]

                if "type" in idx_to_prompt[idx]:
                    question_type = idx_to_prompt[idx]["type"]
                elif "subject" in idx_to_prompt[idx]:
                    question_type = idx_to_prompt[idx]["subject"]
                else:
                    raise ValueError("No type or subject in prompt")
                level = str(idx_to_prompt[idx]["level"])

                type_list.add(question_type)
                level_list.add(level)

                # Initialize dictionary entries if they do not exist
                if question_type not in results_table:
                    results_table[question_type] = {}
                if level not in results_table[question_type]:
                    results_table[question_type][level] = [0, 0]

                correct_answer = grader.grade_answer(
                    model_short_answer, idx_to_prompt[idx]["gt_answer"]
                )

                # if model_short_answer == gt_answer:
                if correct_answer:
                    results_table[question_type][level][0] += 1
                    if "Overall" not in results_table[question_type]:
                        results_table[question_type]["Overall"] = [0, 0]
                    results_table[question_type]["Overall"][0] += 1
                results_table[question_type][level][1] += 1
                if "Overall" not in results_table[question_type]:
                    results_table[question_type]["Overall"] = [0, 0]
                results_table[question_type]["Overall"][1] += 1

                # Overall accuracy
                if "Overall" not in results_table:
                    results_table["Overall"] = {}
                if level not in results_table["Overall"]:
                    results_table["Overall"][level] = [0, 0]
                results_table["Overall"][level][1] += 1
                if correct_answer:
                    results_table["Overall"][level][0] += 1
                if "Overall" not in results_table["Overall"]:
                    results_table["Overall"]["Overall"] = [0, 0]
                results_table["Overall"]["Overall"][1] += 1
                if correct_answer:
                    results_table["Overall"]["Overall"][0] += 1

            print(answer_file)
            # Sort the lists
            type_list = sorted(type_list)
            level_list = sorted(level_list, key=lambda x: int(x[-1]))

            # Print the table
            print_table(results_table, type_list, level_list)
            print()
