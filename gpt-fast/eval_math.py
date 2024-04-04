import glob
import json
from collections import Counter
import random
import logging

import math
import argparse

import tqdm
import matplotlib.pyplot as plt
import numpy as np
from grading import grader


# from scipy.interpolate import make_interp_spline
class DisableLogger:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


DATA_DIR = "/nobackup/users/yikangs/zhiqings/math"


def figure_accs(
    name_prefix,
    output_filename,
    majority_accs_path,
    weighted_accs_path,
    best_of_n_accs_path,
):
    with open(majority_accs_path, "r") as f:
        majority_accs = json.load(f)
    with open(weighted_accs_path, "r") as f:
        weighted_accs = json.load(f)
    with open(best_of_n_accs_path, "r") as f:
        best_of_n_accs = json.load(f)
    # plt.rcParams["font.family"] = "Times New Roman"

    majority_accs_file = majority_accs[0]
    majority_accs_file = majority_accs_file.split("/")[-1].replace(".jsonl", "")
    majority_accs = majority_accs[1:]

    weighted_accs_file = weighted_accs[0]
    weighted_accs_file = weighted_accs_file.split("/")[-1].replace(".jsonl", "")
    weighted_accs = weighted_accs[1:]

    best_of_n_accs_file = best_of_n_accs[0]
    best_of_n_accs_file = best_of_n_accs_file.split("/")[-1].replace(".jsonl", "")
    best_of_n_accs = best_of_n_accs[1:]

    set_of_voting_nums = set()
    for item in majority_accs:
        set_of_voting_nums.add(item["voting_num"])
    set_of_voting_nums = sorted(list(set_of_voting_nums))

    for acc_key in ["Level1-3", "Level4-5", "Level1-5"]:
        x = []
        y1_acc = []
        y2_acc = []
        y3_acc = []
        y1_acc_var = []
        y2_acc_var = []
        y3_acc_var = []
        for voting_num in set_of_voting_nums:
            x.append(voting_num)
            for accs, y_acc, y_acc_var in zip(
                [majority_accs, weighted_accs, best_of_n_accs],
                [y1_acc, y2_acc, y3_acc],
                [y1_acc_var, y2_acc_var, y3_acc_var],
            ):
                y = np.array(
                    [item[acc_key] for item in accs if item["voting_num"] == voting_num]
                )
                y_acc.append(y.mean())
                y_acc_var.append(y.std())

        del x[-1]
        y1_acc = y1_acc[: len(x)]
        y2_acc = y2_acc[: len(x)]
        y3_acc = y3_acc[: len(x)]
        y1_acc_var = y1_acc_var[: len(x)]
        y2_acc_var = y2_acc_var[: len(x)]
        y3_acc_var = y3_acc_var[: len(x)]

        x = np.array(x)
        y1_acc = np.array(y1_acc) * 100
        y2_acc = np.array(y2_acc) * 100
        y3_acc = np.array(y3_acc) * 100
        y1_acc_var = np.array(y1_acc_var) * 100  # Convert to percentage
        y2_acc_var = np.array(y2_acc_var) * 100  # Convert to percentage
        y3_acc_var = np.array(y3_acc_var) * 100  # Convert to percentage

        plt.clf()
        fig, ax1 = plt.subplots(figsize=(8, 6))

        # plot the file name
        if acc_key == "Level1-3":
            fig_name = f"{name_prefix}\nAccuracy on Easy (Level 1-3) Problems"
        elif acc_key == "Level4-5":
            fig_name = f"{name_prefix}\nAccuracy on Hard (Level 4-5) Problems"
        else:
            fig_name = f"{name_prefix}\nAccuracy on All (Level 1-5) Problems"

        fontsize = 24
        legend_fontsize = 20
        tick_fontsize = 20

        fig.suptitle(
            fig_name,
            fontsize=fontsize,
            horizontalalignment="center",  # Ensure the title is centered
        )

        for color, label, y_acc, y_acc_var in zip(
            # ["#808080", "#6495ED", "#FFA500"],
            ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e"],
            ["Majority Voting", "Weighted Voting w/ RM", "Best-of-N w/ RM"],
            [y1_acc, y2_acc, y3_acc],
            [y1_acc_var, y2_acc_var, y3_acc_var],
        ):
            # ax1.errorbar(
            #     x,
            #     y_acc,
            #     yerr=y_acc_var,
            #     fmt="-o",
            #     color=color,
            #     ecolor=color,
            #     elinewidth=2,
            #     capsize=5,
            #     label=label,
            # )

            # plot and fill_between
            ax1.plot(x, y_acc, "-o", color=color, label=label, markersize=4)
            ax1.fill_between(
                x,
                y_acc - y_acc_var,
                y_acc + y_acc_var,
                alpha=0.2,
                color=color,
            )

        plt.xscale("log")
        desired_ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        # desired_ticks = [1, 2, 4, 8, 16, 32, 64, 128, 512, 2048]
        desired_ticks = [tick for tick in desired_ticks if tick <= x.max()]

        plt.xticks(
            desired_ticks,
            labels=[str(tick) for tick in desired_ticks],
            fontsize=tick_fontsize,
        )
        plt.yticks(fontsize=fontsize)
        plt.xlabel("N = number of solutions per problem", fontsize=fontsize)
        plt.ylabel("% Problems Solved", fontsize=fontsize)
        plt.legend(fontsize=legend_fontsize)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(
            output_filename.replace(".pdf", f"_{acc_key}.pdf"),
            format="pdf",
            bbox_inches="tight",
        )


def print_table(results_table, type_list, level_list, printing=True):
    # Header for levels
    if printing:
        print(f"{'Type/Level':<24}", end="")
        for level in level_list:
            print(f"{level:>16}", end="")
        print(f"{'Overall':>16}")

    # Rows for types
    for type_name in type_list:
        if printing:
            print(f"{type_name:<24}", end="")
        for level_name in level_list:
            correct, total = results_table[type_name].get(level_name, [0, 0])
            accuracy = correct / total if total > 0 else 0
            if printing:
                print(f"{accuracy:>16.3f}", end="")
        # Overall accuracy for type
        correct, total = results_table[type_name].get("Overall", [0, 0])
        accuracy = correct / total if total > 0 else 0
        if printing:
            print(f"{accuracy:>16.3f}")
    if printing:
        print()

    # Overall row at the bottom
    if printing:
        print(f"{'Overall':<24}", end="")
    acc_list = []
    # num_list = [437, 894, 1131, 1214, 1324]
    num_list = [43, 90, 105, 128, 134]
    for level_name in level_list:
        correct, total = results_table["Overall"].get(level_name, [0, 0])
        accuracy = correct / total if total > 0 else 0
        acc_list.append(accuracy)
        if printing:
            print(f"{accuracy:>16.3f}", end="")
    # Overall accuracy for all types and levels
    correct, total = results_table["Overall"]["Overall"]
    accuracy = correct / total if total > 0 else 0
    if printing:
        print(f"{accuracy:>16.3f}")
    all_acc = {
        "Level1-3": round(
            (
                acc_list[0] * num_list[0]
                + acc_list[1] * num_list[1]
                + acc_list[2] * num_list[2]
            )
            / (num_list[0] + num_list[1] + num_list[2]),
            2,
        ),
        "Level4-5": round(
            (acc_list[3] * num_list[3] + acc_list[4] * num_list[4])
            / (num_list[3] + num_list[4]),
            2,
        ),
        "Level1-5": round(accuracy, 2),
    }

    assert not (
        all_acc["Level4-5"] > all_acc["Level1-5"]
        and all_acc["Level1-3"] > all_acc["Level1-5"]
    ), all_acc
    return all_acc


def main(args):
    if args.mode == "figure":
        figure_accs(
            args.name_prefix,
            args.output_filename,
            args.majority_accs,
            args.weighted_accs,
            args.best_of_n_accs,
        )
        return

    saving_map = {}
    ## need assign voting_nums
    file_cnt = 0
    idx_answer_to_score = {}

    gt_file = args.gt_file
    answer_pattern = args.answer_pattern
    assert gt_file is not None
    assert answer_pattern is not None

    with open(gt_file, "r") as f:
        gt_prompts = json.load(f)

    prompt_idx = list(range(len(gt_prompts)))
    idx_to_prompt = {idx: prompt for idx, prompt in zip(prompt_idx, gt_prompts)}

    for answer_file in sorted(glob.glob(answer_pattern)):
        idx_to_answer = {}
        idx_to_sample_idx = {}
        n_samples = 0

        with open(answer_file, "r") as f:
            lines = f.readlines()
            for line in tqdm.tqdm(lines):
                answer = json.loads(line)
                if answer["idx"] not in idx_to_answer:
                    idx_to_answer[answer["idx"]] = []
                    idx_to_sample_idx[answer["idx"]] = set()

                sample_idx = answer["sample_idx"]
                if sample_idx not in idx_to_sample_idx[answer["idx"]]:
                    idx_to_sample_idx[answer["idx"]].add(sample_idx)
                    filled_flag = False
                    if "# Answer\n\n" in answer["output"]:
                        concise_answer = (
                            answer["output"].rsplit("# Answer\n\n")[1].strip()
                        )

                        sample_idx = answer["sample_idx"]
                        if len(concise_answer) > 0:
                            filled_flag = True
                            with DisableLogger():
                                if (
                                    answer["idx"],
                                    concise_answer,
                                ) not in idx_answer_to_score:
                                    gt_score = grader.grade_answer(
                                        concise_answer,
                                        idx_to_prompt[answer["idx"]]["gt_answer"],
                                    )
                                    idx_answer_to_score[
                                        (answer["idx"], concise_answer)
                                    ] = gt_score

                            if "reward" in answer:
                                assert args.decoding_mode in ["best_of_n", "weighted"]

                                reward = [sigmoid(r) for r in answer["reward"]]
                                # final reward is the minimun of all rewards
                                if len(reward) == 0:
                                    final_reward = 0.0
                                elif args.aggregation == "prod":
                                    final_reward = 1
                                    for r in reward:
                                        final_reward *= r
                                elif args.aggregation == "min":
                                    final_reward = min(reward)
                                elif args.aggregation == "max":
                                    final_reward = max(reward)
                                elif args.aggregation == "mean":
                                    final_reward = sum(reward) / len(reward)
                                elif args.aggregation == "sum_logit":
                                    final_reward = 1
                                    for r in reward:
                                        final_reward += math.log(r / (1 - r))
                                    final_reward = sigmoid(final_reward)
                                elif args.aggregation == "mean_logit":
                                    final_reward = 1
                                    for r in reward:
                                        final_reward += math.log(r / (1 - r))
                                    final_reward /= len(reward)
                                    final_reward = sigmoid(final_reward)
                                elif args.aggregation == "mean_odd":
                                    final_reward = 1
                                    for r in reward:
                                        final_reward += r / (1 - r)
                                    final_reward /= len(reward)
                                    if final_reward < 0:
                                        final_reward = 0
                                elif args.aggregation == "sum_odd":
                                    final_reward = 0
                                    for r in reward:
                                        final_reward += r / (1 - r)
                                    if final_reward < 0:
                                        final_reward = 0
                                elif args.aggregation == "last":
                                    final_reward = reward[-1]
                                else:
                                    raise ValueError("Unknown aggregation")

                                idx_to_answer[answer["idx"]].append(
                                    (concise_answer, final_reward)
                                )
                            else:
                                assert args.decoding_mode in ["majority"]
                                idx_to_answer[answer["idx"]].append(concise_answer)

                    if not filled_flag:
                        idx_to_answer[answer["idx"]].append("I don't know")
                    n_samples = max(n_samples, len(idx_to_answer[answer["idx"]]))

        # print the distribution of number of solutions per problem
        num_solution_average = sum(
            [len(idx_to_answer[idx]) for idx in idx_to_answer]
        ) / len(idx_to_answer)
        print("Average number of solutions per problem:", num_solution_average)
        print("Max number of solutions per problem:", n_samples)

        # do majority voting or weighted voting
        voting_nums = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        voting_nums = [num for num in voting_nums if num <= n_samples]
        voting_accs = []
        for voting_num in voting_nums:
            if voting_num == n_samples:
                n_trials = 1
            else:
                n_trials = 400
            for trial in tqdm.tqdm(range(n_trials)):
                type_list = set()
                level_list = set()
                results_table = {}
                for idx in idx_to_answer:
                    np.random.shuffle(idx_to_answer[idx])
                    answers = idx_to_answer[idx]
                    answers = answers[:voting_num]

                    # filter out "I don't know"
                    answers = [_ for _ in answers if _ != "I don't know"]
                    if len(answers) == 0:
                        model_short_answer = "I don't know"
                    else:
                        if args.decoding_mode in ["best_of_n", "weighted"]:
                            # weighted voting
                            # answers = [(answer, reward), ...]
                            # reward is a float between 0 and 1
                            # we get the answer with the highest sum of rewards
                            answer_reward = {}
                            for answer, reward in answers:
                                if answer not in answer_reward:
                                    answer_reward[answer] = 0.0
                                if args.decoding_mode == "best_of_n":
                                    answer_reward[answer] = max(
                                        reward, answer_reward[answer]
                                    )
                                else:
                                    answer_reward[answer] += reward

                            model_short_answer = max(
                                answer_reward, key=answer_reward.get
                            )
                        else:
                            # unique, counts = np.unique(answers, return_counts=True)
                            # counter_dict = dict(zip(unique, counts))
                            # model_short_answer = max(counter_dict, key=counter_dict.get)
                            counter = Counter(answers)
                            model_short_answer = counter.most_common(1)[0][0]

                    if "type" in idx_to_prompt[idx]:
                        question_type = idx_to_prompt[idx]["type"]
                    elif "subject" in idx_to_prompt[idx]:
                        question_type = idx_to_prompt[idx]["subject"]
                    else:
                        raise ValueError("No type or subject in prompt")
                    level = idx_to_prompt[idx]["level"]

                    type_list.add(question_type)
                    level_list.add(level)

                    # Initialize dictionary entries if they do not exist
                    if question_type not in results_table:
                        results_table[question_type] = {}
                    if level not in results_table[question_type]:
                        results_table[question_type][level] = [0, 0]

                    if model_short_answer == "I don't know":
                        gt_score = False
                    else:
                        gt_score = idx_answer_to_score[(idx, model_short_answer)]

                    if gt_score:
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
                    if gt_score:
                        results_table["Overall"][level][0] += 1
                    if "Overall" not in results_table["Overall"]:
                        results_table["Overall"]["Overall"] = [0, 0]
                    results_table["Overall"]["Overall"][1] += 1
                    if gt_score:
                        results_table["Overall"]["Overall"][0] += 1

                if trial == 0:
                    print(answer_file)
                    print("Voting num:", voting_num)

                # Sort the lists
                type_list = sorted(list(type_list))
                level_list = sorted(list(level_list), key=lambda x: int(x[-1]))
                # Print the table
                all_acc = print_table(
                    results_table, type_list, level_list, printing=(trial == 0)
                )
                all_acc["voting_num"] = voting_num
                all_acc["weighted_voting"] = args.decoding_mode in [
                    "best_of_n",
                    "weighted",
                ]
                if trial == 0:
                    print(all_acc)
                voting_accs.append(all_acc)
            print()
        if args.decoding_mode == "best_of_n":
            save_path = args.best_of_n_accs.replace(".json", f"_{file_cnt}.json")
        elif args.decoding_mode == "weighted":
            save_path = args.weighted_accs.replace(".json", f"_{file_cnt}.json")
        elif args.decoding_mode == "majority":
            save_path = args.majority_accs.replace(".json", f"_{file_cnt}.json")
        else:
            raise ValueError("Unknown decoding mode")
        # print(answer_file)
        # print("Is saved to", save_path)
        saving_map[answer_file] = (save_path, [_["Level1-5"] for _ in voting_accs])
        file_cnt += 1
        voting_accs.insert(0, answer_file)

        if args.mode == "prepare":
            with open(save_path, "w") as f:
                json.dump(voting_accs, f)

    for answer_file in sorted(glob.glob(answer_pattern)):
        print(answer_file, "Is saved to", saving_map[answer_file][0])
        # print("Voting accs:", saving_map[answer_file][1])
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_prefix", type=str, default="No Name")
    parser.add_argument(
        "--mode", type=str, default="prepare"
    )  # one of ['eval', 'figure', 'prepare']
    parser.add_argument(
        "--decoding_mode", type=str, default="best_of_n"
    )  # one of ['best_of_n', 'weighted', "majority"]
    parser.add_argument(
        "--aggregation",
        type=str,
        default="prod",
    )
    parser.add_argument("--gt_file", type=str, default=None)
    parser.add_argument("--answer_pattern", type=str, default=None)
    parser.add_argument(
        "--majority_accs",
        type=str,
        default=f"{DATA_DIR}/figs/majority_accs.json",
    )
    parser.add_argument(
        "--weighted_accs",
        type=str,
        default=f"{DATA_DIR}/figs/weighted_accs.json",
    )
    parser.add_argument(
        "--best_of_n_accs",
        type=str,
        default=f"{DATA_DIR}/figs/best_of_n_accs.json",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=f"{DATA_DIR}/figs/plot.pdf",
    )
    args = parser.parse_args()
    main(args)
