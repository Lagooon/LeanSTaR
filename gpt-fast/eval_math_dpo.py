import glob
import json
from collections import Counter
import random
import logging

import math
import argparse

import tqdm
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


BEST_OF_N = True


def main(args):
    idx_answer_to_score = {}

    gt_file = args.gt_file
    answer_pattern = args.answer_pattern

    with open(gt_file, "r") as f:
        gt_prompts = json.load(f)

    prompt_idx = list(range(len(gt_prompts)))
    idx_to_prompt = {idx: prompt for idx, prompt in zip(prompt_idx, gt_prompts)}

    for answer_file in sorted(glob.glob(answer_pattern)):
        weighted_voting = False
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

                    if len(answer["output"].split("# Answer\n\n")) == 2:
                        concise_answer = (
                            answer["output"].rsplit("# Answer\n\n")[1].strip()
                        )
                    else:
                        concise_answer = "I don't know"

                    with DisableLogger():
                        if (
                            answer["idx"],
                            concise_answer,
                        ) not in idx_answer_to_score:
                            gt_score = grader.grade_answer(
                                concise_answer,
                                idx_to_prompt[answer["idx"]]["gt_answer"],
                            )
                            idx_answer_to_score[(answer["idx"], concise_answer)] = (
                                gt_score
                            )

                    if "reward" in answer:
                        weighted_voting = True

                        reward = [sigmoid(r) for r in answer["reward"]]
                        # final reward is the minimun of all rewards
                        final_reward = 1.0
                        for r in reward:
                            # final_reward *= r
                            final_reward = min(final_reward, r)

                        if concise_answer == "I don't know":
                            final_reward = final_reward - 1.0
                        elif len(concise_answer) == 0:
                            final_reward = final_reward - 2.0

                        idx_to_answer[answer["idx"]].append(
                            (concise_answer, final_reward, answer["output"])
                        )
                    else:
                        idx_to_answer[answer["idx"]].append(concise_answer)

                    n_samples = max(n_samples, len(idx_to_answer[answer["idx"]]))

        # print the distribution of number of solutions per problem
        num_solution_average = sum(
            [len(idx_to_answer[idx]) for idx in idx_to_answer]
        ) / len(idx_to_answer)
        print("Average number of solutions per problem:", num_solution_average)
        print("Max number of solutions per problem:", n_samples)

        save_results = None
        if args.save_to_file is not None:
            assert weighted_voting and BEST_OF_N
            save_results = []

        voting_num = n_samples
        results = [0, 0]
        for idx in idx_to_answer:
            np.random.shuffle(idx_to_answer[idx])
            answers = idx_to_answer[idx]
            answers = answers[:voting_num]
            answer_reward = {}
            best_solution = (None, -5.0)
            bad_solutions = []

            for answer, reward, solution in answers:
                if answer not in answer_reward:
                    answer_reward[answer] = 0.0
                if BEST_OF_N:
                    answer_reward[answer] = max(reward, answer_reward[answer])
                else:
                    answer_reward[answer] += reward

            if save_results is not None:
                for answer, reward, solution in answers:
                    answer_gt_score = idx_answer_to_score[(idx, answer)]
                    reward = reward + 1.0 if answer_gt_score else reward

                    if reward > 0.0 and reward > best_solution[1] and answer_gt_score:
                        best_solution = (solution, reward, answer, answer_gt_score)

                for answer, reward, solution in answers:
                    answer_gt_score = idx_answer_to_score[(idx, answer)]
                    reward = reward + 1.0 if answer_gt_score else reward
                    if reward < best_solution[1] - 0.5:
                        bad_solutions.append(
                            (solution, reward, answer, answer_gt_score)
                        )

            if len(bad_solutions) > 0 and best_solution[1] > 0.0:
                random.shuffle(bad_solutions)
                for bad_solution in bad_solutions[: args.negative_duplicate]:
                    save_results.append(
                        {
                            "instruction": "",
                            "input": idx_to_prompt[idx]["input"],
                            "output_1": best_solution[0],
                            "output_2": bad_solution[0],
                            "output_1_score": best_solution[1],
                            "output_2_score": bad_solution[1],
                            "output_1_answer": best_solution[2],
                            "output_2_answer": bad_solution[2],
                            "preference": 1,
                            "subject": idx_to_prompt[idx]["subject"],
                            "level": idx_to_prompt[idx]["level"],
                            "gt_answer": idx_to_prompt[idx]["gt_answer"],
                            "answer": idx_to_prompt[idx]["answer"],
                            "is_eos_1": "# Answer\n\n" in best_solution[0],
                            "is_eos_2": "# Answer\n\n" in bad_solution[0],
                        }
                    )

            model_short_answer = max(answer_reward, key=answer_reward.get)

            gt_score = idx_answer_to_score[(idx, model_short_answer)]

            # Overall accuracy
            results[1] += 1
            if gt_score:
                results[0] += 1

        print("File:", answer_file)
        print("Overall accuracy:", results[0] / results[1])

        if save_results is not None:
            # shuffle examples and output_1 and output_2
            random.shuffle(save_results)
            for result in save_results:
                if random.random() < 0.5:
                    result["output_1"], result["output_2"] = (
                        result["output_2"],
                        result["output_1"],
                    )
                    result["output_1_score"], result["output_2_score"] = (
                        result["output_2_score"],
                        result["output_1_score"],
                    )
                    result["output_1_answer"], result["output_2_answer"] = (
                        result["output_2_answer"],
                        result["output_1_answer"],
                    )
                    result["is_eos_1"], result["is_eos_2"] = (
                        result["is_eos_2"],
                        result["is_eos_1"],
                    )
                    result["preference"] = 2

            with open(args.save_to_file, "w") as f:
                json.dump(save_results, f, indent=2)

            print("Total number of examples:", len(save_results))
            print("Saved to", args.save_to_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_file",
        type=str,
    )
    parser.add_argument(
        "--answer_pattern",
        type=str,
    )
    parser.add_argument(
        "--save_to_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--negative_duplicate",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    main(args)
