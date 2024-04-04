import json
from typing import List

import random
import tqdm
import fire

from datasets import load_dataset

from transformers import AutoTokenizer

import llm_blender

blender = llm_blender.Blender()
blender.loadranker("llm-blender/PairRM")

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")


def main(
    labeled_data_path: str = "/nobackup/users/yikangs/zhiqings/dpo/alpaca_rm_10k_v2.json",
    unlabeled_data_path: str = "/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-unlabeled-20k-rpo-v2-seq-300.json",
    output_path: str = "/nobackup/users/yikangs/zhiqings/dpo/alpaca_rm_mixlabeled_rpo-v2-v6_30k.json",
    # preference_model_augmentation: bool = True,
):
    outputs = []

    with open(labeled_data_path, "r") as f:
        outputs = json.load(f)

    print("Restoring labeled data from", labeled_data_path)
    print(f"Number of labeled data: {len(outputs)}")

    idx_to_sample = {}
    idx_to_sample_idx = {}
    idx_to_output = {}

    with open(unlabeled_data_path, "r") as f:
        for line in f:
            example = json.loads(line)

            idx = example["idx"]
            prompt = example["prompt"]
            output = example["output"]
            sample_idx = example["sample_idx"]

            if prompt.count("\n\n### Instruction:\n") == 1:
                assert prompt.count("\n\n### Response:\n") == 1

                if "\n\n### Input:\n" in prompt:
                    assert prompt.count("\n\n### Input:\n") == 1
                    # instruction is after "\n\n### Instruction:\n" and before "\n\n### Input:\n"
                    instruction = prompt.split("\n\n### Instruction:\n")[1].split(
                        "\n\n### Input:\n"
                    )[0]
                    input_ = prompt.split("\n\n### Input:\n")[1].split(
                        "\n\n### Response:\n"
                    )[0]
                else:
                    # instruction is after "\n\n### Instruction:\n" and before "\n\n### Response:\n"
                    instruction = prompt.split("\n\n### Instruction:\n")[1].split(
                        "\n\n### Response:\n"
                    )[0]
                    input_ = ""

                if idx not in idx_to_sample:
                    idx_to_sample_idx[idx] = []
                    idx_to_sample[idx] = []
                    idx_to_output[idx] = []

                if sample_idx not in idx_to_sample_idx[idx]:
                    if output not in idx_to_output[idx]:
                        idx_to_sample_idx[idx].append(sample_idx)
                        idx_to_output[idx].append(output)
                        idx_to_sample[idx].append(
                            {
                                "instruction": instruction,
                                "input": input_,
                                "output": output,
                            }
                        )
            else:
                print([prompt])
                raise NotImplementedError

    num_skipped_idx = 0

    count_num_outputs_per_idx = {}
    for idx in idx_to_sample:
        count_num_outputs_per_idx[len(idx_to_sample[idx])] = (
            count_num_outputs_per_idx.get(len(idx_to_sample[idx]), 0) + 1
        )
    print(f"count_num_outputs_per_idx: {count_num_outputs_per_idx}")

    all_inputs = []
    all_candidate_texts = []
    all_samples = []
    all_paddings = []

    num_samples = max(count_num_outputs_per_idx.keys())
    padding_candidate_text = "padding_candidate_text"

    for idx in idx_to_sample:
        instruction = idx_to_sample[idx][0]["instruction"]
        input_ = idx_to_sample[idx][0]["input"]

        if len(idx_to_sample[idx]) == num_samples:
            all_inputs.append(
                instruction + "\n\n" + input_ if input_ != "" else instruction
            )

            all_samples.append([])
            all_candidate_texts.append([])

            for sub_idx in range(len(idx_to_sample[idx])):
                all_samples[-1].append(idx_to_sample[idx][sub_idx])
                all_candidate_texts[-1].append(idx_to_sample[idx][sub_idx]["output"])

            all_paddings.append(len(idx_to_sample[idx]))

            if len(idx_to_sample[idx]) < num_samples:
                for _ in range(num_samples - len(idx_to_sample[idx])):
                    all_samples[-1].append(idx_to_sample[idx][0])
                    all_candidate_texts[-1].append(padding_candidate_text)

    # max_examples = 1024
    # all_inputs = all_inputs[:max_examples]
    # all_candidate_texts = all_candidate_texts[:max_examples]
    # all_samples = all_samples[:max_examples]
    # all_paddings = all_paddings[:max_examples]

    all_scores = blender.rank(
        all_inputs,
        all_candidate_texts,
        return_scores=True,
        batch_size=32,
        policy="max_probs",
    )

    print(all_scores.shape)

    all_scores = ((all_scores * num_samples - 0.5) / (num_samples - 1.0)).tolist()

    print(sum(all_scores[0]))

    for i in range(len(all_samples)):
        i_scores = all_scores[i][: all_paddings[i]]
        i_samples = all_samples[i][: all_paddings[i]]

        best_score_idx = i_scores.index(max(i_scores))
        worst_score_idx = i_scores.index(min(i_scores))

        best_sample = i_samples[best_score_idx]
        worst_sample = i_samples[worst_score_idx]

        best_score = max(i_scores)
        worst_score = min(i_scores)

        best_sample_eos = True
        worst_sample_eos = True

        def is_sentence_ended_with_az(sentence):
            # Check if the last character of the sentence is a lowercase letter between a and z
            return sentence[-1].islower() and "a" <= sentence[-1] <= "z"

        if is_sentence_ended_with_az(best_sample["output"]):
            best_sample_eos = False
        if "###" in best_sample["output"]:
            best_sample_eos = False

        if is_sentence_ended_with_az(worst_sample["output"]):
            worst_sample_eos = False
        if "###" in worst_sample["output"]:
            worst_sample_eos = False

        better_sample = -1
        if best_sample_eos and not worst_sample_eos:
            better_sample = 1
        if worst_sample_eos and not best_sample_eos:
            better_sample = 2

        if better_sample == 1:
            gt_best_score = 1.0
            gt_worst_score = 0.0
        elif better_sample == 2:
            gt_best_score = 0.0
            gt_worst_score = 1.0
        else:
            gt_best_score = 0.5
            gt_worst_score = 0.5

        best_score = (best_score + gt_best_score) / 2
        worst_score = (worst_score + gt_worst_score) / 2

        outputs.append(
            {
                "instruction": best_sample["instruction"],
                "input": best_sample["input"],
                "output_1": best_sample["output"],
                "output_2": worst_sample["output"],
                "preference": 1 if best_score > worst_score else 2,
                "win_rate_1": best_score,
                "win_rate_2": worst_score,
                "type": "iter-2",
            }
        )

    print(f"Number of skipped idx in unlabeled: {num_skipped_idx}")

    random.shuffle(outputs)

    print(f"Number of alpaca_farm mix-unlabeld data: {len(outputs)}")

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
