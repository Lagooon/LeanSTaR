import json

import tqdm

import random

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llemma_34b")


if __name__ == "__main__":
    file1 = "/nobackup/users/yikangs/zhiqings/math/train_1_2_3_metamath.json"

    file2 = "/nobackup/users/yikangs/zhiqings/math/train_1_2_3_prm_v4.json"

    file_out = "/nobackup/users/yikangs/zhiqings/math/train_1_2_3_metaprm_v4.json"

    with open(file1, "r") as f:
        data1 = json.load(f)

    total_output_tokens_1 = 0

    for ex in tqdm.tqdm(data1):
        total_output_tokens_1 += len(tokenizer(ex["output"])["input_ids"])

    print("Number of MetaMath annotations: ", len(data1))
    print("Total output tokens: ", total_output_tokens_1)

    with open(file2, "r") as f:
        data2 = json.load(f)

    total_output_tokens_2 = 0

    for ex in tqdm.tqdm(data2):
        total_output_tokens_2 += len(tokenizer(ex["output"])["input_ids"])

    print("Number of MetaMath annotations: ", len(data2))
    print("Total output tokens: ", total_output_tokens_2)

    data = data1 + data2

    print("Number of MetaMath annotations: ", len(data))

    random.shuffle(data)

    # with open(file_out, "w") as f:
    #     json.dump(data, f, indent=2)
