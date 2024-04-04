import json

if __name__ == "__main__":
    file1 = "/nobackup/users/yikangs/zhiqings/math/outputs/test_1_2_3_prm_v3_34b_epoch-3_lr-2e-5_seq-768_g.jsonl"
    # file2 = "/nobackup/users/yikangs/zhiqings/math/outputs/test_1_2_3_prm_v3_7b_v4_epoch-3_lr-2e-5_seq-768_g2.jsonl"

    id2prompt = {}
    id2output1 = {}

    with open(file1, "r") as f:
        for line in f:
            line = json.loads(line.strip())
            id2prompt[line["idx"]] = line["prompt"]
            id2output1[line["idx"]] = line["output"]

    # id2output2 = {}

    # with open(file2, "r") as f:
    #     for line in f:
    #         line = json.loads(line.strip())
    #         id2output2[line["idx"]] = line["output"]

    for idx in range(len(id2output1)):
        print("idx: ", idx)
        print("-" * 20)
        print("Prompt: ", id2prompt[idx])
        print("-" * 20)
        print("Result 1: ", id2output1[idx])
        # print("-" * 20)
        # print("Result 2: ", id2output2[idx])
        print("=" * 20)
        input()
