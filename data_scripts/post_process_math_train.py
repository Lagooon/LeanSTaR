import json

if __name__ == "__main__":
    data_path = "/nobackup/users/yikangs/zhiqings/math/"
    # data_path = "/workspace/zhiqings/output3/data/"

    input_file = data_path + "train_sft.json"

    output_file = data_path + "train_sft_post.json"

    with open(input_file, "r") as f:
        data = json.load(f)

    for i in range(len(data)):
        data[i]["output"] = data[i]["output"] + "\n\n### User"

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
