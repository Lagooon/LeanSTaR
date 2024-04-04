from collections import Counter
import json

if __name__ == "__main__":
    file_name = "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json"

    # We get statistics for "level"

    with open(file_name, "r") as f:
        gt_data = json.load(f)

    level_counter = Counter()
    for prompt in gt_data:
        level_counter[prompt["level"]] += 1

    print(level_counter)
