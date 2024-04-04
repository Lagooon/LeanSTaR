import json
import random

random.seed(42)
def extract_prm_v2_dataset(example):
    ret = {
            "input": "# Question\n\n" + example["input"] + "\n\n# Solution",
            "output": "\n\n" + example["output"],
        }

    return ret

data = []
for i in range(1, 6):
    file_path = f'/home/wliu/longhui/llms-all/ScalableMath_sun-main/output/MATH_test1-5_llemma-7b_prm_2e-5_128_num_samples3_{i}.jsonl'
    with open(file_path, 'r') as file:
        for line in file:
            json_data = json.loads(line)
            data.append(json_data)

file_path = f'/home/wliu/longhui/llms-all/ScalableMath_sun-main/output/MATH_test1-5_llemma-7b_prm_2e-5_128_num_samples1.jsonl'
with open(file_path, 'r') as file:
    for line in file:
        json_data = json.loads(line)
        data.append(json_data)
        
with open(f'/home/wliu/longhui/llms-all/ScalableMath_sun-main/output/test_merge.jsonl', 'w') as outfile:
    for entry in data:
        json.dump(entry, outfile)
        outfile.write('\n')
print(len(data))