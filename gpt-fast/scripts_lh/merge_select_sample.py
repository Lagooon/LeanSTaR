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
for i in range(1, 17):
    try:
        file_path = f'/home/wliu/longhui/llms-all/ScalableMath_sun-main/outputs/MATH_train1-3_llemma-7b_prm_2e-5_128_num_samples8_tmp0.9_{i}.jsonl'
        with open(file_path, 'r') as file:
            for line in file:
                json_data = json.loads(line)
                data.append(json_data)
    except:
        pass
with open('/home/wliu/longhui/llms-all/ScalableMath_sun-main/data/prm_splits_MATH_train-cleaned_processed.json', 'r') as f:
    math_train = json.load(f)
new_data = []
print(len(data))
for item in data:
    if "# Answer\n\n" in item["output"]:
        concise_answer = item["output"].rsplit("# Answer\n\n")[1].strip()
        if concise_answer == math_train[item['idx']]['answer']:
            if math_train[item['idx']]['level'] in ['Level 1', 'Level 2', 'Level 3']:
            # import pdb;pdb.set_trace()
                new_data.append(item)

subset = []
selected_count = {} 

for item in new_data:
    idx = item.get('idx')
    if idx is not None and selected_count.get(idx, 0) < 5:
        item['input'] = item['prompt']
        subset.append(item)
        selected_count[idx] = selected_count.get(idx, 0) + 1
with open('/home/wliu/longhui/llms-all/ScalableMath_sun-main/data/train_1_2_3_prm_v4_mapped.json', 'r') as f:
    ori_train = json.load(f)
    
rest_train = ori_train + subset
rest_train = subset
rest_train = random.sample(rest_train, len(rest_train))
print(len(ori_train), len(subset), len(rest_train))
with open('/home/wliu/longhui/llms-all/ScalableMath_sun-main/data/rest_train_mapped_only_tmp0.7.json', "w") as f:
    json.dump(rest_train, f)
    
