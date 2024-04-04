import json
########################
# The input in the saved file contains only the question information, not the output_prefix information of the PRM
########################
with open('/home/wliu/longhui/llms-all/ScalableMath_sun-main/data/train_1_2_3_prm_v4.json', "r") as f:
    data = json.load(f)

def extract_prm_v2_dataset(example):
    ret = {
        "input": "# Question\n\n" + example["input"] + "\n\n# Solution",
        "output": "\n\n" + example["output_prefix"] + example["output"],
    }
    if "is_eos" in example:
        ret["is_eos"] = example["is_eos"]

    return ret

map_list = []
for item in data:
    map_item = extract_prm_v2_dataset(item)
    if map_item['is_eos']:
        map_list.append(map_item)
    
with open('/home/wliu/longhui/llms-all/ScalableMath_sun-main/data/train_1_2_3_prm_question_for_rest.json', "w") as f:
    json.dump(map_list, f)
