import json

with open('/home/wliu/longhui/llms-all/ScalableMath_sun-main/data/train_1_2_3_metamath_v4.json', "r") as f:
    data = json.load(f)
print(data[0])
print(len(data))
def extract_prm_v2_dataset(example):
    ret = {
        "input": "# Question\n\n" + example["query"] + "\n\n# Solution",
        "output": "\n\n" + example["output"],
        "is_eos": True,
    }

    return ret

map_list = []
for item in data:
    map_item = extract_prm_v2_dataset(item)
    map_list.append(map_item)
    
with open('/home/wliu/longhui/llms-all/ScalableMath_sun-main/data/train_1_2_3_metamath_v4_mapped.json', "w") as f:
    json.dump(map_list, f)
