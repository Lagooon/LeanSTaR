import glob
import json
import os
import random
import re
import tqdm
from typing import List

import fire

from datasets import load_dataset

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llemma_34b")

import re

def remove_lines_with_pattern(input_string, pattern):
    matches = re.findall(pattern, input_string)

    for match in matches:
        input_string = input_string.replace(match, '')

    return input_string

def add_newlines_after_dot_space(input_string):
    dot_space_positions = [pos for pos, char in enumerate(input_string) if char == '.' and input_string[pos + 1] == ' ']

    for pos in reversed(dot_space_positions):
        input_string = input_string[:pos + 2] + '\n\n' + input_string[pos + 2:]

    return input_string



def main(
    levels: List[str] = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"],
    output_path: str = "/home/wliu/longhui/llms-all/ScalableMath_sun-main/data/train_1_2_3_4_5_metamath_v4.json",
    pruned_output_path: str = "/home/wliu/longhui/llms-all/ScalableMath_sun-main/data/train_1_2_3_4_5_metamath_v4_pruned.json",
    pruned_numbers: int = 8,
    metamath_path: str = "/home/wliu/longhui/llms-all/ScalableMath_sun-main/data/MetaMathQA-395K.json",
    math_path: str = (
        "/home/wliu/longhui/llms-all/ScalableMath_sun-main/data/MATH_train-cleaned_processed.json"
    ),
):
    math_dataset = {}

    math_dataset["train"] = []
    math_dataset["test"] = []

    with open(metamath_path, "r") as f:
        data = json.load(f)
        
    with open(math_path, "r") as f:
        ori_math = json.load(f)
    
    select_list = []
    question_set = set()
    for item in ori_math:
        if item['level'] in levels:
            question_set.add(item['question'])
    for item in data:
        if 'MATH' in item['type']:
            if item['original_question'] in question_set:
                split_ans = item['response'].split('The answer is: ')
                item['output'] = split_ans[0] + '\n\n# Answer\n\n' + split_ans[1]
                select_list.append(item)
        else:
            split_ans = item['response'].split('The answer is: ')
            item['output'] = split_ans[0] + '\n\n# Answer\n\n' + split_ans[1]
            select_list.append(item)
    for item in select_list:
        item['output'] = add_newlines_after_dot_space(item['output'])
        pattern = r'#### \d+'
        item['output'] = remove_lines_with_pattern(item['output'], pattern)
        item['output'] = item['output'].replace('.\n', '.\n\n')
        item['output'] = item['output'].replace('\]\n', '\]\n\n')
        item['output'] = item['output'].replace('\] ', '\]\n\n')
        item['output'] = item['output'].replace('\]\n\n\n', '\]\n\n')
        item['output'] = item['output'].replace('\n\n\n\n\n\n\n# Answer', '\n\n# Answer')
        item['output'] = item['output'].replace('\n\n\n\n\n\n# Answer', '\n\n# Answer')
        item['output'] = item['output'].replace('\n\n\n\n\n# Answer', '\n\n# Answer')
        item['output'] = item['output'].replace('\n\n\n\n# Answer', '\n\n# Answer')
        item['output'] = item['output'].replace('\n\n\n# Answer', '\n\n# Answer')
    num_dict = {}
    for item in select_list:
        num_dict[item['original_question']] = 0
    pruned_metamath = []
    for item in select_list:
        if num_dict[item['original_question']] < pruned_numbers:
            pruned_metamath.append(item)
            num_dict[item['original_question']] +=1
    
    with open(output_path, "w") as f:
        json.dump(select_list, f, indent=2)
        
    with open(pruned_output_path, "w") as f:
        json.dump(pruned_metamath, f, indent=2)
        
    print(len(select_list), len(pruned_metamath))

if __name__ == "__main__":
    fire.Fire(main)
