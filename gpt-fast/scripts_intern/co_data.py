import json
import os
import random
def chat_template_to_prompt(prompt_list):
    result = ""
    total_step = len(prompt_list)
    for i, message in enumerate(prompt_list):
        result += ('<|im_start|>' + message['role'] +
            '\n' + message['content'])
        if i+1 != total_step:
            result += '<|im_end|>\n'
        elif message['role'] == 'user':
            result += '<|im_end|>\n<|im_start|>assistant\n'
    return result
def _prompt(ts):
    prompt = f"My LEAN 4 state is:\n```lean\n" + ts + \
        "```\nPlease predict a possible tactic to help me prove the theorem."
    prompt = [{"role": "user", "content": prompt}]
    prompt = chat_template_to_prompt(prompt)
    return prompt
random.seed(1926)
paths = ["01-07-2024-15-56", "01-07-2024-16-26", "01-07-2024-16-27", "01-07-2024-16-30", "01-07-2024-16-31", "01-07-2024-21-16", "01-07-2024-21-18", "02-07-2024-18-57", "02-07-2024-18-58", "02-07-2024-19-00", "03-07-2024-16-02", "03-07-2024-16-03", "03-07-2024-16-04"]
data = []
cntt = 0
visited = set()
for pa in paths:
    for i in range(10):
        p = "output/internLM2-7b-sample_mathlib_train/" + pa + f"/results__internLM-7b-math__{i}.json"
        if not os.path.isfile(p):
            continue
        with open(p, "r") as f:
            js = json.load(f)
        for obj in js["results"]:
            if obj["success"] == False or obj["attempt_results"][0]["theorem"] in visited:
                continue
            #print(obj["attempt_results"])
            visited.add(obj["attempt_results"][0]["theorem"])
            random.shuffle(obj["attempt_results"])
            for ob in obj["attempt_results"][:3]:
                state_before = None
                cntt += 1
                for tactics in ob["trace"]:
                    if tactics["state_before"] == state_before:
                        data = data[:-1]
                    data.append({
                        "input" : _prompt(tactics["state_before"]),
                        "output": tactics["full_cot"] + "<|im_end|>"
                    })
                    state_before = tactics["state_before"]
print(len(data), cntt)
data1 = []
for obj in data:
    tactic = obj["output"].split("```lean\n")[-1].split("\n```")[0]
    if '\n' not in tactic:
        data1.append(obj)
print(len(data1))
with open(f"data/leandojo_benchmark_4/processed/STaR-generated-train-sft.json", "w") as f:
    json.dump(data1, f, indent=4)
                                                                                                                                
