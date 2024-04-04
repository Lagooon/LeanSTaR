import os
import json
import random

from openai import AzureOpenAI



client = AzureOpenAI(

  azure_endpoint = "https://haohan-0125.openai.azure.com/", 

  api_key=os.getenv("AZURE_OPENAI_KEY"),  

  api_version="2024-02-15-preview"

)


def generate(theorem):
    prompt = f"""Please act as a professional mathematician.
Your goal is to accurately prove a math theorem in Lean4.
You are given the first tactic that should be taken to prove the Given Theorem.

# Example:

Given Theorem:
```lean4
X : Type u_1
Y : Type u_2
inst✝¹ : TopologicalSpace X
inst✝ : TopologicalSpace Y
f : X →. Y
h : PContinuous f
⊢ IsOpen (PFun.preimage f Set.univ)
```
Given Tactic:
```lean4
exact h _ isOpen_univ
```
Assistant Output:
### PLANNING OF REASONING
Given the above goal and the given the target tactic `exact h _ isOpen_univ`, I should produce a reasoning that only based on the goal, and conclude to the `exact h _ isOpen_univ`, but not in the hindsight format.

### REASONING

By directly using the hypothesis `h : PContinuous f`, we state that for any open set, its preimage via `f` is also open, and this applies to `Set.univ` as well.

### TACTIC

```lean4
exact h _ isOpen_univ
```

To achieve the goal, you have three jobs.
# Write down the planning of reasoning.
# Write down the reasoning that can result in the selection of the Given Tactic.
# Write down the Given Tactic.
You have four principles to do this.
# Craft reasoning that is comprehensive yet concise, contained ideally within two sentences.
# Begin reasoning with the objective of the Given Theorem, ensuring logical progression, rather than constructing it retrospectively from the Given Tactic.
# Avoid mentioning the Given Tactic and specific statement names from the Given Tactic throughout your reasoning.
# Position the conclusion of the tactic after presenting all reasons, ensuring that details regarding the Given Tactic are not introduced at the start of your reasoning.
Your output should be strictly in the following format and should not contain extra content:

### PLANNING OF REASONING

Given the above goal and the given the target tactic <Given Tactic>, I should produce a reasoning that only based on the goal, and conclude to the exact <Given Tactic>, but not in the hindsight format.

### REASONING

<your reasoning to the Given Tactic>

### TACTIC

<Given Tactic>

# Given Theorem:
```lean4
{theorem["state_before"]}
```

# Given Tactic:
```lean4
{theorem["tactic"]}
```

"""
    prompt = prompt.encode("utf-16", 'surrogatepass').decode('utf-16')
    message_text = [{"role":"system","content":"You are an AI assistant that helps people find information."},
                    {"role":"user","content":prompt}]

    completion = None
    for _ in range(5):
        try:
            completion = client.chat.completions.create(
            model="haohan-gpt4", # model = "deployment_name"
            messages = message_text,
            temperature=0.7,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
            )
            break
        except Exception as e:
            print(e)
    if completion is not None:
        example = {
            "state_before": theorem["state_before"],
            "state_after": theorem["state_after"],
            "tactic": theorem["tactic"],
            'file_path' : theorem["file_path"],
            'full_name' : theorem["full_name"],
            "output": completion.choices[0].message.content,
        }
    else:
        example = None
    return example

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/leandojo_benchmark_4')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--size', type=int, default=200)
    args = parser.parse_args()

    dataset = json.load(open(os.path.join(
            args.data_dir, 'processed', 'proofstep_g-train.json'), 'r')
        )
    random.seed(42)
    random.shuffle(dataset)
    dataset = dataset[args.start:args.start+args.size]
    examples = []
    for data in dataset:
        example = generate(data)
        if example == None:
            print(data)
        else:
            examples.append(example)
            print("\n-----OUTPUT-----")
            print(example["output"].encode("utf-16", 'surrogatepass').decode('utf-16'))
    result_file = os.path.join(args.data_dir, 'processed', f'gpt4-generated-{args.start}-{args.start+args.size}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=4)