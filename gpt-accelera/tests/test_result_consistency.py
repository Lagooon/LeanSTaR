import json
import glob

if __name__ == "__main__":
    # file1_name = "/workspace/zhiqings/output4/math/outputs/test_1to5_prm_v3_max256_7b-metamath_ppo-vd_g.jsonl"
    file1_name = "/workspace/zhiqings/output4/math/outputs/test_1to5_prm_v3_max256_7b-metamath_ppo-vd_g_fp32.jsonl"
    # file1 format
    # {"idx": int, "prompt": str, "output": str}

    file1_prompt_to_output = {}
    with open(file1_name, "r") as f:
        for line in f:
            data = json.loads(line)
            file1_prompt_to_output[data["prompt"]] = data["output"]

    print("Length of file1: ", len(file1_prompt_to_output))

    file2_name = "/workspace/zhiqings/output4/math/checkpoints/llemma-7b-ppo_metamath-1to5_epoch-50_lr-1e-6_seq-768-vd/evaluate/eval_results_410_rank_*.json"

    file2_prompt_to_output = {}

    for file2 in glob.glob(file2_name):
        with open(file2, "r") as f:
            data = json.load(f)
            for line in data:
                file2_prompt_to_output[line["text_query"]] = line["text_response"].split("</s>")[0].strip()
    print("Length of file2: ", len(file2_prompt_to_output))

    # assert that all prompts in file1 are in file2
    for prompt in file1_prompt_to_output:
        assert prompt in file2_prompt_to_output

    num_diff = 0

    # check that all outputs in file1 are in file2, otherwise print them out
    for prompt in file1_prompt_to_output:
        if file1_prompt_to_output[prompt] != file2_prompt_to_output[prompt]:
            print("==== Prompt ====", prompt)
            print("==== File1 output ====", file1_prompt_to_output[prompt])
            print("==== File2 output ====", file2_prompt_to_output[prompt])
            print()
            num_diff += 1

    print("Number of different outputs: ", num_diff)
