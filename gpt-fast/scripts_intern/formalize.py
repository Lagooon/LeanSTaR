import os
import json
from lean_dojo import *




def _tactic_state(state):
    if isinstance(state, TacticState):
        ts = state.pp
    else:
        ts = None
    return ts

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/leandojo_benchmark_4/processed/gpt4-generated-80-100.json')
    parser.add_argument('--size', type=int, default=200)
    args = parser.parse_args()
    
    dataset = json.load(open(args.data_dir))
    results = []
    repo = LeanGitRepo(
        "https://github.com/leanprover-community/mathlib4",
        "3c307701fa7e9acbdc0680d7f3b9c9fed9081740"
    )
    for data in dataset:
        output = data["output"]
        output = output.split("### PROOF PLAN\n\n")[-1]
        plan, r_tacs = output.split("### REASONING\n\n")[0], output.split("### REASONING\n\n")[1:]
        result = []
        theorem = Theorem(repo, data["file_path"], data["full_name"])
        with Dojo(theorem, hard_timeout=600) as (dojo, state):
            for tac in data["tactics_before"]:
                state = dojo.run_tac(state, tac)
            assert _tactic_state(state) == data["state_before"]
            for r_tac in r_tacs:
                r_tac = r_tac.split("### TACTIC\n\n")
                if len(r_tac) != 2:
                    continue
                reason, tactic = r_tac[0], r_tac[1]
                tactic = tactic.split("```lean4\n")[-1].split("```")[0].strip().strip(',')
                gen_state = dojo.run_tac(state, tactic)
                print(gen_state, tactic)
                if isinstance(gen_state, ProofFinished) or _tactic_state(gen_state) == data["state_after"]:
                    result.append({"input":"",})
        if len(result) > 0:
            results.append(result[0])
            print("SUCCESS")
        else:
            print("----------STATE----------")
            print(data["state_before"])
            #print("----------TACTIC----------")
            #print(data["output"])
    print(len(results))
    
    