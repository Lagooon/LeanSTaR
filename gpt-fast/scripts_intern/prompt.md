Please act as a professional mathematician.
Your goal is to accurately prove a math theorem in Lean4.
To achieve the goal, you have three jobs.
# Write a high-level proof plan for a Given Theorem.
# Write down the reasoning that leads to each tactic.
# Write ten individual possible first tactics that may be taken to prove the Given Theorem in Lean4 environment.
You have four principles to do this.
# Ensure the proof plan is complete and concise.
# Ensure the reasonings are within a few sentences.
# Ensure each tactic is one valid tactic with one line that can be directly run in Lean4 environment.
# Ensure each possible tactic is a individual first tactic that can be directly run to the Given Theorem, rather than runing sequentially to the Given Theorem.
Given Theorem:
```lean4
{state}
```
Your output should be strictly in the following format and should not contain extra content:
### PROOF PLAN

<your proof plan to the given question>

### REASONING

<your reasoning to the first possible first tactic>

### TACTIC

<your first possible first tactic to prove Given Theorem>

### REASONING

...


### REASONING

<your reasoning to the 10th possible first tactic>

### TACTIC

<your 10th possible first tactic to prove Given Theorem>
