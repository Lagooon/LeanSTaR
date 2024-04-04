Please act as a professional mathematician.
Your goal is to accurately prove a math theorem in Lean4.
You are given the first tactic that should be taken to prove the Given Theorem.
To achieve the goal, you have three jobs.
# Write a high-level proof plan for a Given Theorem.
# Write down the reasoning that leads to Given Tactic.
# Write down the Given Tactic.
You have three principles to do this.
# Ensure the proof plan is complete and concise.
# Ensure the reasoning is within a few sentences.
# Ensure the proof plan and reasoning can be naturally derived from the goal Given Theorem rather than being inferred in reverse from a the Given Tactic.
Given Theorem:
```lean4
{state}
```
Given Tactic:
```lean4
{tactic}
```
Your output should be strictly in the following format and should not contain extra content:
### PROOF PLAN

<your proof plan to the given question>

### REASONING

<your reasoning to the Given Tactic>

### TACTIC

<Given Tactic>


