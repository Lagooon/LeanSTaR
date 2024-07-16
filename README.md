# `Lean-STaR` MiniF2F performance reproduce 

Scripts for the Lean formal2formal (tactic prediction) experiments. Adapted from
[llemma_formal2formal](https://github.com/wellecks/llemma_formal2formal).


#### Setup
Install Python packages:
```
bash scripts/prepare_env.sh
```

Install Lean:
```
# from https://leanprover-community.github.io/install/linux.html
# After running this command, select (2), then `nightly`, then `y`:
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source $HOME/.elan/env
lake
```

Configure LeanDojo:
```
export CONTAINER="native"
```

#### Finetune
```
cd gpt-fast
bash scripts_intern/prepare_intern_math_7b.sh
bash scripts_intern/finetune_7b_intern.sh
bash scripts_intern/finetune_7b_cot.sh
bash scripts_intern/finetune_7b_star.sh
```

#### Evaluation
```
cd gpt-fast
bash scripts_intern/inverse_intern_math_7b.sh
bash scripts_intern/sample_cot_7b.sh
```
