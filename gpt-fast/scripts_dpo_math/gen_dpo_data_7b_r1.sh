set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/math
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8
# export ENABLE_INTRA_NODE_COMM=1

DUP=5

python -u eval_math_dpo.py \
    --gt_file "/nobackup/users/yikangs/zhiqings/math/train_1to5_1-2-3_prm_ppo.json" \
    --answer_pattern "/nobackup/users/yikangs/zhiqings/math/outputs/train_1to5_prm_7b-v4_epoch-3_lr-2e-5_seq-768_r1_s8_scored-toet.jsonl" \
    --save_to_file "/nobackup/users/yikangs/zhiqings/math/outputs/train_1to5_prm_7b-v4_epoch-3_lr-2e-5_seq-768_r1_s8_dpo-toet_dup-$DUP.json" \
    --negative_duplicate $DUP

# --answer_pattern "/nobackup/users/yikangs/zhiqings/math/outputs/train_1to5_prm_7b-v4_epoch-3_lr-2e-5_seq-768_r1_s8*.jsonl"

# r1
# 0.273583333333333356 (SC)
# 0.269 (BoN)
# 0.2971666666666667 (WSC)
