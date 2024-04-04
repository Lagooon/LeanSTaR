set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/math
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8

python -u eval_math.py \
    --gt_file "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json" \
    --answer_pattern "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-metamath_1-to-3_s256*.jsonl"
