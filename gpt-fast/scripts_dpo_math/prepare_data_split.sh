set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/math
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8

cd ../

python -u data_scripts/prepare_easy_prm_ppo_split.py \
    --train_output_path ${DATA_DIR}/train_v3_1to3_1-2-3_prm_ppo.json \
    --valid_output_path ${DATA_DIR}/valid_v3_1to5_1-2-3_prm_ppo.json \
    --train_math_path ${DATA_DIR}/prm800k/math_splits/train.jsonl \
    --test_math_path ${DATA_DIR}/prm800k/math_splits/test.jsonl \
    --skip_unavailable True \
    --seed 1234

python -u data_scripts/prepare_easy_prm_ppo_split.py \
    --train_output_path ${DATA_DIR}/train_v3_1to5_1-2-3_prm_ppo.json \
    --valid_output_path ${DATA_DIR}/valid_v3_1to5_1-2-3_prm_ppo.json \
    --train_math_path ${DATA_DIR}/prm800k/math_splits/train.jsonl \
    --test_math_path ${DATA_DIR}/prm800k/math_splits/test.jsonl \
    --skip_unavailable False \
    --seed 1234
