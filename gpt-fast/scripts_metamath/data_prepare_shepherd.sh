set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/math
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8

cd ..

cd data_scripts

python -u prepare_shepherd_zs.py \
    --math_path $DATA_DIR/MATH_train-cleaned_processed.json \
    --shepherd_path $DATA_DIR/math-shepherd.jsonl \
    --metamath_path $DATA_DIR/MetaMathQA-395K.json \
    --output_1to3_path $DATA_DIR/processed_shepherd_v1_level1-3.json \
    --output_1to5_path $DATA_DIR/processed_shepherd_v1_level1-5.json
