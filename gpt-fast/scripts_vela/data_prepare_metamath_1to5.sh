set -e
set -x

export DATA_DIR=/workspace/zhiqings/output4/math
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8

cd ..

cd data_scripts

python -u prepare_metamath_zs.py \
    --math_path $DATA_DIR/MATH_train-cleaned_processed.json \
    --metamath_path $DATA_DIR/MetaMathQA-395K.json \
    --levels "Level 1, Level 2, Level 3, Level 4, Level 5" \
    --pruned_numbers 4 \
    --pruned_output_path $DATA_DIR/train_1to5_metamath_v4_pruned.json \
    --output_path $DATA_DIR/train_1to5_metamath_v4.json
