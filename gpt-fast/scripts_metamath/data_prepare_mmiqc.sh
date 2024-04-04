set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/math
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8

cd ..

cd data_scripts

# python -u prepare_mmiqc_zs.py \
#     --math_path $DATA_DIR/MATH_train-cleaned_processed.json \
#     --metamath_path $DATA_DIR/MetaMathQA-395K.json \
#     --levels "Level 1, Level 2, Level 3" \
#     --pruned_numbers 3 \
#     --epoch 3 \
#     --pruned_output_path $DATA_DIR/train_1to3_mmiqc_v6_pruned.json \
#     --output_path $DATA_DIR/train_1to3_mmiqc_v6.json \
#     --print_examples

python -u prepare_mmiqc_zs.py \
    --math_path $DATA_DIR/MATH_train-cleaned_processed.json \
    --metamath_path $DATA_DIR/MetaMathQA-395K.json \
    --levels "Level 1, Level 2, Level 3, Level 4, Level 5" \
    --pruned_numbers 3 \
    --epoch 3 \
    --pruned_output_path $DATA_DIR/train_1to5_mmiqc_v6_pruned.json \
    --output_path $DATA_DIR/train_1to5_mmiqc_v6.json \
    --print_examples
