set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/math
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8
# export ENABLE_INTRA_NODE_COMM=1

SEED=44
SFT_CONFIG="1to3"
DATA_CONFIG="1-to-3"
DATA_REPEAT=3
NUM_GPUS=4
NUM_SAMPLES=8
TEMPERATURE=1.2
BATCH_SIZE=64
BETA=0.1
LEARNING_RATE=2e-6

# TRAIN_PROMPT_FILE=/nobackup/users/yikangs/zhiqings/math/train_1to3_1-2-3_prm_ppo.json
TRAIN_PROMPT_FILE=/nobackup/users/yikangs/zhiqings/math/train_v3_1to3_1-2-3_prm_ppo.json
VALID_PROMPT_FILE=/nobackup/users/yikangs/zhiqings/math/valid_v3_1to5_1-2-3_prm_ppo.json
TEST_PROMPT_FILE=/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json
# SFT_CKPT=llemma-7b_metamath_v6_pruned_lr-8e-6_1to3_epoch1_amp
SFT_CKPT=llemma-7b_metamath_v3_5e-6_128_1_2_3_tp1_lr_min_5e-10_320acc
RM_CKPT=llemma-7b-rm_toet_shepherd-1-2-3-v1_epoch-2_lr-2e-5_seq-768
VERSION=v4

# /nobackup/users/yikangs/zhiqings/math/outputs/train_1to3_metamath_seed-1234_1-to-3_dup-3_lr-3e-6_beta-0.1_r*_s8_t*.jsonl
# /nobackup/users/yikangs/zhiqings/math/outputs/train_1to3_metamath_seed-1234_1-to-3_dup-3_lr-3e-6_beta-0.1_r2_s8_t*.jsonl
# /nobackup/users/yikangs/zhiqings/math/outputs/train_1to3_metamath_seed-1234_1-to-3_dup-3_lr-3e-6_beta-0.1_r3_s8_t*.jsonl
# /nobackup/users/yikangs/zhiqings/math/outputs/train_1to3_metamath_seed-1234_1-to-3_dup-3_lr-3e-6_beta-0.1_r4_s8_t*.jsonl
# /nobackup/users/yikangs/zhiqings/math/outputs/train_1to3_metamath_seed-1234_1-to-3_dup-3_lr-3e-6_beta-0.1_r*_s8_t*.json
# /nobackup/users/yikangs/zhiqings/math/checkpoints/llemma-7b_dpo-metamath_1to3_v3_seed-4*_1-to-3_dup-3_lr-*e-6_beta-0.1_r*_s8_t*
# /nobackup/users/yikangs/zhiqings/math/checkpoints/llemma-7b_dpo-metamath_1to3_\{v3\}_seed-1234_1-to-3_dup-3_lr-3e-6_beta-0.1_r*_s8_t*
# /nobackup/users/yikangs/zhiqings/math/checkpoints/llemma-7b_dpo-metamath_1to3_v6_seed-1234_1-to-3_dup-3_lr-3e-6_beta-0.1_r*_s8_t*
# /nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_dpo-metamath_seed-1234_1-to-3_dup-3_lr-3e-6_beta-0.1_r*_s8_t*_g.jsonl

for ROUND in {1..8}; do
    if [ $ROUND -eq 1 ]; then
        LOAD_CKPT=$SFT_CKPT
    else
        LAST_ROUND=$((ROUND - 1))
        LOAD_CKPT=llemma-7b_dpo-metamath_${SFT_CONFIG}_${VERSION}_seed-${SEED}_${DATA_CONFIG}_dup-${DATA_REPEAT}_lr-${LEARNING_RATE}_beta-${BETA}_r${LAST_ROUND}_s${NUM_SAMPLES}_t${TEMPERATURE}
    fi

    NAME_SIGNATURE=${VERSION}_seed-${SEED}_${DATA_CONFIG}_dup-${DATA_REPEAT}_lr-${LEARNING_RATE}_beta-${BETA}_r${ROUND}_s${NUM_SAMPLES}_t${TEMPERATURE}
    NEW_CKPT=llemma-7b_dpo-metamath_${SFT_CONFIG}_${NAME_SIGNATURE}

    TRAIN_SAMPLE_FILE=train_${SFT_CONFIG}_metamath_${NAME_SIGNATURE}.jsonl
    TRAIN_SCORED_FILE=train_${SFT_CONFIG}_metamath_${NAME_SIGNATURE}_scored.jsonl
    DPO_TRAIN_DATA=train_${SFT_CONFIG}_metamath_${NAME_SIGNATURE}_dpo-toet.json

    # test is always on level 1-5
    TEST_OUTPUT=test_1to5_dpo-metamath_${NAME_SIGNATURE}_g.jsonl
    VALID_OUTPUT=valid_1to5_dpo-metamath_${NAME_SIGNATURE}_g.jsonl

    # check if the output file not exist
    FULL_TRAIN_SAMPLE_FILE="/nobackup/users/yikangs/zhiqings/math/outputs/$TRAIN_SAMPLE_FILE"
    if [ ! -f "$FULL_TRAIN_SAMPLE_FILE" ]; then
        echo "File $DATA_DIR/outputs/$TEST_OUTPUT not exists."
        torchrun --standalone --nproc_per_node=${NUM_GPUS} \
            batched_generate.py \
            --default_compile \
            --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
            --finetune_checkpoint_path "$DATA_DIR/checkpoints/$LOAD_CKPT" \
            --batch_size 32 \
            --max_new_tokens 1024 \
            --num_samples ${NUM_SAMPLES} \
            --prompt_file ${TRAIN_PROMPT_FILE} \
            --temperature ${TEMPERATURE} \
            --top_k 50 \
            --output_file "$FULL_TRAIN_SAMPLE_FILE" \
            --tensor_parallel_size 1 \
            --resume_generation \
            --seed $SEED
    else
        echo "File $FULL_TRAIN_SAMPLE_FILE exists."
        echo ""
    fi

    FULL_TRAIN_SCORED_FILE="/nobackup/users/yikangs/zhiqings/math/outputs/$TRAIN_SCORED_FILE"
    if [ ! -f "$FULL_TRAIN_SCORED_FILE" ]; then
        echo "File $DATA_DIR/outputs/$TRAIN_SCORED_FILE not exists."
        torchrun --standalone --nproc_per_node=${NUM_GPUS} \
            batched_rm_score.py \
            --compile \
            --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
            --finetune_checkpoint_path "$DATA_DIR/checkpoints/$RM_CKPT" \
            --batch_size 128 \
            --prompt_file "$FULL_TRAIN_SAMPLE_FILE" \
            --output_file "$FULL_TRAIN_SCORED_FILE" \
            --tensor_parallel_size 1 \
            --resume_generation
    else
        echo "File $FULL_TRAIN_SCORED_FILE exists."
        echo ""
    fi

    FULL_DPO_DATA="/nobackup/users/yikangs/zhiqings/math/outputs/$DPO_TRAIN_DATA"
    if [ ! -f "$FULL_DPO_DATA" ]; then
        echo "File $FULL_DPO_DATA not exists."
        python -u eval_math_dpo.py \
            --gt_file ${TRAIN_PROMPT_FILE} \
            --answer_pattern "$FULL_TRAIN_SCORED_FILE" \
            --save_to_file "$FULL_DPO_DATA" \
            --negative_duplicate $DATA_REPEAT
    else
        echo "File $FULL_DPO_DATA exists."
        echo ""
    fi

    # per_device_batch_size = ${BATCH_SIZE} // ${NUM_GPUS}
    PER_DEVICE_TRAIN_BATCH_SIZE=$((${BATCH_SIZE} / ${NUM_GPUS}))
    echo "Per device train batch size: $PER_DEVICE_TRAIN_BATCH_SIZE"
    echo ""

    # let's check folder $DATA_DIR/checkpoints/$NEW_CKPT
    if [ ! -f "$DATA_DIR/checkpoints/$NEW_CKPT/last_checkpoint" ]; then
        echo "File $DATA_DIR/checkpoints/$NEW_CKPT/last_checkpoint not exists."
        torchrun --standalone --nproc_per_node=${NUM_GPUS} \
            finetune_dpo.py \
            --do_train \
            --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
            --sft_checkpoint_path "$DATA_DIR/checkpoints/$LOAD_CKPT" \
            --source_max_len 2048 \
            --target_max_len 2048 \
            --total_max_len 2048 \
            --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
            --micro_train_batch_size 1 \
            --learning_rate $LEARNING_RATE \
            --lr_eta_min 3e-8 \
            --warmup_ratio 0.03 \
            --num_train_epochs 1 \
            --dataset "$FULL_DPO_DATA" \
            --save_strategy epoch \
            --save_total_limit 1 \
            --dpo_beta ${BETA} \
            --save_dir $DATA_DIR/checkpoints/$NEW_CKPT \
            --report_to "wandb" \
            --wandb_project "scalable-math-dpo" \
            --wandb_entity "zhiqings" \
            --wandb_name "$NEW_CKPT" \
            --param_dtype fp32 \
            --optim_dtype fp32 \
            --optimizer_cpu_offload True \
            --tensor_parallel_size 1 \
            --print_training_examples True \
            --save_only_model True \
            --adam_beta2 0.95 \
            --adam_eps 1e-5 \
            --add_eos_to_marked_target True \
            --seed $SEED
    else
        echo "Folder $DATA_DIR/checkpoints/$NEW_CKPT exists."
        echo ""
    fi

    FULL_TEST_OUTPUT="/nobackup/users/yikangs/zhiqings/math/outputs/$TEST_OUTPUT"

    if [ ! -f "$FULL_TEST_OUTPUT" ]; then
        echo "File $FULL_TEST_OUTPUT not exists."
        torchrun --standalone --nproc_per_node=${NUM_GPUS} \
            batched_generate.py \
            --default_compile \
            --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
            --finetune_checkpoint_path "$DATA_DIR/checkpoints/$NEW_CKPT" \
            --batch_size 32 \
            --max_new_tokens 1024 \
            --num_samples 1 \
            --prompt_file ${TEST_PROMPT_FILE} \
            --temperature 0.0 \
            --top_k 1 \
            --output_file "$FULL_TEST_OUTPUT" \
            --tensor_parallel_size 1 \
            --resume_generation
    else
        echo "File $FULL_TEST_OUTPUT exists."
        echo ""
    fi

    FULL_VALID_OUTPUT="/nobackup/users/yikangs/zhiqings/math/outputs/$VALID_OUTPUT"

    if [ ! -f "$FULL_VALID_OUTPUT" ]; then
        echo "File $FULL_VALID_OUTPUT not exists."
        torchrun --standalone --nproc_per_node=${NUM_GPUS} \
            batched_generate.py \
            --default_compile \
            --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
            --finetune_checkpoint_path "$DATA_DIR/checkpoints/$NEW_CKPT" \
            --batch_size 32 \
            --max_new_tokens 1024 \
            --num_samples 1 \
            --prompt_file ${VALID_PROMPT_FILE} \
            --temperature 0.0 \
            --top_k 1 \
            --output_file "$FULL_VALID_OUTPUT" \
            --tensor_parallel_size 1 \
            --resume_generation
    else
        echo "File $FULL_VALID_OUTPUT exists."
        echo ""
    fi

    # python -u eval_math_dpo.py \
    #     --gt_file "/nobackup/users/yikangs/zhiqings/math/train_1to3_1-2-3_prm_ppo.json" \
    #     --answer_pattern "/nobackup/users/yikangs/zhiqings/math/outputs/$TRAIN_PREFIX"
done
