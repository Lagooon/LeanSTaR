set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/dpo
export MODEL_REPO=huggyllama/llama-7b
export OMP_NUM_THREADS=8
# export NCCL_IB_TIMEOUT=22
# export ENABLE_INTRA_NODE_COMM=1
# export NCCL_CROSS_NIC=1
# export CUDA_LAUNCH_BLOCKING=1

# per_device_train_batch_size = accumulate_steps * micro_batch_size

EPOCH=1
LR=3e-6
BS=64
BETA=0.0002
VARIANT=rpo-cringe
PREV_SAVE_FILE_NAME=alpaca-7b-rpo_cringe_epoch-2_lr-${LR}_bs-32_beta-${BETA}-v2
SAVE_FILE_NAME=alpaca-7b-rpo_cringe_epoch-${EPOCH}_lr-${LR}_bs-${BS}_beta-${BETA}-v2-v6-iter2

torchrun --standalone --nproc_per_node=4 \
    finetune_dpo.py \
    --do_train \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --sft_checkpoint_path $DATA_DIR/checkpoints/$PREV_SAVE_FILE_NAME \
    --source_max_len 512 \
    --target_max_len 512 \
    --total_max_len 512 \
    --dpo_pm_total_max_len 1024 \
    --per_device_train_batch_size $BS \
    --micro_train_batch_size 8 \
    --learning_rate ${LR} \
    --lr_eta_min 3e-8 \
    --warmup_ratio 0.03 \
    --num_train_epochs ${EPOCH} \
    --dataset "/nobackup/users/yikangs/zhiqings/dpo/alpaca_rm_mixlabeled_rpo-v2-v6_30k.json" \
    --dataset_format "alpaca" \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_dir $DATA_DIR/checkpoints/$SAVE_FILE_NAME \
    --resume_from_checkpoint \
    --report_to "wandb" \
    --wandb_project "scalable-alpacafarm" \
    --wandb_entity "zhiqings" \
    --wandb_name "$SAVE_FILE_NAME" \
    --param_dtype fp32 \
    --optim_dtype fp32 \
    --optimizer_cpu_offload True \
    --tensor_parallel_size 1 \
    --print_training_examples True \
    --save_only_model True \
    --adam_beta2 0.95 \
    --adam_eps 1e-5 \
    --add_eos_to_target True \
    --dpo_beta $BETA \
    --dpo_variant $VARIANT || echo "done"

torchrun --standalone --nproc_per_node=4 \
    batched_generate.py \
    --compile \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --finetune_checkpoint_path $DATA_DIR/checkpoints/$SAVE_FILE_NAME \
    --batch_size 32 \
    --max_new_tokens 300 \
    --num_samples 1 \
    --prompt_file "/nobackup/users/yikangs/zhiqings/dpo/alpaca_eval_805.json" \
    --temperature 0.7 \
    --top_k 50 \
    --output_file "/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_$SAVE_FILE_NAME.json" \
    --tensor_parallel_size 1 || echo "done"

ORIGINAL_INPUT=/nobackup/users/yikangs/zhiqings/dpo/alpaca_eval_805.json
REF_OUTPUTS=/nobackup/users/yikangs/zhiqings/dpo/alpaca_eval_reference.json
FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_$SAVE_FILE_NAME.json
MODEL_NAME=$VARIANT-7b-10k-lr-${LR}-bs-${BS}-beta-${BETA}-epoch-${EPOCH}-iter2
OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_$SAVE_FILE_NAME.outputs.json

if [ -z "$FILE_NAME" ]; then
    echo "FILE_NAME is not defined"
    exit 1
fi

python -u scripts_dpo/sort_outputs.py \
    --original_input_file $ORIGINAL_INPUT \
    --input_file $FILE_NAME \
    --output_file $OUTPUT_FILE_NAME \
    --model_name $MODEL_NAME

echo $OUTPUT_FILE_NAME
echo $REF_OUTPUTS

# alpaca_eval \
#     --model_outputs $OUTPUT_FILE_NAME \
#     --reference_outputs $REF_OUTPUTS \
#     --annotators_config 'alpaca_farm_greedy_gpt4' \
#     --precomputed_leaderboard None
