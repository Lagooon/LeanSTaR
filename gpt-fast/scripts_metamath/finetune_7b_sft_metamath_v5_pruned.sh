set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/math
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8

torchrun --standalone --nproc_per_node=4 \
    finetune.py \
    --do_train \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --source_max_len 1024 \
    --target_max_len 1024 \
    --total_max_len 1024 \
    --per_device_train_batch_size 32 \
    --micro_train_batch_size 8 \
    --learning_rate 8e-6 \
    --lr_eta_min 5e-10 \
    --num_train_epochs 1 \
    --dataset "/nobackup/users/yikangs/zhiqings/math/train_1to5_metamath_v6_pruned.json" \
    --dataset_format "metamath" \
    --add_eos_to_target \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_dir $DATA_DIR/checkpoints/llemma-7b_metamath_v6_pruned_lr-8e-6_1to5_epoch1_amp \
    --save_only_model True \
    --tensor_parallel_size 1 \
    --weight_decay 0.1 \
    --warmup_ratio 0.02 \
    --param_dtype fp32 \
    --optim_dtype fp32 \
    --report_to "wandb" \
    --wandb_project "scalable-math-sft" \
    --wandb_entity "zhiqings" \
    --wandb_name "llemma-7b_metamath_v6_pruned_lr-8e-6_1to5_epoch1_amp"
