set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/math
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8

LEARNING_RATE=1e-5
KL_COEF=0.001
EPOCH=50
NOPTEPOCHS=1
ROLLOUT_BATCH_SIZE=1024
STEP_BATCH_SZIE=256
ROLLOUT_PER_DEVICE_BATCH_SIZE=64
REWARD_MODEL_PER_DEVICE_BATCH_SIZE=16
STEP_PER_DEVICE_BATCH_SIZE=2

SAMPLING_TEMPARATURE=0.7
UNFINISHED_REWARD_PENALTY=0.0

torchrun --standalone --nproc_per_node=4 \
    finetune_ppo.py \
    --compile \
    --do_train \
    --base_checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --policy_checkpoint_path $DATA_DIR/checkpoints/llemma-7b_prm-1-2-3-v4_epoch-3_lr-2e-5_seq-768 \
    --reward_checkpoint_path $DATA_DIR/checkpoints/llemma-7b-rm_sft-init-toet_prm-1-2-3-v4_epoch-1_lr-2e-5_seq-768 \
    --source_max_len 256 \
    --target_max_len 768 \
    --step_batch_size $STEP_BATCH_SZIE \
    --step_per_device_batch_size $STEP_PER_DEVICE_BATCH_SIZE \
    --rollout_batch_size $ROLLOUT_BATCH_SIZE \
    --rollout_per_device_batch_size $ROLLOUT_PER_DEVICE_BATCH_SIZE \
    --reward_model_per_device_batch_size $REWARD_MODEL_PER_DEVICE_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --init_value_with_reward True \
    --noptepochs $NOPTEPOCHS \
    --ppo_warmup_steps 5 \
    --seed 42 \
    --dataset "/nobackup/users/yikangs/zhiqings/math/train_1to5_1-2-3_prm_ppo.json" \
    --save_strategy steps \
    --save_steps 5 \
    --save_total_limit 1 \
    --save_dir $DATA_DIR/checkpoints/llemma-7b-ppo_prm-1to5_epoch-10_lr-1e-5_seq-768-v2 \
    --resume_from_checkpoint \
    --stop_token "\n\n# Answer\n\n" \
    --penalize_no_stop_token True \
    --relative_stop_token_penalty True \
    --penalty_reward_value $UNFINISHED_REWARD_PENALTY \
    --kl_coef $KL_COEF \
    --max_grad_norm 1.0 \
    --whitening_async_stats "full_batch" \
    --whiten_rewards True \
    --temperature $SAMPLING_TEMPARATURE \
    --num_train_epochs $EPOCH \
    --ref_policy_model_bits 8 \
    --reward_model_bits 8 \
    --report_to "wandb" \
    --wandb_name "llemma-7b-ppo_prm-1to5_epoch-10_lr-1e-5_seq-768-vanilla-v2" \
    --wandb_project "scalable-math" \
    --wandb_entity "zhiqings" \
    --policy_optimizer_cpu_offload True \
    --value_optimizer_cpu_offload True \
    --outcome_reward True
