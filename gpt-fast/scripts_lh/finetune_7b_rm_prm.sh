set -e
set -x
module load cuda/11.8
export DATA_DIR=/lustre/fast/fast/wliu/longhui/trust_math_ckpt
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8
# export ENABLE_INTRA_NODE_COMM=1
export HF_HOME=/tmp

torchrun --standalone --nproc_per_node=4 \
    finetune_rm.py \
    --do_train \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --sft_checkpoint_path $DATA_DIR/checkpoints/llemma-7b_metamath_v3_5e-6_1_2_3_tp_288 \
    --source_max_len 768 \
    --target_max_len 768 \
    --total_max_len 768 \
    --per_device_train_batch_size 32 \
    --micro_train_batch_size 16 \
    --learning_rate 2e-5 \
    --lr_eta_min 2e-7 \
    --num_train_epochs 1 \
    --dataset "/home/wliu/longhui/llms-all/ScalableMath_sun-main/data/train_1_2_3_prm_v4_rm.json" \
    --dataset_format "prm-v3" \
    --save_strategy epoch \
    --save_total_limit 1 \
    --train_on_every_token \
    --save_dir $DATA_DIR/checkpoints/llemma-7b_metamath_v3_5e-6_1_2_3_tp_288_epoch-1_lr-2e-5_seq-768 \
    --resume_from_checkpoint
