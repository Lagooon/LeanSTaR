set -e
set -x

export DATA_DIR=/lustre/fast/fast/wliu/longhui/trust_math_ckpt
export MODEL_REPO=EleutherAI/llemma_34b
export OMP_NUM_THREADS=8
export ENABLE_INTRA_NODE_COMM=1

torchrun --standalone --nproc_per_node=8 \
    finetune.py \
    --do_train \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --source_max_len 768 \
    --target_max_len 768 \
    --total_max_len 768 \
    --per_device_train_batch_size 32 \
    --micro_train_batch_size 2 \
    --learning_rate 1e-5 \
    --lr_eta_min 1e-7 \
    --num_train_epochs 3 \
    --dataset "/home/wliu/longhui/llms-all/ScalableMath_sun-main/data/train_1_2_3_prm_v4_mapped.json" \
    --dataset_format "mapped" \
    --add_eos_to_marked_target \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_dir $DATA_DIR/checkpoints/llemma-34b_prm_v4_mapped_1e-5
