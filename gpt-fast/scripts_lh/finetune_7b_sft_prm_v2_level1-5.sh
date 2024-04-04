set -e
set -x
module load cuda/11.8
export DATA_DIR=/lustre/fast/fast/wliu/longhui/trust_math_ckpt
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8
export HF_HOME=/tmp

torchrun --standalone --nproc_per_node=8 \
    finetune.py \
    --do_train \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --source_max_len 768 \
    --target_max_len 768 \
    --total_max_len 768 \
    --per_device_train_batch_size 16 \
    --micro_train_batch_size 8 \
    --learning_rate 2e-5 \
    --lr_eta_min 2e-7 \
    --num_train_epochs 3 \
    --dataset "/home/wliu/longhui/llms-all/ScalableMath_sun-main/data/train_1_2_3_4_5_prm_v4.json" \
    --dataset_format "prm-v2" \
    --add_eos_to_marked_target \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_dir $DATA_DIR/checkpoints/llemma-7b_prm_2e-5_128_1_2_3_4_5 \
    --report_to "wandb" \
    --wandb_project "scalable-math-sft" \
    --wandb_entity "longhui-yu" \
    --wandb_name "llemma-7b_prm_2e-5_128_1_2_3_4_5" \
