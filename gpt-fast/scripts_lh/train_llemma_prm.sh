# MODEL_PATH='EleutherAI/llemma_7b'
# SAVE_PATH=''
# export master_port=1231
# RANK=$1
# export local_rank=${RANK}

# export MASTER_ADDR="localhost"
# export MASTER_PORT="1231"
# export GLOO_SOCKET_IFNAME="lo"
# export NCCL_SOCKET_IFNAME="lo"
# export WANDB_DISABLED=true
# wandb disabled

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=8 --use_env train_llemma.py \
#     --model_name_or_path $MODEL_PATH \
#     --data_path "metamath_json_file" \
#     --data_length 10000000 \
#     --data_type gsm8k \
#     --bf16 True \
#     --output_dir $SAVE_PATH \
#     --num_train_epochs 1 \
#     --model_max_length 768 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 100 \
#     --save_total_limit 2 \
#     --learning_rate 5e-6 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --deepspeed ./zero2_config_30b.json



export MODEL_PATH='/lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/llemma-7b_metamath_v3_5e-6_1_2_3_prm'
export SAVE_PATH="/lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/llemma-7b_metamath_v3_5e-6_1_2_3_prm"
export MASTER_ADDR="localhost"
export MASTER_PORT="1231"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
module load cuda/11.8
export WANDB_DISABLED=true
wandb disabled
rm -rf /lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/llemma-7b_metamath_v3_5e-6_1_2_3_prm_bin
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=8 --use_env train_llemma_prm.py \
    --model_name_or_path $MODEL_PATH \
    --data_path "/home/wliu/longhui/llms-all/ScalableMath_sun-main/data/train_1_2_3_prm_v4_mapped.json" \
    --data_length 100000000 \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 1 \
    --model_max_length 768 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 0 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True

# # /lustre/fast/fast/wliu/longhui/inverse_ckpt/metamath_llama-7b-ablation_bootstrap_120000
# python eval_gsm8k.py --model $SAVE_PATH --data_file ./data/test/GSM8K_test.jsonl \
# --tensor_parallel_size 1
# # python eval_math.py --model $SAVE_PATH --data_path ./data/test/MATH_test.jsonl
# # conda create -n metamath python=3.10 -c conda-forge -y