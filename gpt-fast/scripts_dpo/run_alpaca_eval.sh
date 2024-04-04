# !/bin/bash
# pip install alpaca-eval

set -e
set -x

ORIGINAL_INPUT=/nobackup/users/yikangs/zhiqings/dpo/alpaca_eval_805.json
REF_OUTPUTS=/nobackup/users/yikangs/zhiqings/dpo/alpaca_eval_reference.json

#  *** Alpaca Farm ***
# Ref - 52k: 39.2 ± 1.7
# Ref - 10k: 36.7 ± 1.7
# Ref - PPO: 46.8 ± 1.8
# Ref - DPO: 46.8 ± 1.7

# default ls=0.0, b (beta)=0.1, 10k rm, 20k unlabeled
#                                *** Ours ***
#                               win_rate  standard_error  n_total  avg_length
# alpaca-7b-10k                   *36.52*           1.68      805         382
# dpo-7b-10k-lr-1e-6               39.32            1.71      805         414
# dpo-7b-10k-lr-2e-6               43.23            1.73      805         449
# dpo-7b-10k-lr-3e-6              *46.40*           1.74      805         489
# dpo-7b-10k-lr-3e-6 (ls=0.1)      44.60            1.74      805         464
# dpo-7b-10k-lr-5e-6               45.16            1.74      805         510
# dpo-7b-10k-lr-5e-6 (ls=0.1)     *46.40*           1.74      805         485
# ipo-7b-10k-lr-2e-6               38.07            1.69      805         395
# ipo-7b-10k-lr-3e-6 (b=0.001)     45.65            1.74      805         559
# ipo-7b-10k-lr-3e-6 (b=0.002)    *45.84*           1.75      805         566
# ipo-7b-10k-lr-3e-6 (b=0.005)     45.16            1.74      805         559
# ipo-7b-10k-lr-3e-6 (b=0.01)      45.47            1.74      805         541
# ipo-7b-10k-lr-3e-6 (b=0.02)      43.60            1.73      805         491
# ipo-7b-10k-lr-3e-6 (b=0.05)      42.11            1.72      805         416
# ipo-7b-10k-lr-3e-6 (b=0.1)       38.82            1.70      805         405
# ipo-7b-10k-lr-3e-6 (b=0.2)       37.76            1.68      805         391
# ipo-7b-10k-lr-5e-6               33.85            1.65      805         454
# ... (ours)
# rpo-7b-10k-lr-3e-6 (b=0.00005)   46.58            1.75      805         548
# rpo-7b-10k-lr-1e-6 (b=0.0001)    47.14            1.75      805         631
#             (bs=16)              47.70            1.75      805         544
# rpo-7b-10k-lr-2e-6 (b=0.0001)    47.14            1.75      805         633
#             (bs=16)              49.25            1.75      805         547
# rpo-7b-10k-lr-3e-6 (b=0.0001)   *49.32*           1.75      805         635
# rpo-7b-10k-lr-5e-6 (b=0.0001)    48.63            1.75      805         636
# rpo-7b-10k-lr-1e-5 (b=0.0001)    48.14            1.75      805         635
# rpo-7b-10k-lr-3e-6 (b=0.0001,v2) 47.64            1.75      805         641
# rpo-7b-10k-lr-3e-6 (b=0.0002)    48.51            1.75      805         532
# ---         (bs=32)              48.76            1.75      805         537
# ---         (bs=32, epoch=2)     44.53            1.74      805         521
# rpo-7b-10k-lr-3e-6 (b=0.0002)    48.76            1.75      805         537
# rpo-7b-10k-lr-3e-6 (b=0.0002,v2) 47.14            1.75      805         556
# rpo-7b-10k-lr-3e-6 (b=0.0002,v3) 46.71            1.75      805         571
# rpo-7b-10k-lr-3e-6 (b=0.001)     47.27            1.75      805         514
# ---         (bs=32, epoch=2)     46.02            1.74      805         486
# rpo-7b-10k-lr-3e-6 (b=0.001,v2) *48.57*           1.75      805         563
# rpo-7b-10k-lr-3e-6 (b=0.001,v3)  46.21            1.75      805         607
# ... (ours, 10k rm + 20k unlabeled)
# rpo-7b-30k-lr-3e-6 (b=0.0002)    44.97            1.75      805         497
# rpo-7b-30k-lr-3e-6 (b=0.001)     44.29            1.74      805         476
# rpo-7b-30k-lr-3e-6 (b=0.005)     42.17            1.73      805         461
# rpo-7b-30k-lr-3e-6 (b=0.0002,v2) 11.61            1.13      805         575
# rpo-7b-30k-lr-3e-6 (b=0.001,v2)  42.36            1.74      805         555

# rpo-v1-7b-aug-10k-lr-3e-6-bs-128-beta-0.0001: 49.57

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_alpaca-7b-10k_epoch-3_lr-1e-4_seq-768_new-300.json
# MODEL_NAME=alpaca-7b-10k-new-300
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_alpaca-7b-10k_epoch-3_lr-1e-4_seq-768_new-300.outputs.json

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_dpo-7b-10k_epoch-2_lr-1e-6_seq-1024.json
# MODEL_NAME=dpo-7b-10k
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_dpo-7b-10k_epoch-2_lr-1e-6_seq-1024.outputs.json

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_dpo-7b-10k_epoch-2_lr-2e-6_seq-1024.json
# MODEL_NAME=dpo-7b-10k-lr-2e-6
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_dpo-7b-10k_epoch-2_lr-2e-6_seq-1024.outputs.json

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_dpo-7b-10k_epoch-2_lr-3e-6_seq-1024.json
# MODEL_NAME=dpo-7b-10k-lr-3e-6
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_dpo-7b-10k_epoch-2_lr-3e-6_seq-1024.outputs.json

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_dpo-7b-10k_epoch-2_lr-5e-6_seq-1024.json
# MODEL_NAME=dpo-7b-10k-lr-5e-6
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_dpo-7b-10k_epoch-2_lr-5e-6_seq-1024.outputs.json

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_dpo-7b-10k_epoch-2_lr-5e-6_seq-1024-p.json
# MODEL_NAME=dpo-7b-10k-lr-5e-6-p
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_dpo-7b-10k_epoch-2_lr-5e-6_seq-1024-p.outputs.json

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_dpo-7b-10k_epoch-2_lr-3e-6_seq-1024-v3p.json
# MODEL_NAME=dpo-7b-10k-lr-3e-6-v3p
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_dpo-7b-10k_epoch-2_lr-3e-6_seq-1024-v3p.outputs.json

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_ipo-7b-10k_epoch-2_lr-5e-6_seq-1024.json
# MODEL_NAME=ipo-7b-10k
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_ipo-7b-10k_epoch-2_lr-5e-6_seq-1024.outputs.json

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_ipo-7b-10k_epoch-2_lr-2e-6_seq-1024.json
# MODEL_NAME=ipo-7b-10k-lr-2e-6
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_ipo-7b-10k_epoch-2_lr-2e-6_seq-1024.outputs.json

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_ipo-7b-10k_epoch-2_lr-3e-6_seq-1024.json
# MODEL_NAME=ipo-7b-10k-lr-3e-6
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_ipo-7b-10k_epoch-2_lr-3e-6_seq-1024.outputs.json

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_ipo-7b-10k_epoch-2_lr-3e-6_seq-1024_beta-0.2.json
# MODEL_NAME=ipo-7b-10k-lr-3e-6-beta-0.2
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_ipo-7b-10k_epoch-2_lr-3e-6_seq-1024_beta-0.2.outputs.json

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_ipo-7b-10k_epoch-2_lr-3e-6_seq-1024_beta-0.05.json
# MODEL_NAME=ipo-7b-10k-lr-3e-6-beta-0.05
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_ipo-7b-10k_epoch-2_lr-3e-6_seq-1024_beta-0.05.outputs.json

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_ipo-7b-10k_epoch-2_lr-3e-6_seq-1024_beta-0.02.json
# MODEL_NAME=ipo-7b-10k-lr-3e-6-beta-0.02
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_ipo-7b-10k_epoch-2_lr-3e-6_seq-1024_beta-0.02.outputs.json

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_ipo-7b-10k_epoch-2_lr-3e-6_seq-1024_beta-0.005.json
# MODEL_NAME=ipo-7b-10k-lr-3e-6-beta-0.005
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_ipo-7b-10k_epoch-2_lr-3e-6_seq-1024_beta-0.005.outputs.json

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_ipo-7b-10k_epoch-2_lr-3e-6_seq-1024_beta-0.01.json
# MODEL_NAME=ipo-7b-10k-lr-3e-6-beta-0.01
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_ipo-7b-10k_epoch-2_lr-3e-6_seq-1024_beta-0.01.outputs.json

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_ipo-7b-10k_epoch-2_lr-3e-6_seq-1024_beta-0.002.json
# MODEL_NAME=ipo-7b-10k-lr-3e-6-beta-0.002
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_ipo-7b-10k_epoch-2_lr-3e-6_seq-1024_beta-0.002.outputs.json

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_ipo-7b-10k_epoch-2_lr-3e-6_seq-1024_beta-0.001.json
# MODEL_NAME=ipo-7b-10k-lr-3e-6-beta-0.001
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_ipo-7b-10k_epoch-2_lr-3e-6_seq-1024_beta-0.001.outputs.json

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_rpo-7b-30k_epoch-1_lr-3e-6_seq-1024_beta-0.001-v2.json
# MODEL_NAME=rpo-7b-30k-lr-3e-6-beta-0.001-v2
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_rpo-7b-30k_epoch-1_lr-3e-6_seq-1024_beta-0.001-v2.outputs.json

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_rpo-7b-30k_epoch-1_lr-3e-6_seq-1024_beta-0.005.json
# MODEL_NAME=rpo-7b-30k-lr-3e-6-beta-0.005
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_rpo-7b-30k_epoch-1_lr-3e-6_seq-1024_beta-0.005.outputs.json

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_rpo-7b-30k_epoch-1_lr-3e-6_seq-1024_beta-0.001.json
# MODEL_NAME=rpo-7b-30k-lr-3e-6-beta-0.001
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_rpo-7b-30k_epoch-1_lr-3e-6_seq-1024_beta-0.001.outputs.json

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_rpo-7b-30k_epoch-1_lr-3e-6_seq-1024_beta-0.001-v4.json
# MODEL_NAME=rpo-7b-30k-lr-3e-6-beta-0.001-v4
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_rpo-7b-30k_epoch-1_lr-3e-6_seq-1024_beta-0.001-v4.outputs.json

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_rpo-7b-30k_epoch-1_lr-3e-6_seq-1024_beta-0.001-v3.json
# MODEL_NAME=rpo-7b-30k-lr-3e-6-beta-0.001-v3
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_rpo-7b-30k_epoch-1_lr-3e-6_seq-1024_beta-0.001-v3.outputs.json

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_rpo-7b-30k_epoch-1_lr-3e-6_seq-1024_beta-0.0002.json
# MODEL_NAME=rpo-7b-30k-lr-3e-6-beta-0.0002
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_rpo-7b-30k_epoch-1_lr-3e-6_seq-1024_beta-0.0002.outputs.json

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_rpo-7b-30k_epoch-1_lr-3e-6_seq-1024_beta-0.0002-v2.json
# MODEL_NAME=rpo-7b-30k-lr-3e-6-beta-0.0002-v2
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_rpo-7b-30k_epoch-1_lr-3e-6_seq-1024_beta-0.0002-v2.outputs.json

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_rpo-7b-30k_epoch-1_lr-3e-6_seq-1024_beta-0.0002-v4.json
# MODEL_NAME=rpo-7b-30k-lr-3e-6-beta-0.0002-v4
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_rpo-7b-30k_epoch-1_lr-3e-6_seq-1024_beta-0.0002-v4.outputs.json

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_rpo-7b-30k_epoch-1_lr-3e-6_seq-1024_beta-0.0002-v3.json
# MODEL_NAME=rpo-7b-30k-lr-3e-6-beta-0.0002-v3
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_rpo-7b-30k_epoch-1_lr-3e-6_seq-1024_beta-0.0002-v3.outputs.json

# FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_rpo-7b-30k_epoch-1_lr-3e-6_seq-1024_beta-0.001-v6.json
# MODEL_NAME=rpo-7b-30k-lr-3e-6-beta-0.001-v6
# OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_rpo-7b-30k_epoch-1_lr-3e-6_seq-1024_beta-0.001-v6.outputs.json

FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_rpo-7b-30k_epoch-1_lr-3e-6_seq-1024_beta-0.0002-v6.json
MODEL_NAME=rpo-7b-30k-lr-3e-6-beta-0.0002-v6
OUTPUT_FILE_NAME=/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_rpo-7b-30k_epoch-1_lr-3e-6_seq-1024_beta-0.0002-v6.outputs.json

if [ -z "$FILE_NAME" ]; then
    echo "FILE_NAME is not defined"
    exit 1
fi

python -u scripts_dpo/sort_outputs.py \
    --original_input_file $ORIGINAL_INPUT \
    --input_file $FILE_NAME \
    --output_file $OUTPUT_FILE_NAME \
    --model_name $MODEL_NAME

alpaca_eval \
    --model_outputs $OUTPUT_FILE_NAME \
    --reference_outputs $REF_OUTPUTS \
    --annotators_config 'alpaca_farm_greedy_gpt4' \
    --precomputed_leaderboard None
