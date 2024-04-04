set -e
set -x

# export DATA_DIR=/nobackup/users/yikangs/zhiqings/math
# export MODEL_REPO=EleutherAI/llemma_7b
# export OMP_NUM_THREADS=8
# export ENABLE_INTRA_NODE_COMM=1

# python -u eval_math.py \
#     --mode prepare \
#     --gt_file "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json" \
#     --answer_pattern "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_34b-v4_epoch-3_lr-2e-5_seq-768_s2048.jsonl" \
#     --majority_accs "/nobackup/users/yikangs/zhiqings/math/figs/majority_accs_34b.json" \
#     --decoding_mode "majority"

# python -u eval_math.py \
#     --mode prepare \
#     --gt_file "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json" \
#     --answer_pattern "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_34b-v4_epoch-3_lr-2e-5_seq-768_s2048_scored-toet.jsonl" \
#     --best_of_n_accs "/nobackup/users/yikangs/zhiqings/math/figs/best_of_n_accs_34b_34b.json" \
#     --decoding_mode "best_of_n"

# python -u eval_math.py \
#     --mode prepare \
#     --gt_file "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json" \
#     --answer_pattern "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_34b-v4_epoch-3_lr-2e-5_seq-768_s2048_scored-toet.jsonl" \
#     --weighted_accs "/nobackup/users/yikangs/zhiqings/math/figs/weighted_accs_34b_34b.json" \
#     --decoding_mode "weighted"

# python -u eval_math.py \
#     --mode figure \
#     --name_prefix "SFT-34b + PRM-34b (on PRM800K)" \
#     --filename_prefix "prm800k_34b_34b" \
#     --majority_accs "/nobackup/users/yikangs/zhiqings/math/figs/majority_accs_34b_0.json" \
#     --best_of_n_accs "/nobackup/users/yikangs/zhiqings/math/figs/best_of_n_accs_34b_34b_0.json" \
#     --weighted_accs "/nobackup/users/yikangs/zhiqings/math/figs/weighted_accs_34b_34b_0.json"

# python -u eval_math.py \
#     --mode prepare \
#     --gt_file "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json" \
#     --answer_pattern "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-v4_epoch-3_lr-2e-5_seq-768_s2048.jsonl" \
#     --majority_accs "/nobackup/users/yikangs/zhiqings/math/figs/majority_accs_7b.json" \
#     --decoding_mode "majority"

# python -u eval_math.py \
#     --mode prepare \
#     --gt_file "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json" \
#     --answer_pattern "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-v4_epoch-3_lr-2e-5_seq-768_s2048_scored-toet.jsonl" \
#     --best_of_n_accs "/nobackup/users/yikangs/zhiqings/math/figs/best_of_n_accs_7b_7b.json" \
#     --decoding_mode "best_of_n"

# python -u eval_math.py \
#     --mode prepare \
#     --gt_file "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json" \
#     --answer_pattern "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-v4_epoch-3_lr-2e-5_seq-768_s2048_scored-toet.jsonl" \
#     --weighted_accs "/nobackup/users/yikangs/zhiqings/math/figs/weighted_accs_7b_7b.json" \
#     --decoding_mode "weighted"

# python -u eval_math.py \
#     --mode figure \
#     --name_prefix "SFT-7b + PRM-7b (on PRM800K)" \
#     --filename_prefix "prm800k_7b_7b" \
#     --majority_accs "/nobackup/users/yikangs/zhiqings/math/figs/majority_accs_7b_0.json" \
#     --best_of_n_accs "/nobackup/users/yikangs/zhiqings/math/figs/best_of_n_accs_7b_7b_0.json" \
#     --weighted_accs "/nobackup/users/yikangs/zhiqings/math/figs/weighted_accs_7b_7b_0.json"

# python -u eval_math.py \
#     --mode prepare \
#     --gt_file "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json" \
#     --answer_pattern "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-v4_epoch-3_lr-2e-5_seq-768_s2048_scored-34b-toet.jsonl" \
#     --best_of_n_accs "/nobackup/users/yikangs/zhiqings/math/figs/best_of_n_accs_7b_34b.json" \
#     --decoding_mode "best_of_n"

# python -u eval_math.py \
#     --mode prepare \
#     --gt_file "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json" \
#     --answer_pattern "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-v4_epoch-3_lr-2e-5_seq-768_s2048_scored-34b-toet.jsonl" \
#     --weighted_accs "/nobackup/users/yikangs/zhiqings/math/figs/weighted_accs_7b_34b.json" \
#     --decoding_mode "weighted"

# python -u eval_math.py \
#     --mode figure \
#     --name_prefix "SFT-7b + PRM-34b (on PRM800K)" \
#     --filename_prefix "prm800k_7b_34b" \
#     --majority_accs "/nobackup/users/yikangs/zhiqings/math/figs/majority_accs_7b_0.json" \
#     --best_of_n_accs "/nobackup/users/yikangs/zhiqings/math/figs/best_of_n_accs_7b_34b_0.json" \
#     --weighted_accs "/nobackup/users/yikangs/zhiqings/math/figs/weighted_accs_7b_34b_0.json"

M1="7b"
M2="7b"

# for AGG in "prod"; do
#     python -u eval_math.py \
#         --mode prepare \
#         --gt_file "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json" \
#         --answer_pattern "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_${M1}-v4_epoch-3_lr-2e-5_seq-768_s2048_scored-toet-prwa.jsonl" \
#         --best_of_n_accs "/nobackup/users/yikangs/zhiqings/math/figs/best_of_n_accs_${M1}_${M2}_${AGG}_prwa.json" \
#         --aggregation "$AGG" \
#         --decoding_mode "best_of_n"

#     python -u eval_math.py \
#         --mode prepare \
#         --gt_file "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json" \
#         --answer_pattern "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_${M1}-v4_epoch-3_lr-2e-5_seq-768_s2048_scored-toet-prwa.jsonl" \
#         --weighted_accs "/nobackup/users/yikangs/zhiqings/math/figs/weighted_accs_${M1}_${M2}_${AGG}_prwa.json" \
#         --aggregation "$AGG" \
#         --decoding_mode "weighted"

#     python -u eval_math.py \
#         --mode figure \
#         --name_prefix "SFT-${M1} + PRM-${M2} + $AGG (on PRM800K)" \
#         --output_filename "/nobackup/users/yikangs/zhiqings/math/auto_figs/prm800k_${M1}_${M2}_${AGG}_prwa.pdf" \
#         --majority_accs "/nobackup/users/yikangs/zhiqings/math/figs/majority_accs_${M1}_0.json" \
#         --best_of_n_accs "/nobackup/users/yikangs/zhiqings/math/figs/best_of_n_accs_${M1}_${M2}_${AGG}_prwa_0.json" \
#         --weighted_accs "/nobackup/users/yikangs/zhiqings/math/figs/weighted_accs_${M1}_${M2}_${AGG}_prwa_0.json"
# done

for AGG in "prod"; do
    python -u eval_math.py \
        --mode prepare \
        --gt_file "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json" \
        --answer_pattern "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_${M1}-v4_epoch-3_lr-2e-5_seq-768_s2048_scored-${M2}-porm.jsonl" \
        --best_of_n_accs "/nobackup/users/yikangs/zhiqings/math/figs/best_of_n_accs_${M1}_${M2}_${AGG}_porm.json" \
        --aggregation "$AGG" \
        --decoding_mode "best_of_n"

    python -u eval_math.py \
        --mode prepare \
        --gt_file "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json" \
        --answer_pattern "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_${M1}-v4_epoch-3_lr-2e-5_seq-768_s2048_scored-${M2}-porm.jsonl" \
        --weighted_accs "/nobackup/users/yikangs/zhiqings/math/figs/weighted_accs_${M1}_${M2}_${AGG}_porm.json" \
        --aggregation "$AGG" \
        --decoding_mode "weighted"

    python -u eval_math.py \
        --mode figure \
        --name_prefix "SFT-${M1} + OPRM-${M2} + $AGG (on PRM800K)" \
        --output_filename "/nobackup/users/yikangs/zhiqings/math/auto_figs/prm800k_${M1}_${M2}_${AGG}_porm.pdf" \
        --majority_accs "/nobackup/users/yikangs/zhiqings/math/figs/majority_accs_${M1}_0.json" \
        --best_of_n_accs "/nobackup/users/yikangs/zhiqings/math/figs/best_of_n_accs_${M1}_${M2}_${AGG}_porm_0.json" \
        --weighted_accs "/nobackup/users/yikangs/zhiqings/math/figs/weighted_accs_${M1}_${M2}_${AGG}_porm_0.json"
done

# for AGG in "last"; do
#     python -u eval_math.py \
#         --mode prepare \
#         --gt_file "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json" \
#         --answer_pattern "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_${M1}-v4_epoch-3_lr-2e-5_seq-768_s2048_scored-${M2}-orm.jsonl" \
#         --best_of_n_accs "/nobackup/users/yikangs/zhiqings/math/figs/best_of_n_accs_${M1}_${M2}_${AGG}_orm.json" \
#         --aggregation "$AGG" \
#         --decoding_mode "best_of_n"

#     python -u eval_math.py \
#         --mode prepare \
#         --gt_file "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json" \
#         --answer_pattern "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_${M1}-v4_epoch-3_lr-2e-5_seq-768_s2048_scored-${M2}-orm.jsonl" \
#         --weighted_accs "/nobackup/users/yikangs/zhiqings/math/figs/weighted_accs_${M1}_${M2}_${AGG}_orm.json" \
#         --aggregation "$AGG" \
#         --decoding_mode "weighted"

#     python -u eval_math.py \
#         --mode figure \
#         --name_prefix "SFT-${M1} + ORM-${M2} (on PRM800K)" \
#         --output_filename "/nobackup/users/yikangs/zhiqings/math/auto_figs/prm800k_${M1}_${M2}_${AGG}_orm.pdf" \
#         --majority_accs "/nobackup/users/yikangs/zhiqings/math/figs/majority_accs_${M1}_0.json" \
#         --best_of_n_accs "/nobackup/users/yikangs/zhiqings/math/figs/best_of_n_accs_${M1}_${M2}_${AGG}_orm_0.json" \
#         --weighted_accs "/nobackup/users/yikangs/zhiqings/math/figs/weighted_accs_${M1}_${M2}_${AGG}_orm_0.json"
# done
