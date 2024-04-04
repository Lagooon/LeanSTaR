set -e
set -x

# python -u eval_math.py \
#     --mode prepare \
#     --gt_file "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json" \
#     --answer_pattern "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_34b-metamath-8e-6_v6_amp_1-to-3_s256.jsonl" \
#     --majority_accs "/tmp/tmp.json" \
#     --decoding_mode "majority"

python -u eval_math.py \
    --mode prepare \
    --gt_file "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json" \
    --answer_pattern "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_34b-metamath-8e-6_v6_amp_1-to-5_s256.jsonl" \
    --majority_accs "/tmp/tmp.json" \
    --decoding_mode "majority"
