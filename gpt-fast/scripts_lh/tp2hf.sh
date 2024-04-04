python concat_tp_model_7b.py \
    --tp_ckpt_name '/lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/llemma_7b_1e-5_metamath_mapped/epoch_1_step_58_rank_' \
    --pretrain_name '/lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/EleutherAI/llemma_7b' \
    --save_name_hf '/lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/llemma-7b_metamath-hf'

# python concat_tp_model_34b.py \
#     --tp_ckpt_name '/lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/llemma_34b_5e-6_metamath_mapped/epoch_2_step_460_rank_' \
#     --pretrain_name '/lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/EleutherAI/llemma_34b' \
#     --save_name_hf '/lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/llemma-34b_metamath-hf'



