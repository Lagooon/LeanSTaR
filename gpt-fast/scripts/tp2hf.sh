python concat_tp_model_7b.py \
    --tp_ckpt_name '/lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/metamath_llama-7b-395k-1-2-3_epoch-1_lr-3e-5_bs-128_seq-768/epoch_1_step_58_rank_' \
    --pretrain_name '/lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/metamath_llama-7b-395k' \
    --save_name_hf '/lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/metamath_llama-7b-395k_hf'

python concat_tp_model_34b.py \
    --tp_ckpt_name '/lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/llemma-34b_metamath-merge-debug/epoch_1_step_230_rank_' \
    --pretrain_name '/lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/EleutherAI/llemma_34b' \
    --save_name_hf '/lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/llemma-34b_hf'



