set -e
set -x

export DATA_DIR=/nobackup/users/zhiqings/haohanl/Lean
export MODEL_REPO=internlm/internlm2-math-base-7b
export OMP_NUM_THREADS=8

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
    generate.py \
    --compile \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --finetune_checkpoint_path $DATA_DIR/checkpoints/internlm2-7b_sft_epoch-2_lr-3e-5 \
    --prompt "Tactic state:\n---\n\u03b9\u271d : Type u_1\n\u03b1\u271d : Type u\n\u03b2\u271d : Type v\n\u03b3 : Type w\ninst\u271d\u2075 : PseudoEMetricSpace \u03b1\u271d\ninst\u271d\u2074 : PseudoEMetricSpace \u03b2\u271d\ninst\u271d\u00b3 : PseudoEMetricSpace \u03b3\nf\u271d : \u03b1\u271d \u2192 \u03b2\u271d\nx\u271d y\u271d z : \u03b1\u271d\ns : Set \u03b1\u271d\n\u03b9 : Type u_4\ninst\u271d\u00b2 : Fintype \u03b9\n\u03b1 : \u03b9 \u2192 Type u_2\n\u03b2 : \u03b9 \u2192 Type u_3\ninst\u271d\u00b9 : (i : \u03b9) \u2192 PseudoEMetricSpace (\u03b1 i)\ninst\u271d : (i : \u03b9) \u2192 PseudoEMetricSpace (\u03b2 i)\nf : (i : \u03b9) \u2192 \u03b1 i \u2192 \u03b2 i\nhf : \u2200 (i : \u03b9), Isometry (f i)\nx y : (i : \u03b9) \u2192 \u03b1 i\n\u22a2 edist ((fun g i => f i (g i)) x) ((fun g i => f i (g i)) y) = edist x y\n---\nNext tactic:\n---\n"

