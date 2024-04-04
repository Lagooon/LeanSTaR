set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/math
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=4

# python generate.py --compile --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth --prompt "Hello, my name is"

torchrun --standalone --nproc_per_node=4 \
    generate.py --compile \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --prompt "Hello, my name is"
