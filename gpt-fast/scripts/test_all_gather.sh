set -e
set -x

export OMP_NUM_THREADS=1
# /nobackup/users/yikangs/software/miniconda3_x86/envs/math/lib/python3.11/site-packages/transformers/utils/generic.py:441: UserWarning: torch.utils
# _pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.

torchrun --standalone --nproc_per_node=8 \
    test_all_gather.py \
    --use_zero True \
    --mode all_devices 2>&1 | grep -v "TORCH_NCCL_ASYNC_ERROR_HANDLING"

# Max memory allocated after creating DDP: 656.0MB
# Max memory allocated before optimizer step(): 1049.0MB
# Max memory allocated after optimizer step(): 1199.0MB

torchrun --standalone --nproc_per_node=8 \
    test_all_gather.py \
    --use_zero True \
    --mode intra_node 2>&1 | grep -v "TORCH_NCCL_ASYNC_ERROR_HANDLING"

# Max memory allocated after creating DDP: 656.0MB
# Max memory allocated before optimizer step(): 1049.0MB
# Max memory allocated after optimizer step(): 1300.0MB

torchrun --standalone --nproc_per_node=8 \
    test_all_gather.py \
    --use_zero True \
    --mode inter_node 2>&1 | grep -v "TORCH_NCCL_ASYNC_ERROR_HANDLING"

# Max memory allocated after creating DDP: 656.0MB
# Max memory allocated before optimizer step(): 1049.0MB
# Max memory allocated after optimizer step(): 1552.0MB

torchrun --standalone --nproc_per_node=8 \
    test_all_gather.py \
    --use_zero False \
    --mode all_devices 2>&1 | grep -v "TORCH_NCCL_ASYNC_ERROR_HANDLING"

# Max memory allocated after creating local model: 335.0MB
# Max memory allocated after creating DDP: 656.0MB
# Max memory allocated before optimizer step(): 1049.0MB
# Max memory allocated after optimizer step(): 2055.0MB

torchrun --standalone --nproc_per_node=8 \
    test_all_gather.py \
    --use_zero False \
    --mode intra_node 2>&1 | grep -v "TORCH_NCCL_ASYNC_ERROR_HANDLING"

# Max memory allocated after creating DDP: 656.0MB
# Max memory allocated before optimizer step(): 1049.0MB
# Max memory allocated after optimizer step(): 2055.0MB

torchrun --standalone --nproc_per_node=8 \
    test_all_gather.py \
    --use_zero False \
    --mode inter_node 2>&1 | grep -v "TORCH_NCCL_ASYNC_ERROR_HANDLING"

# Max memory allocated after creating DDP: 656.0MB
# Max memory allocated before optimizer step(): 1049.0MB
# Max memory allocated after optimizer step(): 2055.0MB
