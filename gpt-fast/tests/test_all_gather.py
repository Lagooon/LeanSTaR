# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import fire
from typing import Optional, List, Tuple

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed import _functional_collectives as funcol
from torch.distributed.algorithms._comm_hooks import default_hooks
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP


def _get_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_local():
    return _get_rank() == 0


def local_break():
    if is_local():
        breakpoint()
    dist.barrier()


def _get_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))


def maybe_init_dist() -> Optional[int]:
    try:
        # provided by torchrun
        rank = _get_rank()
        world_size = _get_world_size()

        if world_size < 2:
            # too few gpus to parallelize, tp is no-op
            return None
    except KeyError:
        # not run via torchrun, no-op
        return None

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank


def _init_intra_and_inter_node_groups(
    global_process_group: dist.ProcessGroup,
    num_devices_per_node: int,
) -> Tuple[dist.ProcessGroup, dist.ProcessGroup]:
    """
    Initialize intra and inter-node process groups and return the ones corresponding to this process's rank.

    This function can be used to initialize process groups for ``HYBRID_SHARD`` or
    ``_HYBRID_SHARD_ZERO2`` in FSDP.
    This function assumes each node has an equal number of CUDA-enabled devices.
    Returns:
        Tuple[dist.ProcessGroup, dist.ProcessGroup]: Intra and inter-node process group.
    """
    return (
        _init_intra_node_process_group(num_devices_per_node),
        _init_inter_node_process_group(global_process_group, num_devices_per_node),
    )


def _init_intra_node_process_group(num_devices_per_node: int) -> dist.ProcessGroup:
    """
    Return a process group across the current node.

    For example, given each row is a distinct node:
    0 1 2 3 4 5 6 7 8
    9 10 11 12 13 14 15
    This API would return an intra-node subgroup across
    [0, 7] or [8, 15] depending on the process's rank.
    For example, rank 3 would get [0, 7].
    """
    intra_node_subgroup, _ = dist.new_subgroups(num_devices_per_node)
    return intra_node_subgroup


def _init_inter_node_process_group(
    global_process_group: dist.ProcessGroup,
    num_devices_per_node: int,
) -> dist.ProcessGroup:
    """
    Return an inter-node process group where each contained rank has the same local rank.

    For example, given each row is a distinct node:
    0 1 2 3 4 5 6 7 8
    9 10 11 12 13 14 15
    This API would return inter-node process group {0, 8}, {1, 9}, {2, 10}, and so forth
    depending on the process's rank. For example, rank 1 would get {1, 9}, rank 5
    would get {5, 13}.
    """
    # the inter-node pg that is returned
    inter_node_pg = None
    sharding_backend = dist.get_backend(global_process_group)
    world_size = dist.get_world_size(global_process_group)
    # Assuming fully homogeneous setup
    num_nodes = world_size // num_devices_per_node
    my_local_rank = dist.get_rank(global_process_group) % num_devices_per_node
    for local_rank in range(num_devices_per_node):
        ranks_for_inter_group = [
            local_rank + (i * num_devices_per_node) for i in range(num_nodes)
        ]
        # every rank always needs to call dist.new_group
        grp = dist.new_group(ranks=ranks_for_inter_group, backend=sharding_backend)
        if local_rank == my_local_rank:
            inter_node_pg = grp

    assert (
        inter_node_pg is not None
    ), f"{my_local_rank} expected to assign inter-node pg, but did not"
    return inter_node_pg


def maybe_init_dist_for_hybrid_shard(num_devices_per_node):
    default_group = _get_default_group()
    intra_node_group, inter_node_group = _init_intra_and_inter_node_groups(
        default_group, num_devices_per_node
    )
    return intra_node_group, inter_node_group


def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")


def example(rank, use_zero, process_group=None):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # create local model
    model = nn.Sequential(*[nn.Linear(5000, 5000).to(rank) for _ in range(20)])
    print_peak_memory("Max memory allocated after creating local model", rank)

    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank], process_group=process_group)
    print_peak_memory("Max memory allocated after creating DDP", rank)

    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    if use_zero:
        optimizer = ZeroRedundancyOptimizer(
            ddp_model.parameters(),
            optimizer_class=torch.optim.Adam,
            process_group=process_group,
            lr=0.01,
        )
    else:
        optimizer = torch.optim.Adam(
            ddp_model.parameters(),
            lr=0.01,
        )

    # forward pass
    outputs = ddp_model(torch.randn(20, 5000).to(rank))
    labels = torch.randn(20, 5000).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()

    # update parameters
    print_peak_memory("Max memory allocated before optimizer step()", rank)
    optimizer.step()
    print_peak_memory("Max memory allocated after optimizer step()", rank)

    print(f"params sum is: {sum(model.parameters()).sum()}")


def main(
    use_zero: bool = False,
    mode="all_devices",
):
    rank = maybe_init_dist()

    if rank is None:
        print("Not running distributed, skipping test")
        exit(0)

    torch.cuda.set_device(rank)

    world_size = torch.cuda.device_count()

    assert world_size == 8
    group_size = world_size // 2
    intra_node_group, inter_node_group = maybe_init_dist_for_hybrid_shard(group_size)

    print("rank: {}, use_zero: {}, mode: {}".format(rank, use_zero, mode))

    if mode == "all_devices":
        example(rank, use_zero)
    elif mode == "intra_node":
        example(rank, use_zero, process_group=intra_node_group)
    elif mode == "inter_node":
        example(rank, use_zero, process_group=inter_node_group)

    # # test all_gather

    # x = 0.1**rank
    # x = torch.tensor([x], dtype=torch.float32, device="cuda")

    # # 1. all_gather with simple one group

    # group = list(range(world_size))
    # result = funcol.all_reduce(x, "sum", group=group).item()
    # print("test 1 - rank: {}, result: {}".format(rank, result))

    # # 2. all_gather with inter_node_state

    # result = funcol.all_reduce(x, "sum", group=intra_node_group).item()
    # print("test 2 - rank: {}, result: {}".format(rank, result))

    # # 3. all_gather with inter_node_state

    # result = funcol.all_reduce(x, "sum", group=inter_node_group).item()
    # print("test 3 - rank: {}, result: {}".format(rank, result))


if __name__ == "__main__":
    fire.Fire(main)
