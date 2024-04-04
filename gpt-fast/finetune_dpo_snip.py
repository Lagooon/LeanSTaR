# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict
import itertools

import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

import torch._inductor.config
import torch._dynamo.config

from batched_generate_lean import generate_tactic

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    import wandb
except ImportError:
    wandb = None

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from models.model import Transformer, set_global_compile_mode
from models.tp import (
    maybe_init_dist,
    initialize_model_parallel,
    get_model_parallel_group,
    get_model_parallel_world_size,
    get_data_parallel_world_size,
    clip_grad_norm_,
)

from data_utils.common_utils import manual_seed
from data_utils.data_utils_dpo import make_dpo_data_module
from data_utils.tokenizer_utils import FakePreTrainedTokenizer

from training_utils.hf_argparser import HfArgumentParser
from training_utils.training_args import TrainingArguments
from training_utils.checkpoint_hook import (
    checkpoint_hook,
    get_latest_checkpoint_path,
    load_checkpoint,
    load_model_from_from_ckpt,
)
from training_utils.trainer_utils import (
    create_optimizer,
    create_fsdp_model_for_finetune,
    get_cosine_schedule_with_warmup,
)

IGNORE_INDEX = -100


def prepare_data(
    sources: list[str],
    targets: list[str],
    tokenizer: FakePreTrainedTokenizer,
    source_max_len: int,
    target_max_len: int,
    total_max_len: int,
    train_on_source: bool,
    add_eos_to_target: bool,
    return_win_rate: bool = False,
    win_rate: int = 1.0,
):
    # Extract elements

    # Tokenize
    tokenized_sources_with_prompt = tokenizer(
        sources,
        max_length=source_max_len,
        padding="max_length",
        truncation=True,
        add_bos=True,
        add_eos=False,
        padding_side="left",
        truncation_side="left",
    )


    

    # logger.warning(f"Tokenizing {len(targets)} pairs...")
    tokenized_targets = tokenizer(
        targets,
        max_length=target_max_len,
        padding="max_length",
        truncation=True,
        add_bos=False,
        add_eos=add_eos_to_target,
        padding_side="right",
        truncation_side="right",
    )
    # Build the input and labels for causal LM
    input_ids = []
    labels = []
    for source_length, tokenized_source, tokenized_target in zip(
        tokenized_sources_with_prompt["length"],
        tokenized_sources_with_prompt["input_ids"],
        tokenized_targets["input_ids"],
    ):
        full_seq = tokenized_source + tokenized_target

        # move the beginning padding to the end of the full_seq
        num_begin_padding = len(tokenized_source) - source_length
        full_seq = full_seq[num_begin_padding:] + full_seq[:num_begin_padding]

        if total_max_len is not None:
            full_seq = full_seq[:total_max_len]

        # input_ids.append(torch.tensor(full_seq))
        input_ids.append(full_seq)
        if not train_on_source:
            full_seq_label = (
                [tokenizer.pad_id for _ in range(source_length)]
                + tokenized_target
                + [tokenizer.pad_id for _ in range(num_begin_padding)]
            )
            if total_max_len is not None:
                full_seq_label = full_seq_label[:total_max_len]
            # labels.append(torch.tensor(full_seq_label))
            labels.append(full_seq_label)
        else:
            # labels.append(torch.tensor(copy.deepcopy(full_seq)))
            labels.append(full_seq)
    # Apply padding
    # input_ids = pad_sequence(
    #     input_ids, batch_first=True, padding_value=tokenizer.pad_id
    # )
    # labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_id)
    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    data_dict = {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(tokenizer.pad_id),
    }
    if labels is not None:
        data_dict["labels"] = labels
    if return_win_rate:
        data_dict["win_rate"] = torch.tensor(win_rate).view(-1, 1)
    return data_dict

def model_forward(model, x, input_pos):
    return model(x, input_pos, fully_causal=True)


def model_forward_with_loss(
    model: Transformer,
    ref_model: Transformer,
    input_ids_w: torch.Tensor,
    labels_w: torch.Tensor,
    input_ids_l: torch.Tensor,
    labels_l: torch.Tensor,
    win_rate_w: torch.Tensor,
    win_rate_l: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    variant: str = "dpo",
) -> torch.Tensor:
    """
    Compute the loss for a given model and prompts.
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    B, T = input_ids_w.shape
    device = input_ids_w.device
    with torch.device(device):
        ref_model.setup_caches(max_batch_size=B, max_seq_length=T, kv_cache=False)
        model.setup_caches(max_batch_size=B, max_seq_length=T, kv_cache=False)
    input_pos = torch.arange(0, T, device=device)

    labels_w, labels_l = labels_w[..., 1:].contiguous(), labels_l[..., 1:].contiguous()

    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=False, enable_mem_efficient=False
    ):
        with torch.no_grad():
            ref_logits_w = model_forward(ref_model, input_ids_w, input_pos)[
                ..., :-1, :
            ].contiguous().float()
            ref_logits_l = model_forward(ref_model, input_ids_l, input_pos)[
                ..., :-1, :
            ].contiguous().float()

            with torch.autocast(device_type="cuda", enabled=False):
                ref_logprobs_w = -F.cross_entropy(
                    ref_logits_w.transpose(-1, -2).float(),
                    labels_w,
                    reduction="none",
                ).sum(dim=-1)
                ref_logprobs_l = -F.cross_entropy(
                    ref_logits_l.transpose(-1, -2).float(),
                    labels_l,
                    reduction="none",
                ).sum(dim=-1)

                if "average" in variant:
                    ref_w_weight = (labels_w != IGNORE_INDEX).float().sum(dim=-1)
                    ref_l_weight = (labels_l != IGNORE_INDEX).float().sum(dim=-1)
                    ref_logprobs_w = ref_logprobs_w / (ref_w_weight + 1e-8)
                    ref_logprobs_l = ref_logprobs_l / (ref_l_weight + 1e-8)

        logits_w = model_forward(model, input_ids_w, input_pos)[
            ..., :-1, :
        ].contiguous().float()
        logits_l = model_forward(model, input_ids_l, input_pos)[
            ..., :-1, :
        ].contiguous().float()

    with torch.autocast(device_type="cuda", enabled=False):
        logprobs_w = -F.cross_entropy(
            logits_w.transpose(-1, -2).float(), labels_w, reduction="none"
        ).sum(dim=-1)
        logprobs_l = -F.cross_entropy(
            logits_l.transpose(-1, -2).float(), labels_l, reduction="none"
        ).sum(dim=-1)

        if "average" in variant:
            w_weight = (labels_w != IGNORE_INDEX).float().sum(dim=-1)
            l_weight = (labels_l != IGNORE_INDEX).float().sum(dim=-1)
            logprobs_w = logprobs_w / (w_weight + 1e-8)
            logprobs_l = logprobs_l / (l_weight + 1e-8)

        # logits = (logprobs_w - ref_logprobs_w) - (logprobs_l - ref_logprobs_l)
        logits_1 = logprobs_w - ref_logprobs_w
        logits_2 = logprobs_l - ref_logprobs_l
        logits = logits_1 - logits_2

        win_rate_w = win_rate_w.view(logprobs_w.shape)
        win_rate_l = win_rate_l.view(logprobs_l.shape)

        if "ipo" in variant:
            # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
            loss = (logits - 1.0 / (2 * beta)).pow(2).mean()
        elif "rpo" in variant:
            loss_1 = (logits_1 - (1.0 / beta) * (win_rate_w - 0.5)).pow(2).mean()
            loss_2 = (logits_2 - (1.0 / beta) * (win_rate_l - 0.5)).pow(2).mean()
            loss = (loss_1 + loss_2) / 2.0
        elif label_smoothing == 0.0:
            # label_smoothing=0 gives original DPO
            # Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf
            loss = -F.logsigmoid(beta * logits).mean()
        else:
            # Eq. 3 of https://ericmitchell.ai/cdpo.pdf;
            loss = (
                -F.logsigmoid(beta * logits).mean() * (1 - label_smoothing)
                - F.logsigmoid(-beta * logits).mean() * label_smoothing
            )

        accuracy = (logits > 0.0).float().mean()
        accuracy_policy = (logprobs_w > logprobs_l).float().mean()
        accuracy_ref = (ref_logprobs_w > ref_logprobs_l).float().mean()
        metrics = {
            "accuracy": accuracy,
            "accuracy_policy": accuracy_policy,
            "accuracy_reference": accuracy_ref,
        }

    return loss, metrics


def compute_reward_modeling_metrics(logits, labels) -> Dict:
    # eval_prediction.label_ids is a tuple that matches up with `training_args.label_names`.
    predictions = (logits >= 0.0).long()
    accuracy = predictions.eq(labels).float().mean()
    label_positive_rate = (labels == 1).float().mean()
    positive_rate = (predictions == 1).float().mean()
    true_positive_rate = (predictions * labels).float().sum() / labels.sum()
    false_positive_rate = (predictions * (1 - labels)).float().sum() / (
        1 - labels
    ).sum()
    return dict(
        accuracy=accuracy,
        label_positive_rate=label_positive_rate,
        positive_rate=positive_rate,
        true_positive_rate=true_positive_rate,
        false_positive_rate=false_positive_rate,
    )


def encode_tokens(tokenizer, string, bos=True, device="cuda"):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)


def main(
    args: TrainingArguments,
) -> None:
    """Finetune a model on a given dataset."""
    checkpoint_path = args.checkpoint_path
    sft_checkpoint_path = args.sft_checkpoint_path
    compile = args.compile
    assert checkpoint_path.is_file(), checkpoint_path
    assert sft_checkpoint_path.is_dir(), sft_checkpoint_path

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), tokenizer_path

    set_global_compile_mode(args.compile)

    global print
    device_id = maybe_init_dist()
    use_tp = device_id is not None
    if use_tp:
        tp_size = args.tensor_parallel_size or torch.distributed.get_world_size()
        initialize_model_parallel(tp_size)
        torch.distributed.barrier()
        tp_group = get_model_parallel_group()

        if device_id != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    if args.report_to == "wandb" and wandb is not None:
        if device_id == 0:
            wandb_logging_dir = os.path.join(
                tempfile.gettempdir(), f"{os.getuid()}_wandb"
            )
            if not os.path.exists(wandb_logging_dir):
                os.makedirs(wandb_logging_dir, exist_ok=True)
            os.environ["WANDB_DIR"] = wandb_logging_dir
            wandb.init(
                name=args.wandb_name,
                project=args.wandb_project,
                entity=args.wandb_entity,
                resume="allow",
                magic=True,
                dir=wandb_logging_dir,
                force=True,
            )
            wandb.config.update(vars(args))

    device = "cuda"
    precision = args.param_dtype

    print("Loading model ...")
    t0 = time.time()

    resume_from_checkpoint = None
    resume_epoch = 0
    resume_global_step = 0

    if args.resume_from_checkpoint:
        (
            resume_from_checkpoint,
            resume_epoch,
            resume_global_step,
        ) = get_latest_checkpoint_path(args.save_dir)

    if resume_from_checkpoint is not None:
        sft_checkpoint_path = None

    model = load_model_from_from_ckpt(
        checkpoint_path,
        sft_checkpoint_path,
        device,
        precision,
        use_tp,
        requires_grad=True,
    )

    ref_model = load_model_from_from_ckpt(
        checkpoint_path,
        sft_checkpoint_path,
        device,
        precision,
        use_tp,
        requires_grad=False,
    )
    
    _model = load_model_from_from_ckpt(
        checkpoint_path,
        sft_checkpoint_path,
        device,
        precision,
        use_tp,
        requires_grad=False,
    )
        

    torch.cuda.synchronize()
    if use_tp:
        torch.distributed.barrier()
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = FakePreTrainedTokenizer(tokenizer_path)

    data_module = make_dpo_data_module(
        tokenizer=tokenizer,
        args=args,
        snip=True,
    )
    train_dataset = data_module["train_dataset"]
    data_collator = data_module["data_collator"]

    model_size = sum(
        [
            p.numel() * p.dtype.itemsize
            for p in itertools.chain(model.parameters(), model.buffers())
        ]
    )

    print(f"Model size: {model_size / 1e6:.02f} MB")
    manual_seed(args.seed)

    sampler = None
    if use_tp:
        sampler = DistributedSampler(
            train_dataset,
            shuffle=True,
            drop_last=True,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=data_collator,
    )

    if args.print_training_examples:
        print("Training examples:")
        cnt = 3
        for batch in train_loader:
            print("Input 1:")
            print(
                batch["input"]
            )
            print("=" * 20)
            cnt -= 1
            if cnt == 0:
                break

    if compile:
        model = torch.compile(model)
        ref_model = torch.compile(ref_model)

    trainable_param_names = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]

    use_fsdp = False

    if get_data_parallel_world_size() > 1:
        use_fsdp = True
        model = create_fsdp_model_for_finetune(args, model)
        ref_model = create_fsdp_model_for_finetune(args, ref_model)

        print("Using FSDP ...")
        print(model)

    optimizer = create_optimizer(
        args,
        model=model,
        optimizer_cpu_offload=args.optimizer_cpu_offload,
        model_cpu_offload=False,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=len(train_loader) * args.warmup_ratio,
        max_epochs=len(train_loader) * args.num_train_epochs,
        warmup_start_ratio=0.0,
        eta_min_ratio=args.lr_eta_min / args.learning_rate,
    )

    if resume_from_checkpoint is not None:
        print(
            f"Resuming from checkpoint: {resume_from_checkpoint} (epoch {resume_epoch}, global step {resume_global_step})"
        )
        load_checkpoint(
            resume_from_checkpoint, model, optimizer, scheduler, use_fsdp=use_fsdp
        )

    micro_train_batch_size = (
        args.micro_train_batch_size or args.per_device_train_batch_size
    )

    assert (
        args.per_device_train_batch_size % micro_train_batch_size == 0
    ), f"per_device_train_batch_size ({args.per_device_train_batch_size}) must be divisible by micro_train_batch_size ({micro_train_batch_size})"
    accumulate_steps = args.per_device_train_batch_size // micro_train_batch_size

    print(
        "Batch size per GPU for training: {}\n".format(
            args.per_device_train_batch_size
        ),
        "Micro batch size for training: {}\n".format(micro_train_batch_size),
        "Gradient accumulation steps: {}\n".format(accumulate_steps),
    )

    micro_train_batch_size = micro_train_batch_size * torch.distributed.get_world_size()

    epoch_length = len(train_loader)

    if args.do_train:
        print("Starting training ...")
        t0 = time.time()
        for epoch in tqdm.trange(
            args.num_train_epochs, desc="Epoch", disable=device_id != 0
        ):
            if sampler is not None:
                train_loader.sampler.set_epoch(epoch)
            pbar = tqdm.tqdm(
                enumerate(train_loader),
                desc="Iteration",
                disable=device_id != 0,
                total=len(train_loader),
            )
            for it, batch in pbar:
                global_step = epoch * epoch_length + it
                if global_step < resume_global_step:
                    continue

                # torch.cuda.synchronize()
                model.zero_grad()
                
                output_l = generate_tactic(
                    batch["input"],
                    _model,
                    tokenizer,
                    128,
                    args.per_device_train_batch_size,
                    args.per_device_train_batch_size,
                    temperature=0.8,
                    top_k=500,
                    prompted=True,
                )
                torch.cuda.empty_cache()
                data_w = prepare_data(
                    sources=batch["input"],
                    targets=batch["output"],
                    tokenizer=tokenizer,
                    source_max_len=args.source_max_len,
                    target_max_len=args.target_max_len,
                    total_max_len=args.total_max_len,
                    train_on_source=args.train_on_source,
                    add_eos_to_target=args.add_eos_to_target,
                    return_win_rate=True,
                    win_rate=[1.0]*args.per_device_train_batch_size,
                )
                
                data_l = prepare_data(
                    sources=batch["input"],
                    targets=output_l,
                    tokenizer=tokenizer,
                    source_max_len=args.source_max_len,
                    target_max_len=args.target_max_len,
                    total_max_len=args.total_max_len,
                    train_on_source=args.train_on_source,
                    add_eos_to_target=args.add_eos_to_target,
                    return_win_rate=True,
                    win_rate=[0.0]*args.per_device_train_batch_size,
                )

                input_ids_w = data_w["input_ids"].to(device=device)
                labels_w = data_w["labels"].to(device=device)
                input_ids_l = data_l["input_ids"].to(device=device)
                labels_l = data_l["labels"].to(device=device)
                win_rate_w = data_w["win_rate"].to(device=device)
                win_rate_l = data_l["win_rate"].to(device=device)

                (
                    input_ids_w,
                    labels_w,
                    input_ids_l,
                    labels_l,
                    win_rate_w,
                    win_rate_l,
                ) = prepare_batch(
                    input_ids_w,
                    labels_w,
                    input_ids_l,
                    labels_l,
                    win_rate_w,
                    win_rate_l,
                    tokenizer=tokenizer,
                    use_tp=use_tp,
                    sync_group=tp_group,
                )

                loss_scale = 1.0 / accumulate_steps
                for ex_idx in range(0, input_ids_w.size(0), micro_train_batch_size):
                    if ex_idx + micro_train_batch_size < input_ids_w.size(0):
                        with torch.cuda.amp.autocast(dtype=args.compute_dtype):
                            loss, metrics = model_forward_with_loss(
                                model,
                                ref_model,
                                input_ids_w[ex_idx : ex_idx + micro_train_batch_size],
                                labels_w[ex_idx : ex_idx + micro_train_batch_size],
                                input_ids_l[ex_idx : ex_idx + micro_train_batch_size],
                                labels_l[ex_idx : ex_idx + micro_train_batch_size],
                                win_rate_w[ex_idx : ex_idx + micro_train_batch_size],
                                win_rate_l[ex_idx : ex_idx + micro_train_batch_size],
                                beta=args.dpo_beta,
                                variant=args.dpo_variant,
                                label_smoothing=args.dpo_label_smoothing,
                            )
                        (loss_scale * loss).backward()
                    else:
                        with torch.cuda.amp.autocast(dtype=args.compute_dtype):
                            loss, metrics = model_forward_with_loss(
                                model,
                                ref_model,
                                input_ids_w[ex_idx:],
                                labels_w[ex_idx:],
                                input_ids_l[ex_idx:],
                                labels_l[ex_idx:],
                                win_rate_w[ex_idx:],
                                win_rate_l[ex_idx:],
                                beta=args.dpo_beta,
                                variant=args.dpo_variant,
                                label_smoothing=args.dpo_label_smoothing,
                            )
                        (loss_scale * loss).backward()
                        grad_norm = clip_grad_norm_(model, 1.0)
                        optimizer.step()
                        scheduler.step()

                loss_copy = loss.detach().clone()
                acc_copy = metrics["accuracy"].detach().clone()
                acc_ref_copy = metrics["accuracy_reference"].detach().clone()
                acc_pol_copy = metrics["accuracy_policy"].detach().clone()
                torch.distributed.all_reduce(loss_copy)
                torch.distributed.all_reduce(acc_copy)
                torch.distributed.all_reduce(acc_ref_copy)
                torch.distributed.all_reduce(acc_pol_copy)
                avg_loss = (loss_copy / torch.distributed.get_world_size()).item()
                avg_acc = (acc_copy / torch.distributed.get_world_size()).item()
                avg_acc_ref = (acc_ref_copy / torch.distributed.get_world_size()).item()
                avg_acc_pol = (acc_pol_copy / torch.distributed.get_world_size()).item()
                grad_norm_copy = grad_norm.detach().clone().item()

                if device_id == 0:
                    if args.report_to == "wandb" and wandb is not None:
                        wandb.log(
                            {
                                "loss": avg_loss,
                                "accuracy": avg_acc,
                                "accuracy_policy": avg_acc_pol,
                                "accuracy_reference": avg_acc_ref,
                                "learning_rate": scheduler.get_last_lr()[0],
                                "epoch": epoch,
                                "step": it,
                                "grad_norm": grad_norm_copy,
                            },
                            step=global_step,
                        )
                    else:
                        # Just print to stdout.
                        print(
                            {
                                "loss": avg_loss,
                                "accuracy": avg_acc,
                                "accuracy_policy": avg_acc_pol,
                                "accuracy_reference": avg_acc_ref,
                                "learning_rate": scheduler.get_last_lr()[0],
                                "epoch": epoch,
                                "step": it,
                                "grad_norm": grad_norm_copy,
                            }
                        )

                checkpoint_hook(
                    args,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    epoch_length,
                    use_fsdp=use_fsdp,
                    trainable_param_names=trainable_param_names,
                )

        torch.cuda.synchronize()

        checkpoint_hook(
            args,
            model,
            optimizer,
            scheduler,
            epoch,
            epoch * epoch_length,
            epoch_length,
            use_fsdp=use_fsdp,
            trainable_param_names=trainable_param_names,
        )
        
        epoch = args.num_train_epochs

        print(f"Time to train: {time.time() - t0:.02f} seconds")


def prepare_batch(
    input_ids_w,
    labels_w,
    input_ids_l,
    labels_l,
    win_rate_w,
    win_rate_l,
    tokenizer,
    use_tp,
    sync_group,
):
    pad_id = tokenizer.pad_id
    unk_id = tokenizer.unk_id

    labels_w[labels_w == pad_id] = IGNORE_INDEX
    labels_l[labels_l == pad_id] = IGNORE_INDEX
    if pad_id < 0:
        input_ids_w[input_ids_w == pad_id] = unk_id
        input_ids_l[input_ids_l == pad_id] = unk_id

    if use_tp and get_model_parallel_world_size() > 1:
        # aggregate (concat) all the inputs across tp sync_group
        new_labels_w = torch.empty_like(labels_w).repeat(sync_group.size(), 1)
        new_labels_l = torch.empty_like(labels_l).repeat(sync_group.size(), 1)
        new_input_ids_w = torch.empty_like(input_ids_w).repeat(sync_group.size(), 1)
        new_input_ids_l = torch.empty_like(input_ids_l).repeat(sync_group.size(), 1)
        new_win_rate_w = torch.empty_like(win_rate_w).repeat(sync_group.size(), 1)
        new_win_rate_l = torch.empty_like(win_rate_l).repeat(sync_group.size(), 1)

        handle_1 = torch.distributed.all_gather_into_tensor(
            new_labels_w, labels_w, group=sync_group, async_op=True
        )
        handle_2 = torch.distributed.all_gather_into_tensor(
            new_labels_l, labels_l, group=sync_group, async_op=True
        )
        handle_3 = torch.distributed.all_gather_into_tensor(
            new_input_ids_w, input_ids_w, group=sync_group, async_op=True
        )
        handle_4 = torch.distributed.all_gather_into_tensor(
            new_input_ids_l, input_ids_l, group=sync_group, async_op=True
        )
        handle_5 = torch.distributed.all_gather_into_tensor(
            new_win_rate_w, win_rate_w, group=sync_group, async_op=True
        )
        handle_6 = torch.distributed.all_gather_into_tensor(
            new_win_rate_l, win_rate_l, group=sync_group, async_op=True
        )

        handle_1.wait()
        handle_2.wait()
        handle_3.wait()
        handle_4.wait()
        handle_5.wait()
        handle_6.wait()

        return (
            new_input_ids_w,
            new_labels_w,
            new_input_ids_l,
            new_labels_l,
            new_win_rate_w,
            new_win_rate_l,
        )

    return input_ids_w, labels_w, input_ids_l, labels_l, win_rate_w, win_rate_l


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments,))
    parser.add_argument('--local-rank', type=int, default=None)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
