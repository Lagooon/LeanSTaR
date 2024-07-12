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
from training_utils.trainer_utils import (
    create_optimizer,
    create_fsdp_model_for_finetune,
    get_cosine_schedule_with_warmup,
)
import vllm, transformers, json

IGNORE_INDEX = -100




def _load_model(checkpoint_path, precision):
    model = vllm.LLM(
        model=checkpoint_path,
        tensor_parallel_size=1,
        dtype=precision,
        max_num_batched_tokens=16384,
        trust_remote_code=True,
        enforce_eager=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    return model, tokenizer
@torch.no_grad()
def generate_tactic(
    prompt,
    model,
    tokenizer,
    max_seq_len,
    num_samples,
    batch_size,
    temperature,
):
    texts = []
    params = vllm.SamplingParams(
        n=num_samples,
        temperature=temperature,
        use_beam_search=False,
        max_tokens=max_seq_len,
    )
    outputs = model.generate(prompt, params, use_tqdm=False)
    if len(outputs) == 0:
        return [], []
    for output in outputs:
        tokens = output.outputs[0]
        text = tokens.text.replace(tokenizer.eos_token, '')
        texts.append(text)

    return texts
def main(
    args: TrainingArguments,
) -> None:
    """Finetune a model on a given dataset."""
    checkpoint_path = args.checkpoint_path
    device = "cuda"
    precision = args.param_dtype


    print("Loading model ...")
    t0 = time.time()
    model, tokenizer = _load_model(str(checkpoint_path), precision)

    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    data_module = make_dpo_data_module(
        tokenizer=tokenizer,
        args=args,
        snip=True,
    )
    train_dataset = data_module["train_dataset"]
    data_collator = data_module["data_collator"]
    
    
    manual_seed(args.seed)

    sampler = None

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

    epoch_length = len(train_loader)
    sub_length = epoch_length / args.split

    pbar = tqdm.tqdm(
        enumerate(train_loader),
        desc="Iteration",
        total=len(train_loader),
            )
    data = []
    ans, cnt = 0, 0
    #args.frac -= 4
    for it, batch in pbar:
        if it < sub_length * args.frac:
            continue
        if it >= sub_length * (args.frac + 1):
            break
        output_1 = batch["output"][0]
        output_l = generate_tactic(
            batch["input"][0].replace("Next tactic", "Reasoning"),
            model,
            tokenizer,
            512,
            32,
            1,
            temperature=0.7,
        )
        
        for output_2 in output_l:
            tactic = output_2.split("```lean4\n")[-1].split('```')[0].split("---")[0].strip()
            label = output_1.split("---")[0].strip()
            if tactic == label:
                ans += 1
                data.append({
                    "input" : batch["input"][0].replace("Next tactic", "Reasoning"),
                    "output" : output_2
                })
                break
        cnt += 1
    print(ans, cnt, ans/cnt)
    json_file = "nonhinted-generated-train-{args.frac}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
   



if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments,))
    args = parser.parse_args_into_dataclasses()[0]
    main(args)

