# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import gc
import os
import sys
from copy import deepcopy
import time
from pathlib import Path
from typing import Optional, Tuple, Dict
from collections import OrderedDict
import itertools
import fcntl

import torch
from torch.distributed import _functional_collectives as funcol

import torch._inductor.config
import torch._dynamo.config

import heapq
import datetime
from tqdm import tqdm, trange
from lean_dojo import *
import subprocess
from transformers import AutoModelForCausalLM


import vllm
import transformers
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
#torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from models.model import Transformer
from models.tp import maybe_init_dist, initialize_model_parallel, apply_tp
from models.tp import (
    get_model_parallel_rank,
    get_model_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
)
from data_utils.tokenizer_utils import (
    FakePreTrainedTokenizer,
    batch_encode_tokens,
)
#from training_utils.checkpoint_hook import (
#    get_latest_checkpoint_path,
#    load_inference_checkpoint,
#)


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)
    # return torch.argmax(probs_sort, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        pivot = v.select(-1, -1).view(-1, 1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits, vocab_parallel, temperature: float = 1.0, top_k: Optional[int] = None, **sampling_kwargs,
):
    with torch.autocast(device_type="cuda", enabled=False):
        logits = logits[:, -1].float()

        if vocab_parallel:
            logits = funcol.all_gather_tensor(
                logits, gather_dim=-1, group=get_model_parallel_group()
            )

        probs = logits_to_probs(logits, temperature, top_k)
        idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, torch.nn.functional.log_softmax(logits, dim=-1)


def remove_all_backward_hooks(model: torch.nn.Module) -> Dict[str, OrderedDict]:
    all_backward_hooks = {}

    for name, module in model.named_modules():
        all_backward_hooks[name] = module._backward_hooks
        module._backward_hooks = OrderedDict()

    return all_backward_hooks


def prefill(
    model: Transformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    left_pad_mask_pos: torch.Tensor,
    **sampling_kwargs,
) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos, left_pad_mask_pos)
    return sample(logits, model.vocab_parallel, **sampling_kwargs)


def decode_one_token(
    model: Transformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    left_pad_mask_pos: torch.Tensor,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos, left_pad_mask_pos)
    return sample(logits, model.vocab_parallel, **sampling_kwargs)


def decode_n_tokens(
    model: Transformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    left_pad_mask_pos: torch.Tensor,
    num_new_tokens: int,
    eos_token: Optional[str] = None,
    tokenizer = None,
    **sampling_kwargs,
):
    eos_flag = None
    if eos_token is not None:
        eos_flag = torch.zeros_like(
            cur_token, dtype=torch.bool, device=cur_token.device
        )

    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        
        if eos_flag is not None:
            for i in range(cur_token.size(0)):
                last_token = tokenizer.decode(cur_token[i])
                eos_flag[i] = eos_flag[i] | (eos_token in last_token)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):  # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, left_pad_mask_pos, **sampling_kwargs
            )
        input_pos += 1
        prob = []
        for i in range(next_token.size(0)):
            prob.append(next_prob[i, next_token[i]])
        next_prob = torch.Tensor(prob).view(-1, 1)
        if eos_flag is not None:
            next_prob[eos_flag] = 0
            next_token[eos_flag] = tokenizer.eos_id
        new_tokens.append(next_token.clone().view(-1, 1))
        new_probs.append(next_prob.clone().view(-1, 1))
        cur_token = next_token.view(-1, 1)

        if eos_flag is not None:
            eos_flag = eos_flag | (cur_token == tokenizer.eos_id)

        if eos_flag is not None and eos_flag.all():
            break

    return new_tokens, new_probs, i


def model_forward(model, x, input_pos):
    return model(x, input_pos)


@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    left_pad_mask_pos: torch.Tensor,
    max_new_tokens: int,
    eos_token: Optional[str] = None,
    max_seq_len: Optional[int] = None,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    B = prompt.size(0)
    T = prompt.size(1)
    T_new = T + max_new_tokens
    # max_seq_length = min(T_new, model.config.block_size)
    # max_seq_length = max_seq_len or model.config.block_size
    max_seq_length = model.config.block_size

    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(max_batch_size=B, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty((B, T_new), dtype=dtype, device=device)
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_mem_efficient=False, enable_math=True
    ):
        next_token, next_probs = prefill(
            model, prompt, input_pos, left_pad_mask_pos, **sampling_kwargs
        )
        prob = []
        for i in range(next_token.size(0)):
            prob.append(next_probs[i, next_token[i]])
        next_probs = torch.Tensor(prob).view(-1, 1)

    seq[:, T] = next_token.view(B)

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    generated_tokens, generated_probs, num_decoded_tokens = decode_n_tokens(
        model,
        next_token.view(B, -1),
        input_pos,
        left_pad_mask_pos,
        max_new_tokens - 1,
        eos_token,
        **sampling_kwargs,
    )

    generated_tokens = torch.cat(generated_tokens, dim=-1).view(B, -1)
    generated_probs = torch.cat([next_probs.clone().view(-1, 1)] + generated_probs, dim=-1).view(B, -1)

    seq[:, T + 1 : T + 1 + generated_tokens.size(1)] = generated_tokens

    return seq, generated_probs, num_decoded_tokens

def _load_model(checkpoint_path, device, precision, tp_size):
    model_name = "/nobackup/users/zhiqings/haohanl/Lean/checkpoints/internlm/internlm2-math-plus-7b/cots"
    model = vllm.LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        dtype=precision,
        max_num_batched_tokens=32768,
        trust_remote_code=True,
        enforce_eager=True,
    )
    #model_name = "/localdata_ssd/Lean/checkpoints/internlm/internlm2-math-base-7b"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
    with torch.device("meta"):
        model = Transformer.from_name(
            checkpoint_path.parent.name,
            freeze_tok_embeddings=True,
            freeze_output=True,
            freeze_norm=True,
            vocab_parallel=True,
        )

    if "int8" in str(checkpoint_path):
        print("Using int8 weight-only quantization!")
        from models.quantize import WeightOnlyInt8QuantHandler

        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    if "int4" in str(checkpoint_path):
        print("Using int4 quantization!")
        path_comps = checkpoint_path.name.split(".")
        assert path_comps[-2].startswith("g")
        groupsize = int(path_comps[-2][1:])
        from models.quantize import WeightOnlyInt4QuantHandler

        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device=device, dtype=precision)
    return model.eval()
    """

def _tactic_state(state):
    if isinstance(state, TacticState):
        ts = state.pp
    else:
        ts = state.unsolved_tactic_state
    return ts

def chat_template_to_prompt(prompt_list):
    result = ""
    total_step = len(prompt_list)
    for i, message in enumerate(prompt_list):
        result += ('<|im_start|>' + message['role'] +
            '\n' + message['content'])
        if i+1 != total_step:
            result += '<|im_end|>\n'
        elif message['role'] == 'user':
            result += '<|im_end|>\n<|im_start|>assistant\n'
    return result

def _prompt_fewshot(state):
    prompt = f"My LEAN 4 state is:\n```lean\n" + state + \
        "```\nPlease predict a possible tactic to help me prove the theorem."
    prompt = [{"role": "user", "content": prompt}]
    return chat_template_to_prompt(prompt)

def _unique_sorted(texts, scores):
    texts_ = []
    scores_ = []
    for t, s in sorted(zip(texts, scores), key=lambda x: -x[1]):
        if t not in texts_:
            texts_.append(t)
            scores_.append(s)
    return texts_, scores_

@torch.no_grad()
def generate_tactic(
    prompts,
    model,
    tokenizer,
    max_seq_len,
    num_samples,
    temperature
):
    texts = []
    prompts = [_prompt_fewshot(prompt) for prompt in prompts]
    params = vllm.SamplingParams(
        n=1,
        temperature=temperature,
        use_beam_search=False,
        max_tokens=max_seq_len,
    )
    outputs = model.generate(prompts, params, use_tqdm=False)
    for i in range(len(prompts)):
        output = outputs[i].outputs[0]
        text = output.text.replace(tokenizer.eos_token, '')
        texts.append(text)

    return texts
    """
    device = "cuda"
    prompt = _prompt_fewshot(prompt)
    encoded, left_pad_mask_pos = batch_encode_tokens(
                    tokenizer, [prompt] * batch_size, bos=True, device=device
                )
    prompt_length = encoded.size(1)
    if prompt_length > 892:
        encoded = encoded[:, :892].contiguous()
        prompt_length = 892
    full_y_list, y_probs = None, None
    for i in range(0, num_samples // batch_size):
        y, y_prob, num_decoded_tokens = generate(
                model,
                encoded,
                left_pad_mask_pos,
                max_seq_len,
                temperature=temperature,
                eos_token='\n',
                max_seq_len=max_seq_len + prompt_length + 1,
                tokenizer=tokenizer,
                top_k=top_k,
            )
        y, y_prob = y.cpu(), y_prob.cpu()
        if full_y_list == None:
            full_y_list = y.tolist()
            y_probs = y_prob.tolist()
        else:
            full_y_list = full_y_list + y.tolist()
            y_probs = y_probs + y_prob.tolist()
    outputs = []
    step_scores = []
    for y_list, probs in zip(full_y_list, y_probs):
        output = post_process(y_list[prompt_length:], tokenizer)
        score = sum(probs)
        if '\n' in output:
            output = output[:output.index('\n')]
        outputs.append(output)
        step_scores.append(score)
    outputs, step_scores = _unique_sorted(outputs, step_scores)
    return outputs, step_scores
    """
def best_first_search(
        theorem,
        model,
        tokenizer,
        max_iters,
        temperatures,
        num_samples,
        batch_size,
        timeout=600,
        early_stop=False,
        max_seq_len=512,
        top_k=200
) -> dict:
    """Best first search."""
    attempt_results = []
    print("theorem: ", theorem)
    try:
        with Dojo(theorem, hard_timeout=timeout) as (dojo, init_state):

            start = time.time()
            proof_finished = False
            cnt = 0
            states, steps, traces = [], [], []
            for i in range(num_samples):
                states.append(init_state)
                steps.append([])
                traces.append([])

            for iteration in trange(max_iters):
                istart = time.time()
                if istart - start > timeout:
                    break
                if proof_finished:
                    break

                ts = [_tactic_state(state) for state in states]
                
                step_cands = generate_tactic(
                    ts,
                    model,
                    tokenizer,
                    max_seq_len=max_seq_len,
                    num_samples=1,
                    temperature=temperatures
                )

                #if iteration < 2:
                #    print(iteration, " # state: ",ts[0])
                #    print(tatics: ", step_cands[0])

                step_cots = step_cands
                step_cands = [s.split("```lean\n")[-1].split('```')[0].split('---')[0].strip() for s in step_cands]
                #print(step_cands[:10])
                for i in range(num_samples):
                    state, step, step_cot = states[i], step_cands[i], step_cots[i]
                    result = dojo.run_tac(state, step)
                    step_trace = {
                        "tactic": step,
                        "full_cot" : step_cot,
                        "state_before": _tactic_state(state)
                    }
                    if isinstance(result, ProofFinished):
                        attempt_results.append({
                            'theorem': theorem.full_name,
                            'proof': steps[i] + [step],
                            'success': True,
                            'failure_reason': '',
                            'trace': traces[i] + [step_trace],
                            'temperature': temperatures,
                            'elapsed': start - time.time(),
                            'iteration': iteration
                        })
                        if early_stop:
                            return attempt_results
                        proof_finished = True
                        break
                    elif isinstance(result, TacticState):
                        #if _tactic_state(result) not in visited:
                        # Score is negative log probability summed across steps
                        #new_score = (total_score - score)
                        cnt += 1
                        states[i] = result
                        steps[i].append(step)
                        traces[i].append(step_trace)
    except (DojoInitError, DojoHardTimeoutError, DojoCrashError) as e:
        print("Error: ", e)
        if len(attempt_results) == 0:
            attempt_results.append({
                'theorem': theorem.full_name,
                'success': False,
                'failure_reason': type(e).__name__
            })

    if len(attempt_results) == 0:
        attempt_results.append({
            'theorem': theorem.full_name,
            'success': False,
            'failure_reason': 'SearchEnded'
        })

    return attempt_results


def _save(model_name, results, args_dict, output_dir, shard):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        'results__%s__%s.json' % (model_name.replace('/', '_'), shard)
    )
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'args': args_dict
            }, f, indent=4)
        print(output_file)



def _load_data(dataset_name, dataset_path):
    if 'minif2f' in dataset_name:
        data = []
        with open(dataset_path) as f:
            for line in f.readlines():
                data_ = json.loads(line)
                assert data_['commit'] == 'd00c776260c77de7e70125ef0cd119de6c0ff1de'
                data.append(data_)

        if 'valid' in dataset_name:
            data = [x for x in data if x['split'] == 'valid']
        else:
            data = [x for x in data if x['split'] == 'test']
        repo = LeanGitRepo(data[0]['url'], data[0]['commit'])
    elif 'leandojo' in dataset_name:
        with open(dataset_path) as f:
            data = json.load(f)
        repo = LeanGitRepo(data[0]['url'], data[0]['commit'])
    else:
        raise NotImplementedError(dataset_name)

    return repo, data


def print_stats(results):
    print(len([x for x in results if x['success']]) / len(results))
    print("# successes: ", len([x for x in results if x['success']]), sep="\t")


def resume_from(results_filename, data):
    results = json.load(open(results_filename))['results']
    data = data[len(results):]
    print("=== Resuming from %d" % (len(results)))
    return results, data


def make_output_dir(output_dir):
    dt = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")
    output_dir = os.path.join(output_dir, dt)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir

def main(
    seed: int,
    batch_size: int = 4,
    num_samples: int = 5,
    temperature: float = 0.8,
    checkpoint_path: Path = Path(
        "checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"
    ),
    compile: bool = True,
    default_compile: bool = False,
    finetune_checkpoint_path: Optional[Path] = None,
    finetune_checkpoint_prefix: Optional[str] = None,
    tensor_parallel_size: Optional[int] = None,
    on_the_fly_8bit_quantization: bool = False,
    args = None
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer."""
    assert checkpoint_path.is_file(), checkpoint_path

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    if not tokenizer_path.is_file():
        tokenizer_path = checkpoint_path.parent

    global print
    rank = maybe_init_dist()
    use_tp = rank is not None
    tp_size = 1
    if use_tp:
        tp_size = tensor_parallel_size or torch.distributed.get_world_size()
        initialize_model_parallel(tp_size)
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    device = "cuda"
    precision = torch.bfloat16

    print("Loading model ...")
    t0 = time.time()
    model, tokenizer = _load_model(checkpoint_path, device, precision, tp_size)
    
    #tokenizer = FakePreTrainedTokenizer(tokenizer_path, model_vocab_size = 92544)
    '''
    if finetune_checkpoint_path is not None:
        finetune_checkpoint_path, _, _ = get_latest_checkpoint_path(
            finetune_checkpoint_path,
            prefix=finetune_checkpoint_prefix,
        )

        if finetune_checkpoint_path is not None:
            print(f"Loading finetune model from {finetune_checkpoint_path} ...")
            load_inference_checkpoint(finetune_checkpoint_path, model)
        model = model.to(device=device)
        model = model.eval()'''

    if on_the_fly_8bit_quantization:
        print("Quantizing model ...")
        from models.quantize import WeightOnlyInt8QuantHandler

        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime_on_the_fly()
        model = model.to(device=device)
        model = model.eval()

    torch.cuda.synchronize()
    print(f"Time to load model: {time.time() - t0:.02f} seconds")


    torch.manual_seed(seed)
    '''
    model_size = sum(
        [
            p.numel() * p.dtype.itemsize
            for p in itertools.chain(model.parameters(), model.buffers())
        ]
    )

    assert not (compile and default_compile), "Cannot compile with both modes"

    if compile or default_compile:
        global decode_one_token

    if compile:
        decode_one_token = torch.compile(
            decode_one_token, mode="reduce-overhead", fullgraph=True
        )

    if default_compile:
        decode_one_token = torch.compile(
            decode_one_token, mode="default", fullgraph=True
        )'''

    '''
    with open(prompt_file, "r") as f:
        prompts = json.load(f)

    # sort prompts by length to minimize padding

    prompt_idx = list(range(len(prompts)))

    assert "idx" not in prompts[0], "Prompts already have idx field"

    if "prompt" in prompts[0]:
        prompts = [
            {"idx": idx, "prompt": prompt["prompt"]}
            for idx, prompt in zip(prompt_idx, prompts)
        ]
    elif "input" in prompts[0]:
        prompts = [
            {"idx": idx, "prompt": prompt["input"]}
            for idx, prompt in zip(prompt_idx, prompts)
        ]
    else:
        raise ValueError("Prompts must have either prompt or input field")

    print("Tokenizing prompts ...")
    all_prompts = [prompt["prompt"] for prompt in prompts]
    tokenized_full_seq = tokenizer.batch_encode(
        all_prompts, bos=[False] * len(all_prompts), eos=[False] * len(all_prompts)
    )

    for prompt, tokenized in zip(prompts, tokenized_full_seq):
        prompt["prompt_len"] = len(tokenized)

    prompts = sorted(prompts, key=lambda x: x["prompt_len"])

    num_sample_prompts = []
    for prompt in prompts:
        for i in range(num_samples):
            sample_prompt = deepcopy(prompt)
            sample_prompt["sample_idx"] = i
            num_sample_prompts.append(sample_prompt)
    prompts = num_sample_prompts

    skipped_prompt_ids = dict()

    if rank == 0 or not use_tp:
        output_parent = output_file.parent
        if not output_parent.is_dir():
            output_parent.mkdir(exist_ok=True, parents=True)

    
    if use_tp:
        torch.distributed.barrier()

    if resume_generation and os.path.isfile(output_file):
        with open(output_file, "r") as f:
            for line in f:
                sample = json.loads(line)
                if sample["idx"] not in skipped_prompt_ids:
                    skipped_prompt_ids[sample["idx"]] = 0
                skipped_prompt_ids[sample["idx"]] += 1

    # prompts = [prompt for prompt in prompts if prompt["idx"] not in skipped_prompt_ids]
    new_prompts = []
    for prompt in prompts:
        if prompt["idx"] not in skipped_prompt_ids:
            new_prompts.append(prompt)
        else:
            skipped_prompt_ids[prompt["idx"]] -= 1
            if skipped_prompt_ids[prompt["idx"]] == 0:
                del skipped_prompt_ids[prompt["idx"]]
    prompts = new_prompts

    while len(prompts) % batch_size != 0:
        prompts.insert(0, prompts[0])

    model = AutoModelForCausalLM.from_pretrained("/localdata_ssd/Lean/checkpoints/internlm/internlm2-math-base-7b", trust_remote_code=True)
    model = model.to(device=device)
    model = model.eval()
    '''
    
    
    dp_rank = 0#get_data_parallel_rank()
    tp_rank = 0#get_model_parallel_rank()

    dp_size = 0#get_data_parallel_world_size()

    #if tp_rank == 0:
    #    output_writer = open(output_file, "a")

    batch_idx = 0

    gc.collect()
    torch.cuda.empty_cache()


    #if compile:
    #    remove_all_backward_hooks(model)
        
    output_dir = make_output_dir(args.output_dir)

    repo, data = _load_data(args.dataset_name, args.dataset_path)
    shard_size = len(data) // args.num_shards
    #import random
    #random.seed(1926)
    #random.shuffle(data)
    data = data[args.shard*shard_size:(args.shard+1)*shard_size] if args.num_shards > 1+ args.shard else data[args.shard*shard_size:]
    #data = data[(1690+850+1440+1000+800):]
    #data = data[(9000):]
    print("Shard size: %d" % (len(data)))


    start = time.time()
    results = []
    for example in tqdm(data, total=len(data)):
        file_path = example['file_path']
        theorem_name = example['full_name']
        theorem = Theorem(repo, file_path, theorem_name)
        for _ in range(1):
            attempt_results = best_first_search(
                theorem, model, tokenizer,
                max_iters=args.max_iters,
                temperatures=temperature,
                num_samples=args.num_samples,
                batch_size=batch_size,
                timeout=args.timeout,
                early_stop=args.early_stop,
                top_k=args.top_k
            )
            if any([x['success'] for x in attempt_results]):
                break

        result = {
            'attempt_results': attempt_results,
            'success': any([x['success'] for x in attempt_results]),
            'example': example
        }

        results.append(result)

        _save(
            model_name="internLM-7b-math",
            results=results,
            args_dict={},
            output_dir=output_dir,
            shard=args.shard
        )
        print_stats(results)
        # The proof search occasionally leaves Lean processes open. As a workaround,
        # we periodically kill all Lean processes. Note that this may cause a proof search failure.
        if args.shard == 0:
            hours = 60*60*args.clear_process_hours
            if time.time() - start > hours:
                print("=== Killing active leanprover processes to mitigate leak")
                os.system("ps aux | grep leanprover | awk '{print $2}' | xargs kill -9")

    '''
    for batched_prompt_idx in range(0, len(prompts), batch_size):
        batch_idx += 1
        if batch_idx % dp_size != dp_rank:
        if batch_idx % dp_size != dp_rank:
            continue

        batched_prompts = prompts[batched_prompt_idx : batched_prompt_idx + batch_size]

        encoded, left_pad_mask_pos = batch_encode_tokens(
            tokenizer, [_["prompt"] for _ in batched_prompts], bos=True, device=device
        )
        prompt_length = encoded.size(1)

        # torch.cuda.synchronize()
        t0 = time.perf_counter()

        model_max_length = model.config.block_size

        if max_new_tokens + prompt_length >= model_max_length:
            max_new_tokens = model_max_length - prompt_length - 1

        y, num_decoded_tokens = generate(
            model,
            encoded,
            left_pad_mask_pos,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=tokenizer.eos_id,
            max_seq_len=max_seq_len,
        )

        full_y_list = y.tolist()
        print(post_process(full_y_list[0], tokenizer))
        print()

        # torch.cuda.synchronize()
        t = time.perf_counter() - t0
        tokens_generated = num_decoded_tokens * y.size(0)
        tokens_sec = tokens_generated / t
        print(f"Prompt length: {prompt_length}")
        print(
            f"Time for inference {batched_prompt_idx + batch_size} / {len(prompts)}"
            f": {t:.02f} sec total, {tokens_sec:.02f} tokens/sec"
        )
        print(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")

        outputs = []

        for y_list in full_y_list:
            output = post_process(y_list[prompt_length:], tokenizer)
            outputs.append(output)

        if tp_rank == 0:
            fcntl.flock(output_writer, fcntl.LOCK_EX)
            try:
                for prompt, output in zip(batched_prompts, outputs):
                    output_writer.write(
                        json.dumps(
                            {
                                "idx": prompt["idx"],
                                "sample_idx": prompt["sample_idx"],
                                "prompt": prompt["prompt"],
                                "output": output,
                            }
                        )
                        + "\n"
                    )
                output_writer.flush()
            finally:
                fcntl.flock(output_writer, fcntl.LOCK_UN)
    '''


def post_process(y_list, tokenizer):
    y_list = y_list[:]
    if tokenizer.eos_id in y_list:
        y_list = y_list[: y_list.index(tokenizer.eos_id)]

    if tokenizer.pad_id in y_list:
        y_list = y_list[::-1]
        y_list = y_list[: y_list.index(tokenizer.pad_id)]
        y_list = y_list[::-1]
        

    return tokenizer.decode(y_list)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Your CLI description.")

    parser.add_argument(
        "--seed", type=int, default=1926, help="Random seed for reproducibility."
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for sampling."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Whether to compile the model."
    )
    parser.add_argument(
        "--default_compile",
        action="store_true",
        help="Whether to compile the model with default settings.",
    )
    parser.add_argument(
        "--finetune_checkpoint_path",
        type=Path,
        default=None,
        help="Finetune checkpoint path.",
    )

    parser.add_argument(
        "--finetune_checkpoint_prefix",
        type=str,
        default=None,
        help="Finetune checkpoint prefix.",
    )


    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="Size of tensor parallelism.",
    )

    parser.add_argument(
        "--on_the_fly_8bit_quantization",
        action="store_true",
        help="Whether to quantize after loading the model.",
    )

    
    parser.add_argument(
        '--dataset-name',
        default='minif2f-test',
        choices=['minif2f-valid', 'minif2f-test', 'leandojo']
    )
    
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shard-base', type=int, required=True)
    parser.add_argument('--dataset-path', default='data/minif2f.jsonl')
    parser.add_argument('--output-dir', default='output/minif2f')
    parser.add_argument('--early-stop', action='store_true')
    parser.add_argument('--num-shards', type=int, default=8)
    parser.add_argument('--max-iters', type=int, default=100)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--num-examples', type=int, default=-1)
    parser.add_argument('--num-samples', type=int, default=32)
    parser.add_argument('--clear-process-hours', type=int, default=15)
    parser.add_argument("--top_k", type=int, default=200, help="Top-k for sampling.")
    parser.add_argument('--local-rank', type=int, default=None)
    args = parser.parse_args()
    args.shard = args.shard - args.shard_base
    main(
        args.seed,
        args.batch_size,
        args.num_samples,
        args.temperature,
        args.checkpoint_path,
        args.compile,
        args.default_compile,
        args.finetune_checkpoint_path,
        args.finetune_checkpoint_prefix,
        args.tensor_parallel_size,
        args.on_the_fly_8bit_quantization,
        args
    )
