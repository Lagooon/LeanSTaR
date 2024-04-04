from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm
import torch
import re
import argparse
# tp_ckpt_name = '/lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/metamath_llama-7b-395k-1-2-3_epoch-1_lr-3e-5_bs-128_seq-768/epoch_1_step_58_rank_'
# save_name_hf = '/lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/metamath_llama-7b-395k_hf'
# pretrain_name = '/lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/metamath_llama-7b-395k'

def load_and_merge_models(tp_ckpt_name, pretrain_name, save_name_hf, num_tp):
    tp_model_list = []
    for i in tqdm(range(num_tp)):
        ckpt = tp_ckpt_name + f'{i}.pt'
        tp_model_list.append(torch.load(ckpt)['model'])

    state_dict = {}

    for key in tp_model_list[0].keys():
        if "wo" in key or "w2" in key:
            state_dict[key] = torch.cat([tp_model_list[i][key].cpu() for i in range(num_tp)],  dim=1)
        else:
            state_dict[key] = torch.cat([tp_model_list[i][key].cpu() for i in range(num_tp)],  dim=0)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrain_name,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token 
    model = AutoModelForCausalLM.from_pretrained(
        pretrain_name, 
        device_map="auto", 
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
    )
    cpu_state_dict = model.cpu().state_dict()

    pattern = r'layers\.(\d+)\.'

    for key in state_dict.keys():
        match = re.search(pattern, key)
        if match:
            layer = match.group(1)
        if 'wqkv' in key:
            merged_q, merged_k, merged_v = [], [], []
            reconstruct_q, reconstruct_k= [], []
            chunks = torch.split(state_dict[key], split_size_or_sections=1536, dim=0)
            for chunk in chunks:
                q, k, v = chunk.split([512,512,512], dim=0)
                merged_q.append(q)
                merged_k.append(k)
                merged_v.append(v)
            merged_q = torch.cat(merged_q, dim=0)
            merged_k = torch.cat(merged_k, dim=0)
            merged_v = torch.cat(merged_v, dim=0)
            
            #### qk need reconstruction ####
            split_qs = torch.split(merged_q, split_size_or_sections=128, dim=0)
            split_ks = torch.split(merged_k, split_size_or_sections=128, dim=0)
            for split in split_qs:
                matrix0 = split[::2, :]
                matrix1 = split[1::2, :]
                reconstruct_q.append(matrix0)
                reconstruct_q.append(matrix1)
            reconstruct_q = torch.cat(reconstruct_q, dim=0)
            for split in split_ks:
                matrix0 = split[::2, :]
                matrix1 = split[1::2, :]
                reconstruct_k.append(matrix0)
                reconstruct_k.append(matrix1)
            reconstruct_k = torch.cat(reconstruct_k, dim=0)
            #### qk need reconstruction ####
            
            
            name = f'model.layers.{layer}.self_attn.q_proj.weight'
            # print(name, torch.sum(torch.pow(cpu_state_dict[name] - reconstruct_q, 2))) ##check the difference
            cpu_state_dict[name] = reconstruct_q
            
            name = f'model.layers.{layer}.self_attn.k_proj.weight'
            cpu_state_dict[name] = reconstruct_k
            
            name = f'model.layers.{layer}.self_attn.v_proj.weight'
            cpu_state_dict[name] = merged_v

        if 'wo' in key:
            name = f'model.layers.{layer}.self_attn.o_proj.weight'
            cpu_state_dict[name] = state_dict[key]
        if 'w1' in key:
            name = f'model.layers.{layer}.mlp.gate_proj.weight'
            cpu_state_dict[name] = state_dict[key]
        if 'w3' in key:
            name = f'model.layers.{layer}.mlp.up_proj.weight'
            cpu_state_dict[name] = state_dict[key]
        if 'w2' in key:
            name = f'model.layers.{layer}.mlp.down_proj.weight'
            cpu_state_dict[name] = state_dict[key]
            
    model.load_state_dict(cpu_state_dict, strict=False)
    model.save_pretrained(save_name_hf)
    
    config = AutoConfig.from_pretrained(pretrain_name)
    tokenizer.save_pretrained(save_name_hf)
    config.save_pretrained(save_name_hf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--tp_ckpt_name", type=str, help="Path to the TP checkpoint name")
    parser.add_argument("--pretrain_name", type=str, help="Path to the pretrain name")
    parser.add_argument("--save_name_hf", type=str, help="Path to save the HF model")
    parser.add_argument("--num_tp", type=int, default=8, help="Number of TP models")
    
    args = parser.parse_args()
    load_and_merge_models(args.tp_ckpt_name, args.pretrain_name, args.save_name_hf, args.num_tp)
    
# python concat_tp_model_7b.py \
#     --tp_ckpt_name '/lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/metamath_llama-7b-395k-1-2-3_epoch-1_lr-3e-5_bs-128_seq-768/epoch_1_step_58_rank_'
#     --pretrain_name '/lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/metamath_llama-7b-395k'
#     --save_name_hf '/lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/metamath_llama-7b-395k_hf'
