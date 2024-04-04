import torch
print(torch.__version__)
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
# from tqdm import tqdm
# import torch
# tp_model_list = []
# tp_model_list2 = []
# tp_model_list3 = []
# for i in tqdm(range(1)):
#     ckpt = f'/lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/llemma_7b_6e-6_metamath_mapped/epoch_4_step_4_rank_{i}.pt'
#     tp_model_list.append(torch.load(ckpt)['model'])
#     print(torch.load(ckpt)['epoch'], torch.load(ckpt)['global_step'])
# for i in tqdm(range(1)):
#     ckpt = f'/lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/llemma_7b_6e-6_metamath_mapped/epoch_5_step_5_rank_{i}.pt'
#     tp_model_list2.append(torch.load(ckpt)['model'])
#     print(torch.load(ckpt)['epoch'], torch.load(ckpt)['global_step'])
    
# # for i in tqdm(range(1)):
# #     ckpt = f'/lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/llemma_7b_8e-6_metamath_mapped/epoch_1_step_58_rank_{i}.pt'
# #     tp_model_list3.append(torch.load(ckpt)['model'])
# #     print(torch.load(ckpt)['epoch'], torch.load(ckpt)['global_step'])
    
# for key, para in tp_model_list[0].items():
#     print(tp_model_list[0][key][0])
#     print('#######################')
#     print(tp_model_list2[0][key][0])
#     print('************************')
#     # print(tp_model_list3[0][key][0])
#     # print('************************')

