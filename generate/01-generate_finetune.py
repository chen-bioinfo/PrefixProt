# Sequence generation using fine tuning models

import sys
sys.path.append('../')
from transformers import pipeline
import numpy as np
from utils.set_seed import set_seed
from tqdm import tqdm
import os
set_seed(42)

dir_path = os.getcwd()
project_dir_path = os.path.dirname(dir_path)

pretrain = False
task_type = 'Allalpha'   # Allalpha / AMP
pretrain_path = project_dir_path + '/ProtGPT2'
finetune_model = 'finetune_E100_LR5e-06_BS8_ML512'   #finetune_E100_LR5e-06_BS8_ML512
finetune_path = project_dir_path + f'/cache/{task_type}/finetune/{finetune_model}'

if pretrain == True:
    protgpt2 = pipeline('text-generation', model=pretrain_path, device='cuda:1')
else:
    protgpt2 = pipeline('text-generation', model=finetune_path, device='cuda:1')

# length is expressed in tokens, where each token has an average length of 4 amino acids.
gen_seq_num = 2000
batch_size= 100
seq_list = list()


for i in tqdm(range(int(gen_seq_num/batch_size))):
    sequences = protgpt2("<|endoftext|>", max_length=500, do_sample=True, top_k=500, repetition_penalty=1.2, num_return_sequences=batch_size, eos_token_id=0)  # 3824
    for seq in sequences:
        seq_list.append(seq['generated_text'])
remain_seq_num = gen_seq_num-int(gen_seq_num/batch_size)*batch_size
if remain_seq_num != 0:
    sequences = protgpt2("<|endoftext|>", max_length=500, do_sample=True, top_k=500, repetition_penalty=1.2, num_return_sequences=remain_seq_num, eos_token_id=0)  # 生成剩余序列
    for seq in sequences:
        seq_list.append(seq['generated_text'])

if pretrain == True:
    np.save(dir_path + f'/{task_type}/pretrain_generate.npy', seq_list)
else:
    np.save(dir_path + f'/{task_type}/{finetune_model}_generate_mxlen500.npy' ,seq_list)
