"""
We use esmfold to perform structure prediction on the generated sequences. 
esmfold installation method and usage details are available at https://github.com/facebookresearch/esm
"""

import numpy as np
import pandas as pd
import os
dir_path = os.getcwd()
project_dir_path = os.path.dirname(dir_path)

aa_list = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
def clean_seq_set(seq_list):
    new_seq_list = set()
    for i, seq in enumerate(seq_list):   
        seq = seq.replace('\n', '').replace('<|endoftext|>', '')
        seq = seq.replace(' ', '').upper()
        flag = False
        if len(seq) == 0:
            continue
        # if len(seq) > 100:   # Removing sequences longer than 100 when processing AMP sequences
        #     continue
        for aa in seq:
            if aa not in aa_list:
                flag = True
                break
        if flag:
            continue
        else:
            new_seq_list.add(seq)
    return list(new_seq_list)

# All alpha protein
random_seq_list = np.load(dir_path + '/Allalpha/random_seq2000.npy')  # Randomly generated sequences
pretrain_seq_list = np.load(dir_path + '/Allalpha/pretrain_generate.npy')  # Sequences generated by the pretrained model
finetune_seq_list = np.load(dir_path + '/Allalpha/finetune_E100_LR5e-06_BS8_ML512_generate_mxlen500.npy') # Sequences generated by the finetune model
prefix_seq_list = np.load(dir_path + '/Allalpha/prefix_PREFIX_TUNING_CAUSAL_LM_E100_LR0.005_BS8_ML512_VT100_mxlen500.npy')  # Sequences generated by the prefix tuning model
test_data_df = pd.read_csv(project_dir_path + '/data/Allalpha/test_alpha.csv')
test_seq_list = [test_data_df.iloc[i]['Sequence'] for i in range(len(test_data_df))]

random_seq_list =  clean_seq_set(random_seq_list)
pretrain_seq_list = clean_seq_set(pretrain_seq_list)
finetune_seq_list = clean_seq_set(finetune_seq_list)
prefix_seq_list = clean_seq_set(prefix_seq_list)
test_seq_list = clean_seq_set(test_seq_list)


import torch
from tqdm import tqdm
import esm
import os

model = esm.pretrained.esmfold_v1()
model = model.eval().to('cuda:1')

# Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
# Lower sizes will have lower memory requirements at the cost of increased speed.c
# model.set_chunk_size(128)


save_dir = dir_path + '/1.1-pdb/'
def file_check(file_name, data_type):
    temp_file_name = file_name
    i = 1
    while i:
        if os.path.exists(save_dir + f"/{data_type}/" + temp_file_name):
            name, suffix = file_name.split('.')
            name += '(' + str(i) + ')'
            temp_file_name = name+'.'+suffix
            i = i+1
        else:
            return temp_file_name

def pred_seq_fold(data_type, seq_list):
    pred_num = 0
    print(data_type)

    for sequence in tqdm(seq_list, ncols=60):
        file_name = file_check(f'{sequence[:3]}-{sequence[-3:]}-{len(sequence)}.pdb', data_type)
        if len(sequence) > 1000:
            continue
        with torch.no_grad():
            output = model.infer_pdb(sequence)
        with open(save_dir + f"/{data_type}/{file_name}", "w") as f:
            f.write(output)
            pred_num += 1
        # break
    print(f'{data_type} seq predict finish, {pred_num} protein structure are predicted')


pred_seq_fold('random', random_seq_list)
pred_seq_fold('pretrain', pretrain_seq_list)
pred_seq_fold('finetune', finetune_seq_list)
pred_seq_fold('prefix', prefix_seq_list)
pred_seq_fold('test', test_seq_list)

