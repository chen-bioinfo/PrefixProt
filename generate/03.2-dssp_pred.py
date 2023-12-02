# Get the dssp of the predicted structure and save it in dict format

from Bio.PDB import DSSP, PDBParser
import os
import numpy as np
import random
from tqdm import tqdm

dir_path = os.getcwd()
project_dir_path = os.path.dirname(dir_path)

data_dir_name = ['finetune', 'prefix', 'pretrain', 'random', 'test']
pdb_dir = dir_path + '/Allalpha/1.1-pdb/'


all_data_seq_dssp_dict = {}
for dir_name in data_dir_name:
    all_data_seq_dssp_dict[dir_name] = []
    
for dir_name in data_dir_name:
    pdb_name_list=os.listdir(pdb_dir + dir_name)
    print(dir_name)
    for pdb_name in tqdm(pdb_name_list):
        try:
            pdb_path = pdb_dir + dir_name + '/' +  pdb_name
            p = PDBParser()
            structure = p.get_structure("Model", pdb_path)
            model = structure[0]
            dssp = DSSP(model, pdb_path)

            dssp_count = {}
            for row in dssp:
                dssp_info = row[1:3]
                aa, aa_dssp = dssp_info[0], dssp_info[1]
                if aa_dssp not in dssp_count.keys():
                    dssp_count[aa_dssp] = 1
                else:
                    dssp_count[aa_dssp] += 1
            all_data_seq_dssp_dict[dir_name].append(dssp_count)
        except:
            pass
            continue
    print(dir_name + " finish!")
np.save(dir_path + '/Allalpha/all_data_seq_dssp_dict.npy', all_data_seq_dssp_dict)