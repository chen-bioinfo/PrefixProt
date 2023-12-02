"""
We use pyrosetta for rosetta operations on protein structures, version pyrosetta=2022.41+release.28dc2a1, 
see https://www.pyrosetta.org/downloads for installation details
"""

# Statistics of rosetta energy scores of protein structures after rosetta relax operation
from pyrosetta import *
import numpy as np
from tqdm import tqdm
import os
dir_path = os.getcwd()


init()
# exploring the pose object
scorefxn =  get_fa_scorefxn()   
data_name = ['prefix', 'finetune', 'pretrain', 'test', 'random']
# data_name = ['finetune']
energy_dict = {}
for data_type in data_name:
    energy_dict[data_type] = []

for data_type in data_name:
    pdb_file_dir_name = dir_path + f'/Allalpha/1.2-relax_pdb/{data_type}/'
    pdb_list = os.listdir(pdb_file_dir_name)
    for pdb_name in tqdm(pdb_list):
        try:
            print(pdb_name)
            pdb_file_path = pdb_file_dir_name + pdb_name
            pose = pyrosetta.pose_from_pdb(pdb_file_path)
            pose_score = scorefxn(pose)
            start_pose_total_score = pyrosetta.rosetta.protocols.relax.get_per_residue_scores(pose, pyrosetta.rosetta.core.scoring.ScoreType.total_score)
            sequence_length = pose.total_residue()
            avg_energy_pre_red = sum(list(start_pose_total_score)) / sequence_length * 1.0
            energy_dict[data_type].append(avg_energy_pre_red)
        except:
            pass
            continue

np.save(dir_path + '/Allalpha/energy_dict.npy', energy_dict)
