"""
We use pyrosetta for rosetta operations on protein structures, version pyrosetta=2022.41+release.28dc2a1, 
see https://www.pyrosetta.org/downloads for installation details
"""

# Perform rosetta relax on the predicted protein structures
import os
import numpy as np
from tqdm import tqdm
from pyrosetta import *
import time
import os
dir_path = os.getcwd()

# Initialize pyRosetta with the desired flags and options
pyrosetta.init()
scorefxn =  get_fa_scorefxn()   

# Create a relax object
relax = pyrosetta.rosetta.protocols.relax.FastRelax()
# Set the necessary parameters for RelaxBB
relax.set_scorefxn(pyrosetta.get_fa_scorefxn())
relax.max_iter(20)

data_dir_name = ['prefix', 'finetune', 'pretrain', 'test', 'random']

# Allalpha
pdb_dir = dir_path + '/Allalpha/1.1-pdb/'
relax_pdb_dir = dir_path +  '/Allalpha/'


start_time = time.time()
for dir_name in data_dir_name:
    relax_pdb_name_list=os.listdir(dir_path + f'/Allalpha/1.2-relax_pdb/{dir_name}')
    relax_pdb_name_exit_list = [pdb_name for pdb_name in relax_pdb_name_list]
    pdb_name_list=os.listdir(pdb_dir + dir_name)
    print(dir_name)
    for pdb_name in tqdm(pdb_name_list):
        try:
            pdb_id = pdb_name.split('.')[0]
            print(pdb_id)
            pdb_path = pdb_dir + dir_name + '/' +  pdb_name
            relax_file_name = pdb_id + '_relax.pdb'
            if relax_file_name in relax_pdb_name_exit_list:
                continue
            # Load the input protein structure
            pose = pyrosetta.pose_from_pdb(pdb_path)
            ori_score = scorefxn(pose)   # original score
            # print("original score:", ori_score)
            if not os.getenv("DEBUG"):
                relax.apply(pose)
            releax_score = scorefxn(pose)
            relax_path = relax_pdb_dir + dir_name + '/' + pdb_id + '_relax.pdb'
            pose.dump_pdb(relax_path)
            # print("relaex score:", releax_score)
        except:
            pass
            continue
    print(dir_name + " finish!")
end_time = time.time()
total_time = end_time - start_time
print("times: ", total_time)


