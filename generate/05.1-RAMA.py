# pip install RamachanDraw
# https://github.com/alxdrcirilo/ramachandraw.git

import os
dir_path = os.getcwd()

dir_name = 'random'
pdb_dir_path =  dir_path + f'/Allalpha/relax_pdb/{dir_name}/'
pdb_list = os.listdir(pdb_dir_path)

pdb_path_list = [pdb_dir_path + pdb_name for pdb_name in pdb_list]
pdb_path_list_tmp = []
for pdb_path in pdb_path_list:
    if pdb_path[-3:] != 'pdb':
        continue
    pdb_path_list_tmp.append(pdb_path)

from RamachanDraw import fetch, phi_psi, plot

# Drawing the Ramachandran plot
plot(pdb_path_list_tmp, out='plot_random.png', dpi=300)
