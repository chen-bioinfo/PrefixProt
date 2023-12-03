# PrefixProt
📋Flexible and Controllable Protein Design by Prefix-tuning Large-Scale Protein Language Models

## 📘 Abstract
&nbsp;&nbsp;&nbsp;&nbsp; Designing novel proteins tailored for specific purposes presents a promising approach to address various biomedical challenges, including drug discovery, vaccine design, etc... The protein language models (ProtLMs) with control tags as a prefix or fine-tuning in a specific domain have achieved unprecedented breakthroughs in the controllable design of novel protein functions. However, the vocabulary of protein sequences only contains 20 amino acid residues, which is not like natural language vocabulary to make up flexible control tags. Moreover optimizing ProtLMs effectively with limited data has been challenging due to their large number of parameters. In this study, we propose a flexible and controllable method, named PrefixProt, to prompt the pre-trained ProtLMs for \textit{de novo} protein design with desired structures and functions. PrefixProt employs prefix-tuning to learn a virtual token for each protein property on corresponding datasets. We trained two prefix virtual tokens on alpha-helix structure dataset and antimicrobial peptide (AMP) dataset, respectively. Our results demonstrate that prefix virtual tokens are efficient to prompt the pre-trained ProtLM by optimizing fewer trainable parameters to achieve superior results compared with fine tuning, even under low-data settings. Furthermore, these two prefix virtual tokens are combined to precisely control protein generation with AMP function and alpha-helix structure. These results demonstrate prefix virtual tokens are flexible to be integrated to control the generation of proteins. Besides, during the training process, only the prefix virtual token is trainable, while the pre-trained ProtLM is frozen. Therefore PrefixProt has advantages of both control tags and fine-tuning. In summary, PrefixProt offers a flexible and controllable protein design solution. We anticipate that PrefixProt will contribute to the protein discovery and biomedical advancement.

## 🧬 Model Structure
<div align=center><img src=img/framework1.png></div>

## 🚀 Train
```
# 1. Creating a virtual environment
conda activate prefixprot

# 2. the key elements of 'prefixprot' operating environment are listed below(python==3.8):
transformers==4.28.1
torch==1.9.0+cu111 (You can download it from the pytorch(https://pytorch.org/get-started/previous-versions) )
peft==0.3.0
modlamp==4.3.0
pandas==1.4.0
datasets==2.12.0
numpy==1.23.5
tqdm==4.65.0
matplotlib==3.7.1
seaborn==0.12.2
tensorboard==2.13.0

# 3. Clone this repository
git clone https://github.com/chen-bioinfo/PrefixProt.git
cd PrefixProt

# 4. Download the pre-trained protein language models ProtGPT2 from the link (https://drive.google.com/drive/u/1/folders/1ChYdLqm9Onz7LFJej9gdtANWD2o9MvSn or 
https://huggingface.co/nferruz/ProtGPT2/tree/main); Put it in the 'PrefixProt' folder;

# 5. Train model
cd PrefixProt/Task
python prefix_tuning_prot.py
```

## 🚀 Generate
```
# 1. Download the finetune model and prefix tuning model from the link (https://drive.google.com/drive/u/1/folders/11VVqJHdaoccg9knpNC7uuDRTamxLwpyK); 
Put it in the 'PrefixProt' folder;

# 2. generate sequence
cd PrefixProt/generate
python 01-generate_finetune.py 
python 02-generate_prefix.py

# 3.You can get the predicted structure of the protein generated by the model, as well as the structure after RELAX at the following link 
(https://drive.google.com/drive/u/1/folders/1DEgxBnTY8Ty6V1T1R2qFqzHTRK1Qr3FO); 
Unzip it into the 'PrefixProt/generate/Allalpha/' folder;
```

## 🧐 Analysis
The analysis codes are all located in the folder 'PrefixProt/generate'

| File name | Description |
| ----------- | ----------- |
| 03.1-fold_pred.py      | Predicting protein structure using esmfold       |
| 03.2-dssp_pred.py   | Get secondary structure via dssp tool        |
| 03.3-fold(vis).ipynb     | Visualization of secondary structure distribution       |
| 04.1-rosetta_relax.py     | Relax of protein structures using pyrosetta       |
| 04.2-rosetta_energy.py   | Calculating protein energy after RELAX        |
| 04.3-rosetta(vis).ipynb   | Visualizing the energy distribution of generated proteins        |
| 05.1-RAMA.py   | Get Ramachandran plot       |
| 05.2-RAMA(vis).ipynb   | Visualizing the Ramachandran plot   |
| 06-AMP_analysis.ipynb   | Analysis of generated AMP sequences  |


## ✏️ Citation
If you use this code or our model for your publication, please cite the original paper:
