import torch
import os
import random
import numpy as np
def set_seed(seed=42):
    random.seed(seed) # python seed
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True

