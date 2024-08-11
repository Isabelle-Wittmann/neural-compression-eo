import random
import numpy as np
import torch
import os
import sys
import logging

def set_all_seeds(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True


def load_data(d, dataset_name: str, device: torch.device) -> torch.Tensor:

    if dataset_name in {'ImageNet', 'Kodak'} or dataset_name == False:
        return d[0].to(device), 0, 0, 0, 0
    
    elif dataset_name == 'BigEarthNet' or dataset_name == True:

        image = d['image'].to(device)
        label = d['label'].to(device).float()
        crs = d['crs'].to(device)
        date = d['date']
        time = d['time']

        return image, label, crs, date, time
    else:
        logging.error("Unknown dataset")
        sys.exit(0)