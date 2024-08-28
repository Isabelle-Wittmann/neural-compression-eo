import yaml
import os 
import torch.nn as nn
from compressai.optimizers import net_aux_optimizer

def save_config_archive(config_file, cfg, model_name, archive_dir):

    # Ensure the archive directory exists
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)

    # Construct the archive file name
    base_name = os.path.basename(config_file)
    base_archive_file = os.path.join(archive_dir, f"{model_name}_{base_name}")
    archive_file = base_archive_file

    # counter = 1
    # while os.path.exists(archive_file):
    #     archive_file = f"{base_archive_file.split('.yaml')[0]}_{counter}.yaml"
    #     counter += 1

    # # Save a copy of the configuration file in the archive directory
    # with open(archive_file, 'w') as f:
    #     yaml.safe_dump(cfg, f)
        # Check if the archive file already exists
    if os.path.exists(archive_file):
        # Read the existing content
        with open(archive_file, 'r') as f:
            existing_cfg = yaml.safe_load(f)
        
        # Append the new content to the existing content
        if isinstance(existing_cfg, list):
            existing_cfg.append('------------------')
            existing_cfg.append('New configurations')
            existing_cfg.append('------------------')
            existing_cfg.append(cfg)
        else:
            existing_cfg = [existing_cfg, cfg]
        
        # Save the updated content back to the same file
        with open(archive_file, 'w') as f:
            yaml.safe_dump(existing_cfg, f)
    else:
        # Save the new configuration file in the archive directory
        with open(archive_file, 'w') as f:
            yaml.safe_dump(cfg, f)

    print(f"Configuration file saved to {archive_file}")
    
def get_next_model_name(model_name, used_names_file):
    existing_model_numbers = set()
    
    # Read existing model numbers from the file
    if os.path.exists(used_names_file):
        with open(used_names_file, 'r') as f:
            for line in f:
                name, number = line.strip().rsplit('_v', 1)
                if name == model_name and number.isdigit():
                    existing_model_numbers.add(int(number))

    # Find the next available number
    counter = 0
    while counter in existing_model_numbers:
        counter += 1

    return f"{model_name}_v{counter}"

def save_model_name(model_name, used_names_file):
    with open(used_names_file, 'a') as f:
        f.write(f"{model_name}\n")

# From compressai
class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# From compressai
def configure_optimizers(net, optimizer, optimizer_aux, learning_rate, aux_learning_rate):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": optimizer, "lr": learning_rate},
        "aux": {"type": optimizer_aux, "lr": aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]

# From compressai
class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)
