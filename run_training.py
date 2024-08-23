import os
import argparse
import yaml
import torch
from datasets.dataloaders import initialize_dataloaders
from models import *
from utils import *
from training.utils import *
from training.train import *
from training.train_multispectral import *

current_dir = os.path.dirname(os.path.abspath(__file__))

CONFIG = os.path.join(current_dir, 'config.yaml')
CONFIG_DATA = os.path.join(current_dir, 'datasets', 'config_bigearthnet.yaml') 
DATA_DIR = '/dccstor/geofm-finetuning/benediktblumenstiel/similarity-search/data'
MODEL_DIR = os.path.join(current_dir, 'results', 'models')
YAML_ARCHIVE = os.path.join(current_dir, 'results', 'configs')
USED_MODELNAMES = os.path.join(current_dir, 'results', 'modelnames')
BPP_PER_CHANNEL=True


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-v', '--version', type=str)
    args = parser.parse_args()

    if args.version == 'new':
        model_name = get_next_model_name(args.model, USED_MODELNAMES)
        
    else:
        model_name = args.model + '_v' + args.version

    with open(CONFIG, 'r') as f:
        cfg = yaml.safe_load(f)

    os.environ['DATA_DIR'] = DATA_DIR

    set_all_seeds(cfg['randomseed'])
    is_bigearth_data = cfg['dataset']['name'] == 'BigEarthNet' 
    
    data_loader_train, data_loader_test, max_value = initialize_dataloaders(cfg, CONFIG_DATA)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda' :
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:',torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

    try:
        current_model = globals()[args.model](cfg).to(device) #
    except KeyError:
        raise ValueError(f"Unknown model: {args.model}")


    train_net(MODEL_DIR, current_model, data_loader_train, data_loader_test, cfg, device, model_name=model_name)
    
    save_model_name(model_name, USED_MODELNAMES)
    save_config_archive(CONFIG,cfg, model_name, YAML_ARCHIVE)
