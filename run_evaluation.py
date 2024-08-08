import os
import argparse
import yaml
import torch
from datasets.dataloaders import initialize_dataloaders
from evaluation.handcrafted import *
from evaluation.neural import *
from utils import *

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

CONFIG = os.path.join(current_dir, 'config.yaml')
CONFIG_DATA = os.path.join(current_dir, 'datasets', 'config_bigearthnet.yaml') 
DATA_DIR = '/dccstor/geofm-finetuning/similarity-search/data'
RESULTS_CSV = os.path.join(current_dir, 'results', 'results_kodak_detailed.csv')
MODEL_DIR = os.path.join(current_dir, 'results', 'models')
BPP_PER_CHANNEL=True

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-v', '--version', type=str)
    args = parser.parse_args()

    with open(CONFIG, 'r') as f:
        cfg = yaml.safe_load(f)

    os.environ['DATA_DIR'] = DATA_DIR
    set_all_seeds(cfg['randomseed'])
    is_bigearth_data = True if cfg['dataset']['name'] == 'BigEarthNet' else False # Indicator variable for whether big earth net data is used
    
    if cfg['dataloader']['batch_size'] != 1:
        cfg['dataloader']['batch_size'] = 1
        print("Batch size was set to 1 for evaluation")

    data_loader_train, data_loader_test, max_value = initialize_dataloaders(cfg, CONFIG_DATA)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        current_model = globals()[args.model]().to(device)
    except KeyError:
        raise ValueError(f"Unknown model: {args.model}")
    current_model.update(force=True)
    
    model_name = args.model + '_v' + args.version
    
    filename = os.path.join(MODEL_DIR, str(model_name), '.pth.tar')

    if os.path.isfile(filename):
        # current_model.update(force=True)
        checkpoint = torch.load(filename, map_location=device)
        current_model.load_state_dict(checkpoint["state_dict"], strict=False)
    

    tester = Neural_Codec_Tester(data_loader = data_loader_test, 
                                device = device, 
                                max_val = 1,
                                is_bigearth_data = is_bigearth_data,
                                bpp_per_channel = BPP_PER_CHANNEL)

    tester.get_metrics(current_model)
    tester.set_name('Factorized')
    tester.compute_metric_averages()
    tester.write_results_to_csv(RESULTS_CSV)
    tester.flush()

    # tester = Pillow_Codec_Tester(data_loader = data_loader_test, 
    #                         device = device, 
    #                         max_val = 1,
    #                         is_bigearth_data = is_bigearth_data,
    #                         bpp_per_channel = BPP_PER_CHANNEL)
    # for q in [90]:
    #     tester.get_metrics('jpeg', q)
    #     tester.set_name('jpeg')
    #     tester.compute_metric_averages()
    #     tester.write_results_to_csv(RESULTS_CSV)
    #     tester.flush()

  
  