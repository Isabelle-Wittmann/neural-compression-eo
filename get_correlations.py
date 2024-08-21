import os
import argparse
import yaml
import torch
from datasets.dataloaders import initialize_dataloaders
from evaluation.handcrafted import *
from evaluation.neural import *
from evaluation.visualisations import *
from models.compressai_pretrained import *
from models.compressai_based import *
from utils import *


# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

RESULTS_CSV = os.path.join(current_dir, 'results', 'correlations', 'EarthTest')
CONFIG = os.path.join(current_dir, 'config.yaml')
CONFIG_DATA = os.path.join(current_dir, 'datasets', 'config_bigearthnet.yaml') 
DATA_DIR = '/dccstor/geofm-finetuning/benediktblumenstiel/similarity-search/data'
if __name__ == '__main__':

    with open(CONFIG, 'r') as f:
        cfg = yaml.safe_load(f)

    os.environ['DATA_DIR'] = DATA_DIR
    set_all_seeds(cfg['randomseed'])
    is_bigearth_data = cfg['dataset']['name'] == 'BigEarthNet' 
    
    if cfg['dataloader']['batch_size'] != 1:
        cfg['dataloader']['batch_size'] = 1
        print("Batch size was set to 1 for evaluation")

    data_loader_train, data_loader_test, max_value = initialize_dataloaders(cfg, CONFIG_DATA)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tester = Codec_Tester(data_loader = data_loader_train, 
                                device = device, 
                                max_val = 1,
                                is_bigearth_data = is_bigearth_data,
                                bpp_per_channel = None)

    tester.compute_correlation()
        
    path_avg = os.path.join(RESULTS_CSV, 'correlations_avg.csv')
    path_indv = os.path.join(RESULTS_CSV, 'correlations_ind.csv')
    path_avg_png = os.path.join(RESULTS_CSV, 'correlations_avg.png')
    path_indv_png = os.path.join(RESULTS_CSV, 'correlations_ind.png')
    tester.save_correlations(path_avg, path_indv)
    plot_correlation_matrix(path_avg, path_avg_png)

    # plot_per_image_correlations(path_indv, path_indv_png)
    tester.get_summarising_stats()
    # tester.print_image(data_loader_test.dataset[34],RESULTS_CSV,n)
    # tester.print_image_bands_individual(data_loader_test.dataset[145],RESULTS_CSV,n)
    # tester.img_stats(data_loader_test.dataset[1106]['image'])
 

    