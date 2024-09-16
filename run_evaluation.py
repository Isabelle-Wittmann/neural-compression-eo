import os
import argparse
import yaml
import torch
from datasets.dataloaders import initialize_dataloaders
from evaluation.handcrafted import *
from evaluation.neural import *
from models import *
from utils import *


current_dir = os.path.dirname(os.path.abspath(__file__))

RESULTS_CSV = os.path.join(current_dir, 'results', 'results_bottleneck.csv')

CONFIG = os.path.join(current_dir, 'config.yaml')
CONFIG_DATA = os.path.join(current_dir, 'datasets', 'config_bigearthnet.yaml') 
DATA_DIR = '/dccstor/geofm-finetuning/benediktblumenstiel/similarity-search/data'
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
    is_bigearth_data = cfg['dataset']['name'] == 'BigEarthNet' 
    
    if cfg['dataloader']['batch_size'] != 1:
        cfg['dataloader']['batch_size'] = 1
        print("Batch size was set to 1 for evaluation")

    data_loader_train, data_loader_test, data_loader_val, max_value = initialize_dataloaders(cfg, CONFIG_DATA)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        current_model1 = globals()[args.model](cfg).to(device)
    except KeyError:
        raise ValueError(f"Unknown model: {args.model}")
    
    model_name1 = args.model + '_v' + args.version

    filename1 = os.path.join(MODEL_DIR, str(model_name1) +'.pth.tar')

    if os.path.isfile(filename1):
        
        checkpoint = torch.load(filename1, map_location=device)
        with torch.no_grad():
            # Extract and ensure the quantized_cdf is a float32 tensor on the correct device
            quantized_cdf = checkpoint["state_dict"]["entropy_bottleneck._quantized_cdf"].float().to(device)

            # Replace the model's _quantized_cdf directly

            current_model1.entropy_bottleneck._quantized_cdf = quantized_cdf

        
        current_model1.load_state_dict(checkpoint["state_dict"], strict=False)
        current_model1.update(force=True)
        

    tester = NeuralCodecTester(data_loader = data_loader_test, 
                                device = device, 
                                max_val = 1,
                                is_bigearth_data = is_bigearth_data,
                                bpp_per_channel = BPP_PER_CHANNEL)
    # tester.get_summarising_stats()
    tester.get_metrics(current_model1, os.path.join(current_dir, 'visualisations/latents'))
    tester.set_name(model_name1)
    tester.compute_metric_averages()
    tester.write_results_to_csv(RESULTS_CSV)
    # # tester.save_sample_reconstruction(data_loader_test.dataset[145], current_model1, os.path.join(current_dir, 'visualisations/reconstructions', model_name1))
    # tester.flush()


    # tester = Pillow_Codec_Tester(data_loader = data_loader_test, 
    #                         device = device, 
    #                         max_val = 1,
    #                         is_bigearth_data = is_bigearth_data,
    #                         bpp_per_channel = BPP_PER_CHANNEL)
    # for q in [20]:
    #     tester.get_metrics('jpeg', q)
    #     tester.set_name('jpeg')
    #     tester.compute_metric_averages()
    #     tester.write_results_to_csv(RESULTS_CSV)
    #     tester.flush()


 # try:
    #     current_model1 = globals()[args.model](cfg).to(device)
    #     current_model2 = globals()[args.model](cfg).to(device)
    #     current_model3 = globals()[args.model](cfg).to(device)
    #     current_model4 = globals()[args.model](cfg).to(device)
    # except KeyError:
    #     raise ValueError(f"Unknown model: {args.model}")
    
    
    # model_name1 = args.model + '_v' + '2'
    # model_name2 = args.model + '_v' + '3'
    # model_name3 = args.model + '_v' + '4'
    # model_name4 = args.model + '_v' + '5'
    
    # filename1 = os.path.join(MODEL_DIR, str(model_name1) +'.pth.tar')
    # filename2 = os.path.join(MODEL_DIR, str(model_name2) +'.pth.tar')
    # filename3 = os.path.join(MODEL_DIR, str(model_name3) +'.pth.tar')
    # filename4 = os.path.join(MODEL_DIR, str(model_name4) +'.pth.tar')

    # if os.path.isfile(filename1):
        
    #     checkpoint = torch.load(filename1, map_location=device)
    #     current_model1.load_state_dict(checkpoint["state_dict"], strict=False)
    #     # current_model.update(force=True)
    # if os.path.isfile(filename2):
        
    #     checkpoint = torch.load(filename2, map_location=device)
    #     current_model2.load_state_dict(checkpoint["state_dict"], strict=False)
    # if os.path.isfile(filename3):
        
    #     checkpoint = torch.load(filename3, map_location=device)
    #     current_model3.load_state_dict(checkpoint["state_dict"], strict=False)
    # if os.path.isfile(filename4):
        
    #     checkpoint = torch.load(filename4, map_location=device)
    #     current_model4.load_state_dict(checkpoint["state_dict"], strict=False)
    

    # tester = Neural_Codec_Tester_split(data_loader = data_loader_test, 
    #                             device = device, 
    #                             max_val = 1,
    #                             is_bigearth_data = is_bigearth_data,
    #                             bpp_per_channel = BPP_PER_CHANNEL)

    # tester.get_metrics(current_model1,current_model2,current_model3,current_model4)
    # tester.set_name(model_name1)
    # tester.compute_metric_averages()
    # tester.write_results_to_csv(RESULTS_CSV)
    # tester.save_sample_reconstruction(data_loader_test.dataset[145], current_model1, os.path.join(current_dir, 'visualisations/reconstructions', model_name))
    # tester.flush()

  
  