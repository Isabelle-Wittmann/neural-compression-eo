import argparse
import os
import yaml
from evaluation.visualisations import evaluate_and_visualize_results
from utils import *

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the current script directory
RESULTS_CSV = os.path.join(current_dir, 'results', 'results_bigearth_RGB.csv')
OUTPUT_DIR = os.path.join(current_dir, 'visualisations', 'bigearth_RGB')
MODEL_LIST = ['jpeg','Bmshj2018_factorized_v_pt', 'Bmshj2018_hyperprior_v_pt', 'FactorizedPrior_v2', 'ScaleHyperprior_v0', 'ScaleHyperprior_meta_v0', 'ScaleHyperprior_meta_v1', 'ScaleHyperprior_crs_only_v0']

if __name__ == '__main__':

    evaluate_and_visualize_results(model_names=MODEL_LIST,csv_path=RESULTS_CSV, output_dir=OUTPUT_DIR)

