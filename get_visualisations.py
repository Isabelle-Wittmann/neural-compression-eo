import argparse
import os
import yaml
from evaluation.visualisations import evaluate_and_visualize_results
from utils import *

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the current script directory
RESULTS_CSV = os.path.join(current_dir, 'results', 'results_bigearth_RGB new.csv')
OUTPUT_DIR = os.path.join(current_dir, 'visualisations', 'metadata')

MODEL_LIST = [
    'jpeg', 'Bmshj2018_factorized_pretrained', 'FactorizedPrior',  'ScaleHyperprior_0001',  'ScaleHyperpriorCRSCatLater', 'ScaleHyperpriorCRSAddLater']

if __name__ == '__main__':

    evaluate_and_visualize_results(model_names=MODEL_LIST,csv_path=RESULTS_CSV, output_dir=OUTPUT_DIR)

