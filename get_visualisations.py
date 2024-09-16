import argparse
import os
import yaml
from evaluation.visualisations import evaluate_and_visualize_results
from utils import *

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the current script directory
RESULTS_CSV = os.path.join(current_dir, 'results', 'results_bottleneck.csv')
OUTPUT_DIR = os.path.join(current_dir, 'visualisations', 'bottleneck')

MODEL_LIST = ['FactorizedPrior_8', 'FactorizedPrior_16', 'FactorizedPrior_32', 'FactorizedPrior_64', 'FactorizedPrior_320'
              ]

if __name__ == '__main__':

    evaluate_and_visualize_results(model_names=MODEL_LIST,csv_path=RESULTS_CSV, output_dir=OUTPUT_DIR)

