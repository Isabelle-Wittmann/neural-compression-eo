import argparse
import os
import yaml
from evaluation.visualisations import evaluate_and_visualize_results
from utils import *

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the current script directory
RESULTS_CSV = os.path.join(current_dir, 'results', 'newest.csv')
OUTPUT_DIR = os.path.join(current_dir, 'visualisations', 'bottleneck')

MODEL_LIST = [
    'FactorizedPriorEncDec_32_evaluated_on_Portugal_Finland',
    'FactorizedPriorEncDec_32_Portugal_Finland'
    # 'FactorizedPrior_32_filt', 
    #           'FactorizedPrior_8_4_filt',
    #           'FactorizedPrior_8_16_filt',
    #           'FactorizedPriorOne_8_4_filt',
    #           'FactorizedPriorOne_32_filt',
    #           'FactorizedPriorOne_8_16_filt',
            #   'FactorizedPriorEncDec_32_filt',
            #   'FactorizedPriorEncDec_32',
            # #   'FactorizedPriorEncDec_8_16',
            #   'FactorizedPrior_8_64',
            #   'FactorizedPriorCRSDec_8_64',
            #   'FactorizedPriorCRSEnc_8_64',
            #   'FactorizedPrior_16_64',
            #   'FactorizedPriorCRSEnc_16_64',
            #   'FactorizedPriorCRSDec_16_64',
            #   'FactorizedPrior_16_4',
            #   'FactorizedPriorCRSEnc_16_4',
            #   'FactorizedPriorCRSDec_16_4',
            #   'FactorizedPrior_4_4',
            #   'FactorizedPrior_4_32',
            #   'FactorizedPriorCRSDec_4_32',
            #   'FactorizedPriorCRSEnc_4_32',
            # #   'FactorizedPriorCRSEncOne_8_64',
            #   'FactorizedPriorCRSEncOne_16_4',
            #   'FactorizedPriorOne_16_4',




    # 'FactorizedPrior_8','FactorizedPrior_8_1', 
    #           'FactorizedPrior_16',
    #           #'FactorizedPrior_4_32', 
    #           'FactorizedPrior_16_4', 'FactorizedPrior_16_4_1', 
    #           'FactorizedPriorCRSEnc_8', 'FactorizedPriorCRSEnc_8_1',
    #           #'FactorizedPriorCRSEnc_4_32',
    #           'FactorizedPriorCRSEnc_16', 'FactorizedPriorCRSEnc_16_4','FactorizedPriorCRSEnc_16_4_1',
    #           #'FactorizedPriorCRSDec_8', 'FactorizedPriorCRSDec_16','FactorizedPriorCRSDec_16_4',  'FactorizedPriorCRSDec_4_32', 
    #           'FactorizedPrior_320', 'FactorizedPrior_8_new'
              ]

if __name__ == '__main__':

    evaluate_and_visualize_results(model_names=MODEL_LIST,csv_path=RESULTS_CSV, output_dir=OUTPUT_DIR)

