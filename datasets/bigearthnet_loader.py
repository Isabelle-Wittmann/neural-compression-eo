import os
import argparse
import yaml
import attrs as attr
import torch
from torchvision import transforms
from torchgeo.datasets import BigEarthNet
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Any, Dict, Optional

class DictTransforms:
    def __init__(self,
                 dict_transform : dict,
                 ):
        self.dict_transform = dict_transform

    def __call__(self, sample):
        # Apply your transforms to the 'image' key
        for key, function in self.dict_transform.items():
            sample[key] = function(sample[key])
        return sample

class SelectChannels:
    def __init__(self, channels):
        self.channels = channels

    def __call__(self, tensor):
        return tensor[self.channels]

class Unsqueeze:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, tensor):
        return tensor.unsqueeze(dim=self.dim)

class ConvertType:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, tensor):
        return tensor.to(self.dtype)


def preprocess(sample: Dict[str, Any]) -> Dict[str, Any]:
        maxs = torch.tensor(
        [
            10000.0,
            10000.0,
            10000.0,
            # 10000.0,
            # 10000.0,
            # 10000.0,
            # 10000.0,
            # 10000.0,
            # 10000.0,
            # 10000.0,
            # 10000.0,
            # 10000.0,
           
        ]
        ).unsqueeze(1).unsqueeze(1)
        mins = torch.tensor(
            [
                0.0,
                0.0,
                0.0,
                # 0.0,
                # 0.0,
                # 0.0,
                # 0.0,
                # 0.0,
                # 0.0,
                # 0.0,
                # 0.0,
                # 0.0,
            
            ]
        ).unsqueeze(1).unsqueeze(1)
        """Transform a single sample from the Dataset."""
       
        sample = sample.float()
        sample = (sample - mins) / (maxs - mins)
        sample = torch.clip(sample, min=0.0, max=1.0)
        return sample

def init_bigearthnet(cfg, *args, **kwargs):
    """
    Init BigEarthNet dataset, with S2 data and 43 classes as default.
    """
    # Get dataset parameters
    split = cfg['dataset']['split']
    satellite = cfg['dataset']['satellite'] if 'satellite' in cfg['dataset'] else 's2'

    # Get BigEarthNet directory
    DATA_DIR = os.getenv('DATA_DIR', 'data')
    bigearthnet_dir = os.path.join(DATA_DIR, 'BigEarthNet')

    # Check if data is downloaded
    assert os.path.isdir(os.path.join(bigearthnet_dir, BigEarthNet.metadata['s2']['directory'])), \
        "Download BigEarthNet with `sh datasets/bigearthnet_download.sh` or specify the DATA_DIR via a env variable."

    # Init transforms
    image_transforms = [
        SelectChannels(cfg['dataset']['bands']),
        ConvertType(torch.float),
        transforms.Resize((128,128)),
    ]

    if cfg['dataset']['normalize']:
      # Normalize images
      
        image_transforms.append(preprocess)
        #image_transforms.append(Unsqueeze(dim=1))  # add time dim

    transforms_bigearth = DictTransforms({'image': transforms.Compose(image_transforms)})

    # Init dataset
    dataset = BigEarthNet(
        root=bigearthnet_dir,
        split=split,
        bands=satellite,
        num_classes=cfg['dataset']['num_classes'],
        transforms=transforms_bigearth,
    )
    print("Success")
    print(dataset[1])
    print("Success2")


    return dataset


# OLD
# def load_dataset(cfg):
#     # load settings from cfg
#     dataset_name = cfg['dataset']['name']
#     logging.info(f"Load dataset {dataset_name} ({cfg['dataset']['split']} split)")
#     assert dataset_name in DATASET_REGISTRY, (f"Dataset {dataset_name} not registered. "
#                                               f"Select a dataset from {DATASET_REGISTRY.keys()} ")
#     # get dataset fc from registry
#     dataset_fn = DATASET_REGISTRY[dataset_name]
#     # load dataset
#     dataset = dataset_fn(cfg)
#     return dataset

