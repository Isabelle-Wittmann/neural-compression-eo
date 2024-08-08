import os
from typing import Any, Dict
import torch
from torchvision import transforms
from torchgeo.datasets import BigEarthNet

class DictTransforms:
    def __init__(self,dict_transform : dict,):
        self.dict_transform = dict_transform

    def __call__(self, sample):
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
def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def normalize(sample: Dict[str, Any], maxs: torch.Tensor, mins: torch.Tensor) -> Dict[str, Any]:
    """Normalize a single sample from the Dataset."""
    sample = sample.float()
    sample = (sample - mins) / (maxs - mins)
    sample = torch.clip(sample, min=0.0, max=1.0)
    return sample

def init_bigearthnet(cfg_path, bands):
    """
    Initialize the BigEarthNet dataset with specified configuration.
    """
    # Load configuration
    print(os.path)
    cfg = load_config(cfg_path)

    # Get dataset parameters from the configuration
    split = cfg['dataset']['split']
    satellite = cfg['dataset']['satellite'] if 'satellite' in cfg['dataset'] else 's2'
    resize = cfg['dataset']['resize']

    # Get the directory for BigEarthNet data, defaulting to 'data' if not set
    DATA_DIR = os.getenv('DATA_DIR', 'data')
    bigearthnet_dir = os.path.join(DATA_DIR, 'BigEarthNet')

    # Check if the BigEarthNet data directory exists
    assert os.path.isdir(os.path.join(bigearthnet_dir, BigEarthNet.metadata['s2']['directory'])), \
        "Download BigEarthNet with `sh datasets/bigearthnet_download.sh` or specify the DATA_DIR via an env variable."

    # Initialize image transformations
    image_transforms = [
        SelectChannels(bands),          # Select specified bands
        ConvertType(torch.float),       # Convert data type to float
        transforms.Resize(tuple(resize)), # Resize images to specified dimensions
    ]

    # If normalization is specified in the configuration, add it as preprocessing step
    if cfg['dataset']['normalize']:
        # Extract max and min values for the specified bands
        maxs = torch.tensor([cfg['preprocess']['maxs'][i] for i in bands]).unsqueeze(1).unsqueeze(1)
        mins = torch.tensor([cfg['preprocess']['mins'][i] for i in bands]).unsqueeze(1).unsqueeze(1)
        image_transforms.append(lambda sample: normalize(sample, maxs, mins))


    # Wrap the transformations in a dictionary-based transform
    transforms_bigearth = DictTransforms({'image': transforms.Compose(image_transforms)})

    # Initialize the BigEarthNet dataset with the specified parameters and transformations
    dataset = BigEarthNet(
        root=bigearthnet_dir,
        split=split,
        bands=satellite,
        num_classes=cfg['dataset']['num_classes'],
        transforms=transforms_bigearth,
    )

    return dataset, cfg['preprocess']['max_value']