import numpy as np
import torch 
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from datasets.bigearthnet_loader import init_bigearthnet
from .multiearth_loader import *

def initialize_dataloaders(cfg, cfg_data):
    if cfg['dataset']['name'] == 'BigEarthNet':

        dataset, max_val = init_bigearthnet(cfg_path = cfg_data, bands = cfg['dataset']['bands'])

        train_set =  np.random.choice(len(dataset), cfg['dataset']['subset_size'], replace=False)
        test_set = np.random.choice(len(dataset), cfg['dataset']['subset_size_test'], replace=False)

        dataset_train = Subset(dataset,train_set)
        dataset_test = Subset(dataset, test_set)

    elif cfg['dataset']['name'] == 'MultiEarth':
        
        data_loader_ts =  get_dataloaders()

        return data_loader_ts, 0, 10000
    
    elif cfg['dataset']['name'] == 'ImageNet':

        transform = transforms.Compose([
                    transforms.Resize((128,128)),  
                    transforms.ToTensor()])

        dataset = ImageFolder(root='./dataset/imagenet/ILSVRC2012/training', transform=transform)

        train_set = np.random.choice(len(dataset), cfg['dataset']['subset_size'], replace=False)
        test_set = np.random.choice(len(dataset), cfg['dataset']['subset_size_test'], replace=False)

        dataset_train = Subset(dataset, train_set)
        dataset_test = Subset(dataset, test_set)

        max_val = 255


    elif cfg['dataset']['name'] == 'Kodak':

        transform = transforms.Compose([transforms.ToTensor(), TurnTransformKodak()])

        dataset_train = ImageFolder(root='./u/iwittmann/data/Kodak/', transform=transform)
        dataset_test = ImageFolder(root='./u/iwittmann/data/Kodak/', transform=transform)
        train_set = list(range(0,24))
        dataset=dataset_train

        max_val = 255

    else:
        print("-- Unknown Dataset --")    

    # Init data loader
    data_loader_train = DataLoader(
        dataset_train,
        **cfg['dataloader']
    )

    data_loader_test = DataLoader(
        dataset_test,
        **cfg['dataloader']
    )

    return data_loader_train, data_loader_test, max_val
    

class TurnTransformKodak(torch.nn.Module):
    def forward(self, img):
        if img.size() == torch.Size([3, 512, 768]):
            return torch.rot90(img, 1, [1,2])
        else: 
            return img