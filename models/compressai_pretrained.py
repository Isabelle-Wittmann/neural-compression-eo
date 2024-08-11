import compressai
from compressai.zoo import (bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor) 


def Bmshj2018_factorized(cfg):
     return bmshj2018_factorized(quality=cfg['compressai_model']['quality'], pretrained=cfg['compressai_model']['pretrained']).eval()

def Bmshj2018_hyperprior(cfg):
     return bmshj2018_hyperprior(quality=cfg['compressai_model']['quality'], pretrained=cfg['compressai_model']['pretrained']).eval()

