import compressai
from compressai.zoo import (bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor, cheng2020_attn) 


def Bmshj2018_factorized(cfg):
     return bmshj2018_factorized(quality=cfg['compressai_model']['quality'], pretrained=cfg['compressai_model']['pretrained']).eval()

def Bmshj2018_hyperprior(cfg):
     return bmshj2018_hyperprior(quality=cfg['compressai_model']['quality'], pretrained=cfg['compressai_model']['pretrained']).eval()

def Mbt2018_mean(cfg):
     return mbt2018_mean(quality=cfg['compressai_model']['quality'], pretrained=cfg['compressai_model']['pretrained']).eval()

def Mbt2018(cfg):
     return mbt2018(quality=cfg['compressai_model']['quality'], pretrained=cfg['compressai_model']['pretrained']).eval()

def Cheng2020_anchor(cfg):
     return cheng2020_anchor(quality=cfg['compressai_model']['quality'], pretrained=cfg['compressai_model']['pretrained']).eval()

# def Cheng2020_attn(cfg):
#      return cheng2020_attn(quality=cfg['compressai_model']['quality'], pretrained=cfg['compressai_model']['pretrained']).eval()

def Bmshj2018_factorized_ms_ssim(cfg):
     return bmshj2018_factorized(quality=cfg['compressai_model']['quality'], metric ='ms-ssim', pretrained=cfg['compressai_model']['pretrained']).eval()

def Bmshj2018_hyperprior_ms_ssim(cfg):
     return bmshj2018_hyperprior(quality=cfg['compressai_model']['quality'], metric ='ms-ssim', pretrained=cfg['compressai_model']['pretrained']).eval()

def Mbt2018_mean_ms_ssim(cfg):
     return mbt2018_mean(quality=cfg['compressai_model']['quality'], metric ='ms-ssim', pretrained=cfg['compressai_model']['pretrained']).eval()

def Mbt2018_ms_ssim(cfg):
     return mbt2018(quality=cfg['compressai_model']['quality'], metric ='ms-ssim', pretrained=cfg['compressai_model']['pretrained']).eval()

def Cheng2020_anchor_ms_ssim(cfg):
     return cheng2020_anchor(quality=cfg['compressai_model']['quality'], metric ='ms-ssim', pretrained=cfg['compressai_model']['pretrained']).eval()

def Cheng2020_attn_ms_ssim(cfg):
     return cheng2020_attn(quality=cfg['compressai_model']['quality'], metric ='ms-ssim', pretrained=cfg['compressai_model']['pretrained']).eval()
