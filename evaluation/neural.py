from evaluation.base_class import Codec_Tester
import math
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from compressai.ops import compute_padding
from utils import *

class Neural_Codec_Tester(Codec_Tester):

    def compute_bpp(self, out_net):

        size = out_net['x_hat'].size()

        num_pixels = size[0] * size[2] * size[3]

        return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
                for likelihoods in out_net['likelihoods'].values()).item()


    def forward_pass(self, image, label, crs, net):

        rv = net(image, label, crs)  # Forward pass through the network
        bpp = self.compute_bpp(rv)
        bpp = [bpp/image.size(1) if self.bpp_per_channel else bpp][0]

        return bpp, rv['x_hat'] 

    def inference(self, image, label, crs, net):

        pad, unpad = compute_padding(self.height, self.width, min_div=2**6)  # pad to allow 6 strides of 2

        x_padded = F.pad(image, pad, mode="constant", value=0)

        out_enc = net.compress(x_padded, label, crs)
        out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
    
        out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)

        bpp = sum(len(s) for s in out_enc["strings"][0]) * 8.0 / self.num_pixels

        return bpp

    def get_metrics(self, model):
        with torch.no_grad():

            for count, data in enumerate(self.dataloader):
                image, label, crs, date, time = load_data(data, self.is_bigearth_data, self.device)
                
                bpp_val_est, decompressed = self.forward_pass(image, label, crs, model)
                self.bpp_est_all += [bpp_val_est]
                self.bpp_all += [self.inference(image, label, crs,model)]
                self.compute_distortion_metrics(image.squeeze(), decompressed.squeeze())

                if count % 100 == 0:
                    print(f"Processed {count} images")

    def save_sample_reconstruction(self, image, model, path):

        image, label, crs, date, time = load_data(image, self.is_bigearth_data, self.device)
        
        _, decompressed = self.forward_pass(image, label, crs, model)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        image=image['image']
        
        if (image.max() > 0.4) or (image.max() < 0.2):
            print("Flag: Range of pixel values might not be suitable for visualising, Max is: " + str(image.max()))
        
        if self.bands > 3:
            img = image.squeeze()[1:3]
            img_dec = decompressed.squeeze()[1:3]
        else:
            img = image.squeeze()
            img_dec = decompressed.squeeze()

        img = transforms.ToPILImage()( img/ 0.35)
        axes[0].imshow(img)
        axes[0].title.set_text('Original')

        img_dec = transforms.ToPILImage()(img_dec / 0.35)
        axes[1].imshow(img_dec)
        axes[1].title.set_text('Reconstruction')

        plt.savefig(path + '_sample_reconstruction.png')
        plt.close()
