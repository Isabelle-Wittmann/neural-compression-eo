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

        out_enc = net.compress(x_padded, label, crs) #
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
        
        _, decompressed = self.forward_pass(image.unsqueeze(0), label, crs, model)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        if (image.max() > 0.4) or (image.max() < 0.2):
            print("Flag: Range of pixel values might not be suitable for visualising, Max is: " + str(image.max()))
        
        if self.bands > 3:
            img = image.squeeze()[1:3]
            img_dec = decompressed.squeeze()[1:3]
        else:
            img = image.squeeze()
            img_dec = decompressed.squeeze()

        img = transforms.ToPILImage()(np.clip(img.cpu().detach()/ 0.3, 0, 1))
        axes[0].imshow(img)
        axes[0].title.set_text('Original')

        img_dec = transforms.ToPILImage()(np.clip(img_dec.cpu().detach()/ 0.3, 0, 1))
        axes[1].imshow(img_dec)
        axes[1].title.set_text('Reconstruction')

        plt.savefig(path + '_sample_reconstruction.png')
        plt.close()

class Neural_Codec_Tester_split(Codec_Tester):

    def compute_bpp(self, out_net):

        size = out_net['x_hat'].size()

        num_pixels = size[0] * size[2] * size[3]

        return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
                for likelihoods in out_net['likelihoods'].values()).item()


    def forward_pass(self, image, label, crs, net1,net2,net3,net4):

        rv1 = net1(image[:,:3,:,:], label, crs)  # Forward pass through the network
        rv2 = net2(image[:,3:6,:,:], label, crs)
        rv3= net3(image[:,6:9,:,:], label, crs)
        rv4= net4(image[:,9:,:,:], label, crs)
        rv_img =torch.cat((rv1['x_hat'] ,rv2['x_hat'] ,rv3['x_hat'] ,rv4['x_hat'] ), dim=1)
        bpp = self.compute_bpp({'x_hat': rv_img, "likelihoods": {"y": torch.cat((rv1['likelihoods']['y'], rv2['likelihoods']['y'],rv3['likelihoods']['y'],rv4['likelihoods']['y']), dim=1)}})
        bpp = [bpp/image.size(1) if self.bpp_per_channel else bpp][0]

        return bpp, rv_img

    def inference(self, image, label, crs, net1,net2,net3,net4):

        pad, unpad = compute_padding(self.height, self.width, min_div=2**6)  # pad to allow 6 strides of 2

        x_padded = F.pad(image, pad, mode="constant", value=0)

        out_enc1 = net1.compress(x_padded[:,:3,:,:], label, crs)
        out_enc2= net2.compress(x_padded[:,3:6,:,:], label, crs)
        out_enc3= net3.compress(x_padded[:,6:9,:,:], label, crs)
        out_enc4 = net4.compress(x_padded[:,9:,:,:], label, crs) #
        # out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
    
        # out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)

        bpp = sum(len(s) for s in out_enc1["strings"][0]) * 8.0 / self.num_pixels
        bpp += sum(len(s) for s in out_enc2["strings"][0]) * 8.0 / self.num_pixels
        bpp += sum(len(s) for s in out_enc3["strings"][0]) * 8.0 / self.num_pixels
        bpp += sum(len(s) for s in out_enc4["strings"][0]) * 8.0 / self.num_pixels

        return bpp

    def get_metrics(self, model1,model2,model3,model4):
        with torch.no_grad():

            for count, data in enumerate(self.dataloader):
                image, label, crs, date, time = load_data(data, self.is_bigearth_data, self.device)
                
                bpp_val_est, decompressed = self.forward_pass(image, label, crs, model1,model2,model3,model4)
                self.bpp_est_all += [bpp_val_est]
                self.bpp_all += [self.inference(image, label, crs,model1,model2,model3,model4)]
                self.compute_distortion_metrics(image.squeeze(), decompressed.squeeze())

                if count % 100 == 0:
                    print(f"Processed {count} images")

    def save_sample_reconstruction(self, image, model, path):

        image, label, crs, date, time = load_data(image, self.is_bigearth_data, self.device)
        
        _, decompressed = self.forward_pass(image.unsqueeze(0), label, crs, model)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        if (image.max() > 0.4) or (image.max() < 0.2):
            print("Flag: Range of pixel values might not be suitable for visualising, Max is: " + str(image.max()))
        
        if self.bands > 3:
            img = image.squeeze()[1:3]
            img_dec = decompressed.squeeze()[1:3]
        else:
            img = image.squeeze()
            img_dec = decompressed.squeeze()

        img = transforms.ToPILImage()(np.clip(img.cpu().detach()/ 0.3, 0, 1))
        axes[0].imshow(img)
        axes[0].title.set_text('Original')

        img_dec = transforms.ToPILImage()(np.clip(img_dec.cpu().detach()/ 0.3, 0, 1))
        axes[1].imshow(img_dec)
        axes[1].title.set_text('Reconstruction')

        plt.savefig(path + '_sample_reconstruction.png')
        plt.close()