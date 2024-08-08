import math
import csv
import os
import torch
import numpy as np
from statistics import fmean, variance

class Codec_Tester():
    def __init__(self, data_loader, device, max_val, is_bigearth_data, bpp_per_channel):
        self.name = 'Undefined'
        self.dataloader = data_loader
        self.is_bigearth_data = is_bigearth_data 
        self.device = device
        self.bpp_per_channel = bpp_per_channel
        self.max_val = max_val
        self.psnr_all = []
        self.mse_all = []
        self.bpp_all = []
        self.bpp_est_all = []
        self.psnr_per_band_all = {}
        self.mse_per_band_all = {}

        self.set_dims(next(iter(data_loader))['image'] if self.is_bigearth_data else next(iter(data_loader))[0])
        
        for band_index in range(self.bands):
            self.psnr_per_band_all[band_index] = []
            self.mse_per_band_all[band_index] = []

        self.num_pixels = self.height * self.width 
        self.num_pixels = [self.num_pixels * self.bands if self.bpp_per_channel else self.num_pixels][0]

    def compute_rate_metrics(self):
        raise NotImplementedError()
    
    def set_dims(self, image):
        image = image.squeeze()
        dims = image.shape

        if len(dims) != 3:
            raise ValueError("Image must have 3 dimensions (bands, height, width)")

        # Initialize placeholders for band, height, and width values
        self.height = None
        
        # Determine which dimension is which
        for i, dim in enumerate(dims):
            if 1 <= dim <= 20:
                self.bands = dim
                self.bands_dim = i
            elif  self.height is None or dim > self.height:
                self.height = dim
                self.height_dim = i
            else:
                self.width = dim
                self.width_dim = i

        # Debugging prints (can be removed or replaced with logging)
        print(f"Bands: {self.bands}, Dimension: {self.bands_dim}")
        print(f"Height: {self.height}, Dimension: {self.height_dim}")
        print(f"Width: {self.width}, Dimension: {self.width_dim}")

    def compute_distortion_metrics(self, x: torch.Tensor, y = torch.Tensor):

        max_input = x.max().item()
        max_output = y.max().item()
    
        if max(max_input, max_output) > 10 * min(max_input, max_output):
            print("Flag: Max values of inputs and outputs are not within a factor of 10 range.")
    
        mse = (x - y).pow(2).mean()
        psnr = 20 * math.log10(self.max_val) - 10 * torch.log10(mse)

        mse_per_band = ((x - y) ** 2).mean(dim=[self.height_dim, self.width_dim], keepdim=True) 
        psnr_per_band = 20 * math.log10(self.max_val) - 10 * torch.log10(mse_per_band)

        self.psnr_all += [psnr.item()]
        self.mse_all += [mse.item()]

        for band_index in range(self.bands):

            self.psnr_per_band_all[band_index] += [psnr_per_band[band_index]]
            self.mse_per_band_all[band_index] += [mse_per_band[band_index]]

    def compute_metric_averages(self):

        self.psnr_avg = fmean(self.psnr_all)
        self.bpp_avg = fmean(self.bpp_all)
        self.bpp_avg_est = fmean(self.bpp_est_all)
        self.mse_avg = fmean(self.mse_all)

        self.psnr_variance = variance(self.psnr_all)
        self.bpp_variance = variance(self.bpp_all)
        self.mse_variance = variance(self.mse_all)

        self.psnr_band_avg = {band_index: fmean(self.psnr_per_band_all[band_index]) for band_index in self.psnr_per_band_all}
        self.mse_band_avg = {band_index: fmean(self.mse_per_band_all[band_index]) for band_index in self.mse_per_band_all}
    
    def set_name(self, name):
        self.name = name
    
    def write_results_to_csv(self, csv_dir):
        # Check if the file exists
        file_exists = os.path.isfile(csv_dir)
        # Open the file in append mode, create if not exists
        with open(csv_dir, 'a+', newline='') as csvfile:
            csvfile.seek(0)  # Move to the start of the file
            first_char = csvfile.read(1)  # Read the first character
            csvfile.seek(0)  # Reset the position to the start
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            
            # Write header only if the file is new or empty
            if not file_exists or first_char == '':
                header = [
                    'name', 'psnr_avg', 'psnr_variance', 'bpp_avg_est', 'bpp_avg', 'bpp_variance', 'mse_avg', 'mse_variance'
                ] + [f'psnr_band_{i}_avg' for i in range(len(self.psnr_band_avg))] + [f'mse_band_{i}_avg' for i in range(len(self.mse_band_avg))] + ['psnr_all', 'bpp_all']
                writer.writerow(header)
            
            # Write the data row
            row = [
                self.name, self.psnr_avg, self.psnr_variance, self.bpp_avg_est, self.bpp_avg, self.bpp_variance, 
                self.mse_avg, self.mse_variance
            ] + [self.psnr_band_avg[i] for i in range(len(self.psnr_band_avg))] + [self.mse_band_avg[i] for i in range(len(self.mse_band_avg))] + [self.psnr_all, self.bpp_all]
            writer.writerow(row)


    def convert_to_8bit(self, image):
        return np.clip((image / self.max_val) * 255.0, 0, 255).astype(np.uint8)
    
    def img_stats(self, image):
        print(f"  Mean: {image.mean()}")
        print(f"  Std Dev: {image.std()}")
        print(f"  Min: {image.min()}")
        print(f"  Max: {image.max()}")

    def save_sample_reconstruction(self, index):
        raise NotImplementedError()

    def flush(self):
        self.psnr_all = []
        self.mse_all = []
        self.bpp_all = []
        self.bpp_est_all = []
        self.psnr_per_band_all = {}
        self.mse_per_band_all = {}

        for band_index in range(self.bands):
            self.psnr_per_band_all[band_index] = []
            self.mse_per_band_all[band_index] = []