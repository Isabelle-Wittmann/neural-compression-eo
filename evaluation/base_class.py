import math
import csv
import os
import torch
import numpy as np
import pandas as pd
from statistics import fmean, variance
from utils import *
import matplotlib.pyplot as plt
from torchvision import transforms

class CodecTester():
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
        self.correlation_per_image = []
        self.psnr_per_band_all = {}
        self.mse_per_band_all = {}
        self.latents = []

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
            if 1 <= dim <= 12:
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
            print(max_input)
            print(max_output)

    
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
    
    def write_results_to_csv(self, csv_dir, name):

        file_exists = os.path.isfile(csv_dir)

        with open(csv_dir, 'a+', newline='') as csvfile:
            csvfile.seek(0)  # Move to the start of the file
            first_char = csvfile.read(1)  # Read the first character
            csvfile.seek(0)  # Reset the position to the start
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            
            # Write header only if the file is new or empty
            if not file_exists or first_char == '':
                header = [
                    'name', 'name_org','psnr_avg', 'psnr_variance', 'bpp_avg_est', 'bpp_avg', 'bpp_variance', 'mse_avg', 'mse_variance'
                ] + [f'psnr_band_{i}_avg' for i in range(len(self.psnr_band_avg))] + [f'mse_band_{i}_avg' for i in range(len(self.mse_band_avg))] + ['psnr_all', 'bpp_all']
                writer.writerow(header)
            
            # Write the data row
            row = [
                name, self.name, self.psnr_avg, self.psnr_variance, self.bpp_avg_est, self.bpp_avg, self.bpp_variance, 
                self.mse_avg, self.mse_variance
            ] + [self.psnr_band_avg[i] for i in range(len(self.psnr_band_avg))] + [self.mse_band_avg[i] for i in range(len(self.mse_band_avg))] + [self.psnr_all, self.bpp_all]
            writer.writerow(row)

    def convert_to_8bit(self, image):
        return np.clip((image / self.max_val) * 255.0, 0, 255).astype(np.uint8)

    def img_stats(self, image):

        image = image.cpu().numpy() if torch.is_tensor(image) else image
        
        num_bands = image.shape[0]  # Assuming shape is (bands, height, width)
        
        for i in range(num_bands):
            band = image[i, :, :]
            print(f"Statistics for Band {i+1}:")
            print(f"  Mean: {band.mean():.4f}")
            print(f"  Std Dev: {band.std():.4f}")
            print(f"  Min: {band.min():.4f}")
            print(f"  Max: {band.max():.4f}")
            print("-" * 30)
        
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
        self.correlation_per_image = []
        self.psnr_per_band_all = {}
        self.mse_per_band_all = {}

        for band_index in range(self.bands):
            self.psnr_per_band_all[band_index] = []
            self.mse_per_band_all[band_index] = []

    def compute_correlation(self):
        """
        Compute and save the correlation between the different channels of the images in the dataset.
        """
        print("Computing correlation across the dataset...")
        sum_correlation = None
        count_images = 0
        for count, data in enumerate(self.dataloader):
            print(count)
            input = data['image'] if self.is_bigearth_data else data[0]
            input = input.to(self.device)

            image_np = input.cpu().numpy()
            num_channels = image_np.shape[1] 

            for img in image_np:

                correlations = np.corrcoef(img.reshape(num_channels, -1))
                if np.isnan(correlations).any():
                    print(f"Warning: NaN values encountered in correlation matrix for image {count}.")
                    correlations = np.nan_to_num(correlations)  # Replace NaNs with zeros (or other strategies)

                self.correlation_per_image.append(correlations)

                if sum_correlation is None:
                    sum_correlation = np.zeros_like(correlations)
            
                sum_correlation += correlations
                count_images += 1

        if count % 100 == 0:
            print(f"Processed {count} batches of images for correlation.")
        aggregate_correlation = sum_correlation / count_images

        self.correlation_overall = aggregate_correlation
        print("Finished computing correlations.")

    def save_correlations(self, file_path_agg, file_path_all):
        """
        Save the computed correlations for all band combinations to a CSV file.
        """

        with open(file_path_agg, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            num_channels = self.correlation_overall.shape[0]
            writer.writerow(['Band Combination'] + [f'Band_{i}' for i in range(num_channels)])
            
            # write aggregate correlation matrix.
            for i in range(num_channels):
                writer.writerow([f'Band_{i}'] + self.correlation_overall[i].tolist())
   
            
        with open(file_path_all, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(['Image_Index', 'Band_Combination', 'Correlation'])
            
            # write per-image correlation matrices.
            for idx, correlations in enumerate(self.correlation_per_image):
                num_channels = correlations.shape[0]
                for i in range(num_channels):
                    for j in range(i + 1, num_channels):
                        writer.writerow([idx, f'{i}-{j}', correlations[i, j]])


    def get_summarising_stats(self):
        """
        Compute and print dataset-wide statistics: mean, std dev, min, and max.
        """
        print("Computing dataset-wide statistics...")

        sum_mean = 0
        sum_std = 0
        sum_lat = 0
        sum_lon = 0
        # lat_mean = 13.57538875
        # lon_mean = 51.787555
        global_min, global_lat_min ,global_lon_min= float('inf'), float('inf'), float('inf')
        global_max, global_lat_max ,global_lon_max = float('-inf'), float('-inf'), float('-inf')
        count_batch = 0
        # sum_lat_squared_diff = 0
        # sum_lon_squared_diff = 0

        for count, data in enumerate(self.dataloader):
            count_batch += 1
            image, label, crs, date, time = load_data(data, self.is_bigearth_data, self.device)
            lon = np.array(crs.cpu()[:, 1])
            lat = np.array(crs.cpu()[:, 0])
            lon = lon.to(self.device)
            lat = lat.to(self.device)

            input = image

            batch_mean = input.mean()  # mean per channel
            batch_std = input.std()    # std dev per channel
            batch_min = input.min()
            batch_max = input.max()

            sum_lat += lat
            sum_lon += lon
            global_lat_min = min(global_lat_min, lat.item())
            global_lat_max = max(global_lat_max, lat.item())
            global_lon_min = min(global_lon_min, lon.item())
            global_lon_max = max(global_lon_max, lon.item())

            sum_mean += batch_mean 
            sum_std += batch_std 

            global_min = min(global_min, batch_min.item())
            global_max = max(global_max, batch_max.item())

            sum_lat_squared_diff += (lat - lat_mean) ** 2
            sum_lon_squared_diff += (lon - lon_mean) ** 2

            if count % 1000 == 0:
                print(f"Processed {count} batches for summarizing stats.")

        dataset_mean = sum_mean / count_batch
        lat_mean = sum_lat / count_batch
        lon_mean = sum_lon / count_batch
        dataset_std = sum_std / count_batch

        # lat_std = (sum_lat_squared_diff / count_batch) ** 0.5
        # lon_std = (sum_lon_squared_diff / count_batch) ** 0.5

        print(f"  Dataset-wide Mean: {dataset_mean}")
        print(f"  Dataset-wide Std Dev: {dataset_std}")
        print(f"  Dataset-wide Min: {global_min}")
        print(f"  Dataset-wide Max: {global_max}")
        print(f"  Lat Std: {lat_mean}")
        print(f"  Lon Std: {lon_mean}")
        print("Finished computing dataset-wide statistics.")

            
    def print_image(self, image, path, n):

        image, label, crs, date, time = load_data(image, self.is_bigearth_data, self.device)
        
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))
        
        if (image.max() > 0.4) or (image.max() < 0.2):
            print("Flag: Range of pixel values might not be suitable for visualising, Max is: " + str(image.max()))
        
        if self.bands > 3:
            image=image.squeeze()[1:4]
        else:
            image=image.squeeze()

        img = transforms.ToPILImage()(np.clip(image.cpu()/ 0.2, 0, 1))
        axes.imshow(img)

        plt.savefig(path + '/sample_image_' + str(n) + '.png')
        plt.close()

    def print_image_bands_individual(self, image, path, n):

        image, label, crs, date, time = load_data(image, self.is_bigearth_data, self.device)
        

        image = image.cpu().numpy() if torch.is_tensor(image) else image
        num_bands = image.shape[0]  # Assuming shape is (bands, height, width)
        
        # Create a figure with subplots for each band
        fig, axes = plt.subplots(2, int(num_bands/2), figsize=(26, 8)) 

        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        for i in range(num_bands):
            band = image[i, :, :]
            
            # Flag if pixel values are outside expected range
            if (band.max() > 0.4) or (band.max() < 0.2):
                print(f"Flag: Range of pixel values for band {i+1} might not be suitable for visualizing, Max is: {band.max()}")
            
            # Clip and normalize the band for better visualization

            if i in [5,6,7,8,9,10]:
               band_img = np.clip(band / 0.8, 0, 1) 
            else:
                band_img = np.clip(band / 0.2, 0, 1)
            
            # Convert to PIL image
            img = transforms.ToPILImage()(band_img)
            
            # Plot the image
            axes[i].imshow(img, cmap='gray')  # Using grayscale since these are single-band images
            axes[i].set_title(f'Band {i}')
            axes[i].axis('off')
        
        # Save the figure with all bands displayed
        plt.tight_layout()
        plt.savefig(f"{path}/multispectral_bands_" + str(n) + '.png')
        plt.close()

    def compute_image_correlation(self, image):

        image_np = image.unsqueeze(0).numpy()
        
        num_channels = image_np.shape[1] 

        for img in image_np:

            correlations = np.corrcoef(img.reshape(num_channels, -1))
            if np.isnan(correlations).any():
                print(f"Warning: NaN values encountered in correlation matrix")
                correlations = np.nan_to_num(correlations) 
            self.correlation_overall = correlations
