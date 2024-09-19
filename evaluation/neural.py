import math
import os
import logging
from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from compressai.ops import compute_padding
from evaluation.base_class import CodecTester 
from utils import load_data

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)


class NeuralCodecTester(CodecTester):
    """
    Class for testing neural compression models, extending the base CodecTester class.
    """
    def compute_bpp(self, out_net: Dict[str, Any]) -> float:
        """
        Computes bits per pixel (BPP) from the output of the network.

        Args:
            out_net (Dict[str, Any]): Output dictionary from the network containing 'x_hat' and 'likelihoods'.

        Returns:
            float: The computed BPP value.
        """
        num_pixels = out_net['x_hat'].numel() / out_net['x_hat'].size(1)
        total_bits = sum(
            torch.log(likelihoods).sum() / (-math.log(2))
            for likelihoods in out_net['likelihoods'].values()
        )
        bpp = total_bits / num_pixels
        return bpp.item()
    
    def forward_pass(
        self, image: torch.Tensor, crs: torch.Tensor, date: str, net: torch.nn.Module
    ) -> Tuple[float, torch.Tensor]:
        """
        Performs a forward pass through the network and computes the BPP.

        Args:
            image (torch.Tensor): The input image tensor.
            label (torch.Tensor): The label tensor.
            crs (torch.Tensor): Coordinate reference system tensor.
            net (torch.nn.Module): The neural network model.

        Returns:
            Tuple[float, torch.Tensor]: The BPP value and the reconstructed image tensor.
        """
        out_net = net(image, crs, date)
        bpp = self.compute_bpp(out_net)
        bpp = bpp / image.size(1) if self.bpp_per_channel else bpp
        return bpp, out_net['x_hat']

    def inference(
        self, image: torch.Tensor, crs: torch.Tensor, date: str, net: torch.nn.Module
    ) -> float:
        """
        Performs model inference and computes the BPP.

        Args:
            image (torch.Tensor): The input image tensor.
            label (torch.Tensor): The label tensor.
            crs (torch.Tensor): Coordinate reference system tensor.
            net (torch.nn.Module): The neural network model.

        Returns:
            float: The BPP value.
        """
        pad, unpad = compute_padding(self.height, self.width, min_div=2 ** 6)
        x_padded = F.pad(image, pad, mode="constant", value=0)
        out_enc = net.compress(x_padded, crs, date)
        total_bits = sum(len(s) * 8 for s in out_enc["strings"][0])
        bpp = total_bits / self.num_pixels
        return bpp

    def extract_general_pmf(self, entropy_model) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts the probability mass function (PMF) and shifts from a fitted CompressAI entropy model. 
        Parts are taken from the CompressAI library.

        Args:
            entropy_model: CompressAI entropy model from which to extract the PMF.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The PMF tensor and shift values for quantization.
        """
        # Get median values from the entropy model
        medians = entropy_model.quantiles[:, 0, 1]

        # Get range of quantized values
        minima = medians - entropy_model.quantiles[:, 0, 0]
        maxima = entropy_model.quantiles[:, 0, 2] - medians

        minima = torch.ceil(minima).int()
        maxima = torch.ceil(maxima).int()
        minima = torch.clamp(minima, min=0)
        maxima = torch.clamp(maxima, min=0)

        pmf_start = medians - minima
        pmf_length = maxima + minima + 1
        max_length = pmf_length.max().item()
        device = pmf_start.device
        samples = torch.arange(max_length, device=device)

        samples = samples[None, :] + pmf_start[:, None, None]
        pmf, _, _ = entropy_model._likelihood(samples, stop_gradient=True)
        pmf = pmf[:, 0, :]

        return pmf, pmf_start[:, None, None]

    def pmf_stats(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Computes statistics for the PMF tensor.

        Args:
            tensor (torch.Tensor): PMF tensor.

        Returns:
            Tuple[torch.Tensor, ...]: A tuple containing entropies, minima, maxima, zeros_count, ones_count,
                                      zeros_count_overall, and ones_count_overall.
        """
        tensor = tensor.squeeze()
        safe_tensor = torch.where(tensor > 0, tensor, torch.tensor(1e-20, device=tensor.device))
        entropies = -torch.sum(safe_tensor * torch.log2(safe_tensor), dim=1)
        minima = tensor.min(dim=1).values
        maxima = tensor.max(dim=1).values
        zeros_count = (tensor == 0).sum(dim=1)
        ones_count = (tensor == 1).sum(dim=1)
        zeros_count_overall = (tensor == 0).sum()
        ones_count_overall = (tensor == 1).sum()
        return entropies, minima, maxima, zeros_count, ones_count, zeros_count_overall, ones_count_overall

    def save_pmfs(self, pmf: torch.Tensor, path: str, name: str) -> None:
        """
        Saves PMF histograms per channel.

        Args:
            pmf (torch.Tensor): The PMF tensor.
            path (str): The directory path to save the images.
            name (str): The name for the saved file.
        """
        pmf = pmf.squeeze()
        entropies, minima, maxima, zeros_count, ones_count, zeros_count_overall, ones_count_overall = self.pmf_stats(pmf)
        num_channels = pmf.shape[0]
        grid_size = int(math.ceil(math.sqrt(num_channels)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 4, grid_size * 3))
        fig.suptitle(f'Overall: Zeros = {zeros_count_overall}, Ones = {ones_count_overall}', fontsize=16)

        norm = plt.Normalize(entropies.min(), entropies.max())
        colormap = cm.get_cmap('coolwarm_r')

        for idx, ax in enumerate(axes.flatten()):
            if idx >= num_channels:
                ax.axis('off')
                continue
            channel_probs = pmf[idx].cpu().numpy()
            color = colormap(norm(entropies[idx].cpu().numpy()))
            x_vals = np.arange(len(channel_probs))
            ax.bar(x_vals, channel_probs, width=0.9, color=color)
            ax.set_title(
                f"Entropy: {entropies[idx]:.2f}, Min: {minima[idx]:.2f}, Max: {maxima[idx]:.2f}, "
                f"Zeros: {zeros_count[idx]}, Ones: {ones_count[idx]}",
                fontsize=8
            )
            ax.set_xlim([0, len(channel_probs) - 1])
            ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(path, f'{name}_pmfs.png'))
        plt.close()

    def save_latents(self, latents: torch.Tensor, path: str, name: str) -> None:
        """
        Saves the latent variables/ embedding vector per channel as images. 
        Produces a gird with one image, indicating the values the latent vector of that dimension takes, per latent channel.

        Args:
            latents (torch.Tensor): Latent tensor to be saved.
            path (str): Directory path to save the images.
            name (str): Name for the saved file.
        """
        latents = latents.squeeze()
        num_channels = latents.shape[0]
        grid_size = int(math.ceil(math.sqrt(num_channels)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))

        for idx, ax in enumerate(axes.flatten()):
            if idx >= num_channels:
                ax.axis('off')
                continue
            img = transforms.ToPILImage()(latents[idx])
            ax.imshow(img)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(path, f'{name}_latents.png'))
        plt.close()


    def save_latents_histogram(
        self, latents: torch.Tensor, path: str, name: str, pmf_max: int
    ) -> None:
        """
        Saves histograms of latent vector per channel.

        Args:
            latents (torch.Tensor): The latent tensor.
            path (str): The directory path to save the images.
            name (str): The name for the saved file.
            pmf_max (int): The maximum PMF value for histogram bins.
        """
        latents = latents.squeeze()
        num_channels = latents.shape[0]
        grid_size = int(math.ceil(math.sqrt(num_channels)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 4, grid_size * 3))

        for idx, ax in enumerate(axes.flatten()):
            if idx >= num_channels:
                ax.axis('off')
                continue
            channel_vals = latents[idx].cpu().numpy()
            bins = np.arange(0, pmf_max + 1)
            ax.hist(channel_vals.flatten(), bins=bins, rwidth=0.9)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(path, f'{name}_latent_histograms.png'))
        plt.close()

    def save_flattened_latent_histogram(
        self, latents: torch.Tensor, path: str, name: str, bins: Optional[int] = None
    ) -> None:
        """
        Saves a histogram of the flattened latent vector (across all channels).

        Args:
            latents (torch.Tensor): The latent tensor.
            path (str): The directory path to save the image.
            name (str): The name for the saved file.
            bins (int, optional): The number of bins for the histogram. If None, computed from data range.
        """
        latents = latents.flatten()
        latents_np = latents.cpu().detach().numpy()

        if bins is None:
            bins = int(latents_np.max() - latents_np.min())

        ent_counts, ent_bins = np.histogram(latents_np, bins=bins)

        fig = plt.figure(figsize=(12, 4))
        bar_color = 'blue'

        plt.hist(
            ent_bins[:-1],
            ent_bins,
            weights=ent_counts,
            rwidth=0.9,
            color=bar_color
        )

        plt.xlabel('Latent Values')
        plt.ylabel('Frequency')
        plt.xlim( - 2, bins + 2)
        plt.tight_layout()

        plt.savefig(os.path.join(path, f'{name}_flattened_histogram.png'))
        plt.close()

    def save_sample_reconstruction(
        self, image: torch.Tensor, model: torch.nn.Module, path: str
    ) -> None:
        """
        Saves a sample reconstruction comparing the original and reconstructed images.

        Args:
            image (torch.Tensor): The input image tensor.
            model (torch.nn.Module): The neural network model.
            path (str): The directory path to save the images.
        """
        image, label, crs, date, time = load_data(image, self.is_bigearth_data, self.device)
        _, decompressed = self.forward_pass(image.unsqueeze(0), crs.unsqueeze(0), date, model)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        if not (0.2 <= image.max() <= 0.4):
            logging.warning(f"Pixel values may not be suitable for visualization. Max is: {image.max()}")

        if self.bands > 3:
            img = image.squeeze()[1:3]
            img_dec = decompressed.squeeze()[1:3]
        else:
            img = image.squeeze()
            img_dec = decompressed.squeeze()

        img = transforms.ToPILImage()(np.clip(img.cpu().detach() / 0.3, 0, 1))
        axes[0].imshow(img)
        axes[0].set_title('Original')

        img_dec = transforms.ToPILImage()(np.clip(img_dec.cpu().detach() / 0.3, 0, 1))
        axes[1].imshow(img_dec)
        axes[1].set_title('Reconstruction')

        plt.tight_layout()
        plt.savefig(os.path.join(path, 'sample_reconstruction.png'))
        plt.close()

    def cluster_latents(self, path, name, k = 5):
        latent_data = np.stack(self.latents)
        print(latent_data.shape)
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(latent_data)

        # Dimensionality reduction using PCA
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(latent_data)

        # Visualization using PCA
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis')
        plt.title('PCA of Flattened Tensors')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(*scatter.legend_elements(), title="Clusters")
 
        plt.savefig(os.path.join(path, f'{name}_flattened_histogram.png'))
        plt.close()

    def get_metrics(self, model, path: str, save_pmfs: bool) -> None:
        """
        Evaluates the model over the dataset and computes various metrics.

        Args:
            model (torch.nn.Module): The neural network model.
            path (str): The path to save outputs.
        """
        with torch.no_grad():
            model.eval()
            entropy_model = model.entropy_bottleneck
            print(model.measure_channel_influence())
            pmf, normscales = self.extract_general_pmf(entropy_model)

            if save_pmfs:
                self.save_pmfs(pmf, path, self.name)

            # latents_sum = None
            pmf_max = pmf.shape[1]

            for count, data in enumerate(self.dataloader):
                image, label, crs, date, time = load_data(data, self.is_bigearth_data, self.device)
                bpp_est, decompressed = self.forward_pass(image, crs, date, model)
                self.bpp_est_all.append(bpp_est)
                self.bpp_all.append(self.inference(image, crs, date, model))
                self.compute_distortion_metrics(image.squeeze(), decompressed.squeeze())

                y, y_hat, _ = model.embedding(image, crs, date)
                y_hat_norm = y_hat - normscales
                y_hat_norm = torch.clamp(y_hat_norm, min=0, max=pmf_max)
                self.latents.append(y_hat_norm.view(-1).cpu().numpy())

                # if latents_sum is None:
                #     latents_sum = y_hat_norm.clone()
                # else:
                #     latents_sum += y_hat_norm

                if count % 100 == 0:
                    logging.info(f"Processed {count} images")

            # average_latents = latents_sum / (count + 1)
            #name = self.name + '_' +str(index)
            # self.save_latents(latents, path, name)
            # self.save_latents_histogram(latents, path, name, pmf_max)
            # self.save_flattened_latent_histogram(latents, path, name)
            self.cluster_latents(path, self.name, k=5)


class NeuralCodecTesterSplit(NeuralCodecTester):
    """
    A class for testing neural codecs with split networks, extending NeuralCodecTester.
    """

    def forward_pass(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        crs: torch.Tensor,
        net1: torch.nn.Module,
        net2: torch.nn.Module,
        net3: torch.nn.Module,
        net4: torch.nn.Module,
    ) -> Tuple[float, torch.Tensor]:
        """
        Performs a forward pass through multiple networks and computes the BPP.

        Args:
            image (torch.Tensor): The input image tensor.
            label (torch.Tensor): The label tensor.
            crs (torch.Tensor): Coordinate reference system tensor.
            net1, net2, net3, net4 (torch.nn.Module): The neural network models.

        Returns:
            Tuple[float, torch.Tensor]: The BPP value and the reconstructed image tensor.
        """
        rv1 = net1(image[:, :3, :, :], label, crs)
        rv2 = net2(image[:, 3:6, :, :], label, crs)
        rv3 = net3(image[:, 6:9, :, :], label, crs)
        rv4 = net4(image[:, 9:, :, :], label, crs)
        rv_img = torch.cat((rv1['x_hat'], rv2['x_hat'], rv3['x_hat'], rv4['x_hat']), dim=1)
        likelihoods = {
            "y": torch.cat(
                (rv1['likelihoods']['y'], rv2['likelihoods']['y'], rv3['likelihoods']['y'], rv4['likelihoods']['y']),
                dim=1
            )
        }
        bpp = self.compute_bpp({'x_hat': rv_img, 'likelihoods': likelihoods})
        bpp = bpp / image.size(1) if self.bpp_per_channel else bpp
        return bpp, rv_img

    def inference(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        crs: torch.Tensor,
        net1: torch.nn.Module,
        net2: torch.nn.Module,
        net3: torch.nn.Module,
        net4: torch.nn.Module,
    ) -> float:
        """
        Performs model inference and computes the BPP using multiple networks.

        Args:
            image (torch.Tensor): The input image tensor.
            label (torch.Tensor): The label tensor.
            crs (torch.Tensor): Coordinate reference system tensor.
            net1, net2, net3, net4 (torch.nn.Module): The neural network models.

        Returns:
            float: The BPP value.
        """
        pad, _ = compute_padding(self.height, self.width, min_div=2 ** 6)
        x_padded = F.pad(image, pad, mode="constant", value=0)
        out_enc1 = net1.compress(x_padded[:, :3, :, :], label, crs)
        out_enc2 = net2.compress(x_padded[:, 3:6, :, :], label, crs)
        out_enc3 = net3.compress(x_padded[:, 6:9, :, :], label, crs)
        out_enc4 = net4.compress(x_padded[:, 9:, :, :], label, crs)

        total_bits = sum(len(s) * 8 for s in out_enc1["strings"][0])
        total_bits += sum(len(s) * 8 for s in out_enc2["strings"][0])
        total_bits += sum(len(s) * 8 for s in out_enc3["strings"][0])
        total_bits += sum(len(s) * 8 for s in out_enc4["strings"][0])
        bpp = total_bits / self.num_pixels
        return bpp

    def get_metrics(
        self,
        model1: torch.nn.Module,
        model2: torch.nn.Module,
        model3: torch.nn.Module,
        model4: torch.nn.Module,
    ) -> None:
        """
        Evaluates the models over the dataset and computes various metrics.

        Args:
            model1, model2, model3, model4 (torch.nn.Module): The neural network models.
        """
        with torch.no_grad():
            for count, data in enumerate(self.dataloader):
                image, label, crs, date, time = load_data(data, self.is_bigearth_data, self.device)
                bpp_est, decompressed = self.forward_pass(image, label, crs, model1, model2, model3, model4)
                self.bpp_est_all.append(bpp_est)
                self.bpp_all.append(self.inference(image, label, crs, model1, model2, model3, model4))
                self.compute_distortion_metrics(image.squeeze(), decompressed.squeeze())

                if count % 100 == 0:
                    logging.info(f"Processed {count} images")

    def save_sample_reconstruction(
        self,
        image: torch.Tensor,
        model1: torch.nn.Module,
        model2: torch.nn.Module,
        model3: torch.nn.Module,
        model4: torch.nn.Module,
        path: str,
    ) -> None:
        """
        Saves a sample reconstruction comparing the original and reconstructed images.

        Args:
            image (torch.Tensor): The input image tensor.
            model1, model2, model3, model4 (torch.nn.Module): The neural network models.
            path (str): The directory path to save the images.
        """
        image, label, crs, date, time = load_data(image, self.is_bigearth_data, self.device)
        _, decompressed = self.forward_pass(
            image.unsqueeze(0), label, crs.unsqueeze(0), model1, model2, model3, model4
        )

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        if not (0.2 <= image.max() <= 0.4):
            logging.warning(f"Pixel values may not be suitable for visualization. Max is: {image.max()}")

        if self.bands > 3:
            img = image.squeeze()[1:3]
            img_dec = decompressed.squeeze()[1:3]
        else:
            img = image.squeeze()
            img_dec = decompressed.squeeze()

        img = transforms.ToPILImage()(np.clip(img.cpu().detach() / 0.3, 0, 1))
        axes[0].imshow(img)
        axes[0].set_title('Original')

        img_dec = transforms.ToPILImage()(np.clip(img_dec.cpu().detach() / 0.3, 0, 1))
        axes[1].imshow(img_dec)
        axes[1].set_title('Reconstruction')

        plt.tight_layout()
        plt.savefig(os.path.join(path, 'sample_reconstruction.png'))
        plt.close()
