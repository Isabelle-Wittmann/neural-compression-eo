import os
import argparse
import yaml
import random
import numpy as np 
import matplotlib.pyplot as plt
import csv
import ast

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.layers import GDN, MaskedConv2d
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN

import compressai
from compressai.zoo import (bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor) 

from datasets.bigearthnet_loader import init_bigearthnet
# from models.evaluate import forward_pass, test_model, plot_rate_distortion, print_reconstructions, run_impl


from compressai.models.base import (
    SCALES_LEVELS,
    SCALES_MAX,
    SCALES_MIN,
    CompressionModel,
    get_scale_table,
)
from compressai.models.utils import conv, deconv

class FactorizedPrior(CompressionModel):
        r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
        N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
        <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
        (ICLR), 2018.

        .. code-block:: none

                    ┌───┐    y
                x ──►─┤g_a├──►─┐
                    └───┘    │
                            ▼
                            ┌─┴─┐
                            │ Q │
                            └─┬─┘
                            │
                        y_hat ▼
                            │
                            ·
                            EB :
                            ·
                            │
                        y_hat ▼
                            │
                    ┌───┐    │
            x_hat ──◄─┤g_s├────┘
                    └───┘

            EB = Entropy bottleneck

        Args:
            N (int): Number of channels
            M (int): Number of channels in the expansion layers (last layer of the
                encoder and last layer of the hyperprior decoder)
        """

        def __init__(self, cfg, **kwargs):
            super().__init__(**kwargs)
            N = cfg['compressai_model']['N']
            M = cfg['compressai_model']['M']
            self.entropy_bottleneck = EntropyBottleneck(M)

            self.g_a = nn.Sequential(
                conv(3, N),
                GDN(N),
                conv(N, N),
                GDN(N),
                conv(N, N),
                GDN(N),
                conv(N, M),
            )

            self.g_s = nn.Sequential(
                deconv(M, N),
                GDN(N, inverse=True),
                deconv(N, N),
                GDN(N, inverse=True),
                deconv(N, N),
                GDN(N, inverse=True),
                deconv(N, 3),
            )

            self.N = N
            self.M = M

        @property
        def downsampling_factor(self) -> int:
            return 2**4

        def forward(self, x, v, crs):
            y = self.g_a(x)
            y_hat, y_likelihoods = self.entropy_bottleneck(y)
            x_hat = self.g_s(y_hat)

            return {
                "x_hat": x_hat,
                "likelihoods": {
                    "y": y_likelihoods,
                },
            }

        @classmethod
        def from_state_dict(cls, state_dict):
            """Return a new model instance from `state_dict`."""
            N = state_dict["g_a.0.weight"].size(0)
            M = state_dict["g_a.6.weight"].size(0)
            net = cls(N, M)
            net.load_state_dict(state_dict)
            return net

        def compress(self, x, v, crs):
            y = self.g_a(x)
            y_strings = self.entropy_bottleneck.compress(y)
            return {"strings": [y_strings], "shape": y.size()[-2:]}

        def decompress(self, strings, shape):
            assert isinstance(strings, list) and len(strings) == 1
            y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
            x_hat = self.g_s(y_hat).clamp_(0, 1)
            return {"x_hat": x_hat}
        
class FactorizedPrior_12channel(FactorizedPrior):

        def __init__(self, cfg, **kwargs):
            super().__init__(cfg, **kwargs)
            N = cfg['compressai_model']['N']
            M = cfg['compressai_model']['M']
            self.entropy_bottleneck = EntropyBottleneck(M)

            self.g_a = nn.Sequential(
                conv(12, N),
                GDN(N),
                conv(N, N),
                GDN(N),
                conv(N, N),
                GDN(N),
                conv(N, M),
            )

            self.g_s = nn.Sequential(
                deconv(M, N),
                GDN(N, inverse=True),
                deconv(N, N),
                GDN(N, inverse=True),
                deconv(N, N),
                GDN(N, inverse=True),
                deconv(N, 12),
            )

            self.N = N
            self.M = M
        
class ScaleHyperprior(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.


        EB = Entropy bottleneck
        GC = Gaussian conditional

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self,cfg, **kwargs):
        super().__init__(**kwargs)
        N = cfg['compressai_model']['N']
        M = cfg['compressai_model']['M']

        self.entropy_bottleneck = EntropyBottleneck(N)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x, v, crs):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x, v, crs):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}



class MeanScaleHyperprior(ScaleHyperprior):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

        EB = Entropy bottleneck
        GC = Gaussian conditional

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

    def forward(self, x, v, crs):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, v, crs):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

def reshape_to_4d(input_tensor):
    # Ensure the input tensor has the expected shape
    if input_tensor.dim() > 2:
        print(input_tensor.shape)
        raise ValueError("Input tensor must have shape (batch, x)")

    output_tensor = input_tensor.view(-1, 192, 1, 1)
    
    return output_tensor.repeat(1, 1, 4, 4)


class ScaleHyperprior_meta(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.


        EB = Entropy bottleneck
        GC = Gaussian conditional

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self,cfg, **kwargs):
        super().__init__(**kwargs)
        N = cfg['compressai_model']['N']
        M = cfg['compressai_model']['M']
        V = cfg['compressai_model']['V']

        self.entropy_bottleneck = EntropyBottleneck(N)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a_img = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_a_vec = nn.Sequential(
            nn.Linear(V, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, N),
            nn.ReLU(inplace=True),
        )


        self.h_a_joint = nn.Sequential(

            conv(N, N),
            nn.ReLU(inplace=True),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x, v, crs):
        y = self.g_a(x)
        z_img = self.h_a_img(y)
        z_vec = self.h_a_vec(v)
        z_vec = reshape_to_4d(z_vec)
        z_joint = z_img * z_vec
        z = self.h_a_joint(z_joint)
        # z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x, v, crs):
        y = self.g_a(x)
        z_img = self.h_a_img(y)
        z_vec = self.h_a_vec(v)
        z_vec = reshape_to_4d(z_vec)
        z_joint = z_img * z_vec
        z = self.h_a_joint(z_joint)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

class ScaleHyperprior_meta_only(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.


        EB = Entropy bottleneck
        GC = Gaussian conditional

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self,cfg, **kwargs):
        super().__init__(**kwargs)
        N = cfg['compressai_model']['N']
        M = cfg['compressai_model']['M']
        V = cfg['compressai_model']['V']

        self.entropy_bottleneck = EntropyBottleneck(N)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a_img = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_a_vec = nn.Sequential(
            nn.Linear(V, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, N),
            nn.ReLU(inplace=True),
        )


        self.h_a_joint = nn.Sequential(

            conv(N, N),
            nn.ReLU(inplace=True),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x, v, crs):
        y = self.g_a(x)
        z_vec = self.h_a_vec(v)
        z_vec = reshape_to_4d(z_vec)
        z = self.h_a_joint(z_vec)
        # z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x, v, crs):
        y = self.g_a(x)
        z_vec = self.h_a_vec(v)
        z_vec = reshape_to_4d(z_vec)
        z = self.h_a_joint(z_vec)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

class ScaleHyperprior_crs_only(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.


        EB = Entropy bottleneck
        GC = Gaussian conditional

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self,cfg, **kwargs):
        super().__init__(**kwargs)
        N = cfg['compressai_model']['N']
        M = cfg['compressai_model']['M']
        V = cfg['compressai_model']['V']

        self.entropy_bottleneck = EntropyBottleneck(N)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a_img = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_a_vec = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, N),
            nn.ReLU(inplace=True),
        )


        self.h_a_joint = nn.Sequential(

            conv(N, N),
            nn.ReLU(inplace=True),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x, v, crs):
       
        y = self.g_a(x)
        z_vec = self.h_a_vec(crs)
        z_vec = reshape_to_4d(z_vec)
        z = self.h_a_joint(z_vec)
        # z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x, v, crs):
        y = self.g_a(x)
        z_vec = self.h_a_vec(crs)
        z_vec = reshape_to_4d(z_vec)
        z = self.h_a_joint(z_vec)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
