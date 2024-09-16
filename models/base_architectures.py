from torchvision import transforms

import torch.nn.functional as F
from compressai.layers import GDN, MaskedConv2d
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from compressai.models.utils import conv, deconv
import torch
import torch.nn as nn

from compressai.models.base import CompressionModel

class CompressionModelBase(CompressionModel):
    def __init__(self, cfg, input_channels, entropy_channels, **kwargs):
        super().__init__(**kwargs)

        self.N = cfg['compressai_model']['N']
        self.M = cfg['compressai_model']['M']
        self.V = cfg['compressai_model']['V']
        self.embedding_size = cfg['preprocessing']['coordinate_embedding_dim']


        self.entropy_bottleneck = EntropyBottleneck(entropy_channels)
        self.input_channels = input_channels

    def create_g_a(self, layers):
        return nn.Sequential(*layers)

    def create_g_s(self, layers):
        return nn.Sequential(*layers)

    def forward_common(self, x, y_hat, y_likelihoods):
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat, "likelihoods": {"y": y_likelihoods}}

    def compress_common(self, y):
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def decompress_common(self, strings, shape, crs):
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
    
    def embedding(self, x):
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        return y, y_hat, self.entropy_bottleneck._quantized_cdf

class FactorizedPriorBase(CompressionModelBase):
    def __init__(self, cfg, input_channels, **kwargs):
        super().__init__(cfg, input_channels, **kwargs)
        
        self.g_a = self.create_g_a([
            conv(input_channels, self.N),
            GDN(self.N),
            conv(self.N, self.N),
            GDN(self.N),
            conv(self.N, self.N),
            GDN(self.N),
            conv(self.N, self.M),
        ])

        self.g_s = self.create_g_s([
            deconv(self.M, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, input_channels),
        ])

    @property
    def downsampling_factor(self) -> int:
        return 2 ** 4

    def forward(self, x, v=None, crs=None):
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        return self.forward_common(x, y_hat, y_likelihoods)

    def compress(self, x, v=None, crs=None):
        y = self.g_a(x)
        return self.compress_common(y)

    def decompress(self, strings, shape, crs):
        return self.decompress_common(strings, shape, crs)


class FactorizedPrior(FactorizedPriorBase):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, input_channels=3, entropy_channels=cfg['compressai_model']['M'], **kwargs)

class FactorizedPriorBaseSlim(CompressionModelBase):
    def __init__(self, cfg, input_channels, **kwargs):
        super().__init__(cfg, input_channels, **kwargs)
        
        self.g_a = self.create_g_a([
            conv(input_channels, self.N),
            GDN(self.N),
            conv(self.N, self.M),
        ])

        self.g_s = self.create_g_s([
            deconv(self.M, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, input_channels),
        ])

    @property
    def downsampling_factor(self) -> int:
        return 2 ** 4

    def forward(self, x, v=None, crs=None):
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        return self.forward_common(x, y_hat, y_likelihoods)

    def compress(self, x, v=None, crs=None):
        y = self.g_a(x)
        return self.compress_common(y)

    def decompress(self, strings, shape, crs):
        return self.decompress_common(strings, shape, crs)


class FactorizedPriorSlim(FactorizedPriorBaseSlim):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, input_channels=3, entropy_channels=cfg['compressai_model']['M'], **kwargs)

class FactorizedPrior12Channel(FactorizedPriorBase):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, input_channels=12, entropy_channels=cfg['compressai_model']['M'], **kwargs)


class ScaleHyperpriorBase(FactorizedPriorBase):
    def __init__(self, cfg, input_channels, **kwargs):
        super().__init__(cfg, input_channels, entropy_channels=cfg['compressai_model']['N'], **kwargs)

        self.h_a = nn.Sequential(
            conv(self.M, self.N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(self.N, self.N),
            nn.ReLU(inplace=True),
            conv(self.N, self.N),
        )

        self.h_s = nn.Sequential(
            deconv(self.N, self.N),
            nn.ReLU(inplace=True),
            deconv(self.N, self.N),
            nn.ReLU(inplace=True),
            conv(self.N, self.M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x, v=None, crs=None):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        return {"x_hat": self.g_s(y_hat), "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}

    def compress(self, x, v=None, crs=None):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape, crs):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


class ScaleHyperprior(ScaleHyperpriorBase):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, input_channels=3, **kwargs)


class ScaleHyperprior12Channel(ScaleHyperpriorBase):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, input_channels=12, **kwargs)


class MeanScaleHyperprior(ScaleHyperpriorBase):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, input_channels=3, **kwargs)

        self.h_s = nn.Sequential(
            deconv(self.N, self.M),
            nn.LeakyReLU(inplace=True),
            deconv(self.M, self.M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(self.M * 3 // 2, self.M * 2, stride=1, kernel_size=3),
        )

    def forward(self, x, v=None, crs=None):
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

    def compress(self, x, v=None, crs=None):
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape, crs):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
