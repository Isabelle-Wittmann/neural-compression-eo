from torchvision import transforms

import torch.nn.functional as F
from compressai.layers import GDN, MaskedConv2d
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from compressai.models.utils import conv, deconv
import torch
import torch.nn as nn
import numpy as np

from compressai.models.base import (
    SCALES_LEVELS,
    SCALES_MAX,
    SCALES_MIN,
    CompressionModel,
    get_scale_table,
)

from .utils import CoordinatePreprocessor, reshape_to_4d

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

    def decompress_common(self, strings, shape):
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}



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

    def decompress(self, strings, shape):
        return self.decompress_common(strings, shape)


class FactorizedPrior(FactorizedPriorBase):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, input_channels=3, entropy_channels=cfg['compressai_model']['M'], **kwargs)

class FactorizedPrior_split(CompressionModel):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.model1 = FactorizedPrior(cfg)
        self.model2 = FactorizedPrior(cfg)
        self.model3 = FactorizedPrior(cfg)
        self.model4 = FactorizedPrior(cfg)
    
    def forward(self, x, v=None, crs=None):
        y1 = self.model1.g_a(x[:,0:3,:,:])
        y_hat1, y_likelihoods1 = self.model1.entropy_bottleneck(y1)

        y2 = self.model2.g_a(x[:,3:6,:,:])
        y_hat2, y_likelihoods2 = self.model2.entropy_bottleneck(y2)

        y3 = self.model3.g_a(x[:,6:9,:,:])
        y_hat3, y_likelihoods3 = self.model3.entropy_bottleneck(y3)

        y4 = self.model4.g_a(x[:,9:,:,:])
        y_hat4, y_likelihoods4 = self.model4.entropy_bottleneck(y4)

        x_hat1 = self.model1.g_s(y_hat1)
        x_hat2 = self.model2.g_s(y_hat2)
        x_hat3 = self.model3.g_s(y_hat3)
        x_hat4 = self.model4.g_s(y_hat4)

        return {"x_hat": torch.cat((x_hat1,x_hat2,x_hat3,x_hat4), dim=1), "likelihoods": {"y":  torch.cat((y_likelihoods1,y_likelihoods2,y_likelihoods3,y_likelihoods4), dim=1)}}

    def compress(self, x, v=None, crs=None):
        
        y1 = self.model1.g_a(x[:,0:3,:,:])
        
        y_strings1 = self.model1.entropy_bottleneck.compress(y1)

        y2 = self.model2.g_a(x[:,3:6,:,:])
        y_strings2 = self.model2.entropy_bottleneck.compress(y2)

        y3 = self.model3.g_a(x[:,6:9,:,:])
        y_strings3 = self.model3.entropy_bottleneck.compress(y3)

        y4 = self.model4.g_a(x[:,9:,:,:])
        y_strings4 = self.model4.entropy_bottleneck.compress(y4)

        return {"strings": [y_strings1,y_strings2,y_strings3,y_strings4], "shape": y4.size()[-2:]}

    def decompress(self, strings, shape):
        y_hat1 = self.model1.entropy_bottleneck.decompress(strings[0], shape)
        y_hat2 = self.model2.entropy_bottleneck.decompress(strings[1], shape)
        y_hat3 = self.model3.entropy_bottleneck.decompress(strings[2], shape)
        y_hat4 = self.model4.entropy_bottleneck.decompress(strings[3], shape)
        x_hat1 = self.model1.g_s(y_hat1).clamp_(0, 1)
        x_hat2 = self.model2.g_s(y_hat2).clamp_(0, 1)
        x_hat3 = self.model3.g_s(y_hat3).clamp_(0, 1)
        x_hat4 = self.model4.g_s(y_hat4).clamp_(0, 1)
        return {"x_hat": torch.cat((x_hat1,x_hat2,x_hat3,x_hat4), dim=1)}

class FactorizedPrior_split_joint(FactorizedPriorBase):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, input_channels=3, entropy_channels=cfg['compressai_model']['M']*4, **kwargs)

        self.g_a1 = self.create_g_a([
            conv(3, self.N),
            GDN(self.N),
            conv(self.N, self.N),
            GDN(self.N),
            conv(self.N, self.N),
            GDN(self.N),
            conv(self.N, self.M),
        ])

        self.g_s1 = self.create_g_s([
            deconv(self.M, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, 3),
        ])

        self.g_a2 = self.create_g_a([
            conv(3, self.N),
            GDN(self.N),
            conv(self.N, self.N),
            GDN(self.N),
            conv(self.N, self.N),
            GDN(self.N),
            conv(self.N, self.M),
        ])

        self.g_s2 = self.create_g_s([
            deconv(self.M, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, 3),
        ])

        self.g_a3 = self.create_g_a([
            conv(3, self.N),
            GDN(self.N),
            conv(self.N, self.N),
            GDN(self.N),
            conv(self.N, self.N),
            GDN(self.N),
            conv(self.N, self.M),
        ])

        self.g_s3 = self.create_g_s([
            deconv(self.M, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, 3),
        ])

        self.g_a4 = self.create_g_a([
            conv(3, self.N),
            GDN(self.N),
            conv(self.N, self.N),
            GDN(self.N),
            conv(self.N, self.N),
            GDN(self.N),
            conv(self.N, self.M),
        ])

        self.g_s4 = self.create_g_s([
            deconv(self.M, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, 3),
        ])
    
    def forward(self, x, v=None, crs=None):
        y1 = self.g_a1(x[:,0:3,:,:])

        y2 = self.g_a2(x[:,3:6,:,:])

        y3 = self.g_a3(x[:,6:9,:,:])

        y4 = self.g_a4(x[:,9:,:,:])

        y_hat, y_likelihoods = self.entropy_bottleneck(torch.cat((y1,y2,y3,y4), dim=1))

        x_hat1 = self.g_s1(y_hat[:,0:self.M,:,:])
        x_hat2 = self.g_s2(y_hat[:,self.M:self.M*2,:,:])
        x_hat3 = self.g_s3(y_hat[:,self.M*2:self.M*3,:,:])
        x_hat4 = self.g_s4(y_hat[:,self.M*3:,:,:])
        

        return {"x_hat": torch.cat((x_hat1,x_hat2,x_hat3,x_hat4), dim=1), "likelihoods": {"y":  y_likelihoods}}

    def compress(self, x, v=None, crs=None):
        
        y1 = self.g_a1(x[:,0:3,:,:])

        y2 = self.g_a2(x[:,3:6,:,:])

        y3 = self.g_a3(x[:,6:9,:,:])

        y4 = self.g_a4(x[:,9:,:,:])
        y_strings = self.entropy_bottleneck.compress(torch.cat((y1,y2,y3,y4), dim=1))

        return {"strings": [y_strings], "shape": torch.cat((y1,y2,y3,y4), dim=1).size()[-2:]}

    def decompress(self, strings, shape):
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)

        x_hat1 = self.g_s1(y_hat[:,0:self.M,:,:]).clamp_(0, 1)
        x_hat2 = self.g_s2(y_hat[:,self.M:self.M*2,:,:]).clamp_(0, 1)
        x_hat3 = self.g_s3(y_hat[:,self.M*2:self.M*3,:,:]).clamp_(0, 1)
        x_hat4 = self.g_s4(y_hat[:,self.M*3:,:,:]).clamp_(0, 1)
        return {"x_hat": torch.cat((x_hat1,x_hat2,x_hat3,x_hat4), dim=1)}

class FactorizedPrior_split_jointflex(FactorizedPriorBase):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, input_channels=3, entropy_channels=cfg['compressai_model']['M'], **kwargs)

        self.g_a1 = self.create_g_a([
            conv(5, self.N),
            GDN(self.N),
            conv(self.N, self.N),
            GDN(self.N),
            conv(self.N, self.M),
        ])

        self.g_s1 = self.create_g_s([
            deconv(self.M, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, 5),
        ])

        self.g_a2 = self.create_g_a([
            conv(5, self.N),
            GDN(self.N),
            conv(self.N, self.N),
            GDN(self.N),
            conv(self.N, self.M),
        ])

        self.g_s2 = self.create_g_s([
            deconv(self.M, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, 5),
        ])

        self.g_a3 = self.create_g_a([
            conv(2, self.N),
            GDN(self.N),
            conv(self.N, self.N),
            GDN(self.N),
            conv(self.N, self.M),
        ])

        self.g_s3 = self.create_g_s([
            deconv(self.M, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, 2),
        ])

        self.g_a = self.create_g_a([
            conv( self.M*3, self.N),
            GDN(self.N),
            conv(self.N, self.M),
        ])

        self.g_s = self.create_g_s([
            deconv(self.M, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, self.M*3),
        ])

        
    def forward(self, x, v=None, crs=None):
        y1 = self.g_a1(x[:,0:5,:,:])

        y2 = self.g_a2(x[:,5:10,:,:])

        y3 = self.g_a3(x[:,10:,:,:])

        y = self.g_a(torch.cat((y1,y2,y3), dim=1))
        print(y.shape)

 
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        y_hat = self.g_s(y_hat)

        x_hat1 = self.g_s1(y_hat[:,0:self.M,:,:])
        x_hat2 = self.g_s2(y_hat[:,self.M:self.M*2,:,:])
        x_hat3 = self.g_s3(y_hat[:,self.M*2:self.M*3,:,:])
        
        return {"x_hat": torch.cat((x_hat1,x_hat2,x_hat3), dim=1), "likelihoods": {"y":  y_likelihoods}}

    def compress(self, x, v=None, crs=None):
        
        y1 = self.g_a1(x[:,0:5,:,:])

        y2 = self.g_a2(x[:,5:10,:,:])

        y3 = self.g_a3(x[:,10:,:,:])

        y = self.g_a(torch.cat((y1,y2,y3), dim=1))
        y_strings = self.entropy_bottleneck.compress(y)

        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def decompress(self, strings, shape):
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)

        y_hat = self.g_s(y_hat)

        x_hat1 = self.g_s1(y_hat[:,0:self.M,:,:]).clamp_(0, 1)
        x_hat2 = self.g_s2(y_hat[:,self.M:self.M*2,:,:]).clamp_(0, 1)
        x_hat3 = self.g_s3(y_hat[:,self.M*2:self.M*3,:,:]).clamp_(0, 1)

        return {"x_hat": torch.cat((x_hat1,x_hat2,x_hat3), dim=1)}


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

    def decompress(self, strings, shape):
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

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


class ScaleHyperpriorMeta(ScaleHyperpriorBase):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, input_channels=3, **kwargs)

        self.coordinate_preprocessor = CoordinatePreprocessor(cfg)

        self.h_a_img = nn.Sequential(
            conv(self.M, self.N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(self.N, self.N),
        )

        self.h_a_vec = nn.Sequential(
            nn.Linear(self.embedding_size, self.V),
            nn.ReLU(inplace=True),
            nn.Linear(self.V, self.N),
            nn.ReLU(inplace=True),
        )

        self.h_a_joint = nn.Sequential(
            conv(self.N, self.N),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, v, crs):

        y = self.g_a(x)
        z_img = self.h_a_img(y)
        z_vec = self.h_a_vec(v)
        z_vec = reshape_to_4d(z_vec, self.N, 4)
        z_joint = z_img * z_vec
        z = self.h_a_joint(z_joint)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        return {"x_hat": self.g_s(y_hat), "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}

    def compress(self, x, v, crs):
        y = self.g_a(x)
        z_img = self.h_a_img(y)
        z_vec = self.h_a_vec(v)
        z_vec = reshape_to_4d(z_vec, self.N, 4)
        z_joint = z_img * z_vec
        z = self.h_a_joint(z_joint)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}


class ScaleHyperpriorMetaOnly(ScaleHyperpriorBase):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.coordinate_preprocessor = CoordinatePreprocessor(cfg)

        self.h_a_joint = nn.Sequential(
            conv(self.N, self.N),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, v, crs):
        y = self.g_a(x)
        z_vec = self.h_a_vec(v)
        z_vec = reshape_to_4d(z_vec, self.N, 4)
        z = self.h_a_joint(z_vec)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(self.g_a(x), scales_hat)
        return {"x_hat": self.g_s(y_hat), "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}

    def compress(self, x, v, crs):
        y = self.g_a(x)
        z_vec = self.h_a_vec(v)
        z_vec = reshape_to_4d(z_vec, self.N, 4)
        z = self.h_a_joint(z_vec)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


class ScaleHyperpriorCRS(ScaleHyperpriorMeta):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        
        self.h_a_vec = nn.Sequential(
            nn.Linear(self.embedding_size, self.N),
            nn.ReLU(inplace=True),
            nn.Linear(self.N, self.N )
        )

        self.h_a_joint = nn.Sequential(
            conv(self.N * 2, self.N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(self.N, self.N),
        )

    def forward(self, x, v, crs):
        y = self.g_a(x)
        z_img = self.h_a_img(y)
        processed_coords = self.coordinate_preprocessor(crs)
        z_vec = self.h_a_vec(processed_coords)
        embedding_2d = reshape_to_4d(z_vec, self.N, 4)

        combined_features = torch.cat((z_img, embedding_2d), dim=1)
        z = self.h_a_joint(combined_features)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        return {"x_hat": self.g_s(y_hat), "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}

    def compress(self, x, v, crs):
        y = self.g_a(x)
        z_img = self.h_a_img(y)
        processed_coords = self.coordinate_preprocessor(crs)
        z_vec = self.h_a_vec(processed_coords)
        embedding_2d = reshape_to_4d(z_vec, self.N, 4)
        combined_features = torch.cat((z_img, embedding_2d), dim=1)
        z = self.h_a_joint(combined_features)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}


class ScaleHyperpriorCRSOnly(ScaleHyperpriorBase):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, input_channels=3, **kwargs)
        self.coordinate_preprocessor = CoordinatePreprocessor(cfg)

        self.h_a_vec = nn.Sequential(
            nn.Linear(self.embedding_size, self.V),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.N * 16)
        )

        self.h_a_joint = nn.Sequential(
            conv(self.N, self.N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(self.N, self.N),
        )

    def forward(self, x, v, crs):
        batch_size = x.size(0)
        processed_coords = self.coordinate_preprocessor(crs)
        z_vec = self.h_a_vec(processed_coords)
        embedding_2d = z_vec.view(batch_size, 192, 4, 4)
        z = self.h_a_joint(embedding_2d)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(self.g_a(x), scales_hat)
        return {"x_hat": self.g_s(y_hat), "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}

    def compress(self, x, v, crs):
        batch_size = x.size(0)
        processed_coords = self.coordinate_preprocessor(crs)
        z_vec = self.h_a_vec(processed_coords)
        embedding_2d = z_vec.view(batch_size, 192, 4, 4)
        z = self.h_a_joint(embedding_2d)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(self.g_a(x), indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


class Conv3d(CompressionModelBase):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, 3, entropy_channels=cfg['compressai_model']['M'], **kwargs)
        
        self.g_a = self.create_g_a([
            conv(16, self.N),
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
            deconv(self.N, 16),
        ])
        self.hp_a = self.create_g_a([nn.Conv3d(
                in_channels=1,
                out_channels=16,
                kernel_size=(2, 5, 5),
                stride=(2, 2, 2),
                padding=(0, 2, 2),
            ),
        
            # # nn.BatchNorm3d(16),
            # nn.LeakyReLU(),
            # nn.Conv3d(
            #     in_channels=16,
            #     out_channels=32,
            #     kernel_size=(5, 5, 5),
            #     stride=(2, 2, 2),
            #     padding=(2, 2, 2),
            # ),
            # # nn.BatchNorm3d(32),
            # nn.LeakyReLU(),
            # _ResBlock(32),
            # _ResBlock(32),
            # _ResBlock(32),
            # nn.Conv3d(
            #     in_channels=32,
            #     out_channels=64,
            #     kernel_size=(5, 5, 5),
            #     stride=(2, 2, 2),
            #     padding=(2, 2, 2),
            # ),
            # # nn.BatchNorm3d(64),
            # nn.LeakyReLU(),
        ])

        self.hp_s = self.create_g_s([nn.Upsample(scale_factor=(2, 4, 4)),
            # nn.Conv3d(
            #     in_channels=64,
            #     out_channels=32,
            #     kernel_size=(5, 5, 5),
            #     stride=(2, 2, 2),
            #     padding=(2, 2, 2),
            # ),
            # # nn.BatchNorm3d(32),
            # nn.LeakyReLU(),
            # _ResBlock(32),
            # _ResBlock(32),
            # _ResBlock(32),
            # nn.Upsample(scale_factor=(3, 2, 2)),
            # nn.Conv3d(
            #     in_channels=32,
            #     out_channels=16,
            #     kernel_size=(5, 3, 3),
            #     stride=(2, 1, 1),
            #     padding=(2, 1, 1),
            # ),
            # # nn.BatchNorm3d(16),
            # nn.LeakyReLU(),
            # nn.Upsample(scale_factor=(3, 2, 2)),

            nn.Conv3d(
                in_channels=16,
                out_channels=3,
                kernel_size=(2, 3, 3),
                stride=(2, 2, 2),
                padding=(0, 1, 1),  # (2, 1, 1)
            ),
            # nn.BatchNorm3d(1),
            nn.LeakyReLU(),
        ])

    @property
    def downsampling_factor(self) -> int:
        return 2 ** 4

    def forward(self, x, v=None, crs=None):
        # print(x.shape)
        x = x.unsqueeze(1)
        x = self.hp_a(x)
        x = x.squeeze(2)
        y = self.g_a(x)
        
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        
        x_hat = self.g_s(y_hat)
        x_hat = x_hat.unsqueeze(2)
        x_hat = self.hp_s(x_hat)
        
        x_hat = x_hat.squeeze(2)
        # print(x_hat.shape)
        
        return {"x_hat": x_hat, "likelihoods": {"y": y_likelihoods}}

    def compress(self, x, v=None, crs=None):
        x = x.unsqueeze(1)
        y = self.g_a(x)

        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-3:]}

    def decompress(self, strings, shape):
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)

        x_hat = self.g_s(y_hat).clamp_(0, 1)
        x_hat = x_hat.squeeze(1)
        return {"x_hat": x_hat}

class _ResBlock(nn.Module):
    def __init__(self, channels):
        super(_ResBlock, self).__init__()

        self.act = nn.LeakyReLU()

        self.block = nn.Sequential(*[
            nn.Conv3d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
            ),
            # nn.BatchNorm3d(channels),
            self.act,
            nn.Conv3d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
            ),
            # nn.BatchNorm3d(channels),
        ])

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.act(out)
        return out
