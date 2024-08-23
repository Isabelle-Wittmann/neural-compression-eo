from torchvision import transforms

import torch.nn.functional as F
from compressai.layers import GDN, MaskedConv2d
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from compressai.models.utils import conv, deconv
import torch
import torch.nn as nn
import numpy as np
from .utils import CoordinatePreprocessor, reshape_to_4d
from .base_architectures import *


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

