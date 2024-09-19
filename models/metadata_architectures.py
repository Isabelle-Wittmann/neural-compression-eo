from compressai.models.utils import conv, deconv
import torch
import torch.nn as nn
from .utils import CoordinatePreprocessor, reshape_to_4d
from .base_architectures import *

# ---------------------------------------------------------------------------------------------
class FactorizedPriorEnc(FactorizedPriorBase):
    def __init__(self, cfg, **kwargs):

        super().__init__(cfg, input_channels=int(3 +  cfg['preprocessing']['coordinate_embedding_dim']), entropy_channels=cfg['compressai_model']['M'], **kwargs)
                
        self.coordinate_channel = cfg['preprocessing']['coordinate_embedding_dim']
        self.coordinate_preprocessor = CoordinatePreprocessor(cfg)

        self.g_s = self.create_g_s([
            deconv(self.M, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, 3),
        ])
    
    def forward(self, x, crs=None, date=None):
        batch_size = x.shape[0]
        processed_coords = self.coordinate_preprocessor(crs)
        processed_coords = processed_coords.unsqueeze(-1).unsqueeze(-1)
        processed_coords = processed_coords.expand(batch_size, self.coordinate_channel, 128, 128)

        x_joint = torch.cat((x, processed_coords), dim=1)

        y = self.g_a(x_joint)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        return self.forward_common(x, y_hat, y_likelihoods)

    def compress(self, x, crs=None, date=None):
        batch_size = x.shape[0]
        processed_coords = self.coordinate_preprocessor(crs)
        processed_coords = processed_coords.unsqueeze(-1).unsqueeze(-1)
        processed_coords = processed_coords.expand(batch_size, self.coordinate_channel, 128, 128)

        x_joint = torch.cat((x, processed_coords), dim=1)

        y = self.g_a(x_joint)
        return self.compress_common(y)
    
    def embedding(self, x, crs=None, date=None):
        batch_size = x.shape[0]
        processed_coords = self.coordinate_preprocessor(crs)
        processed_coords = processed_coords.unsqueeze(-1).unsqueeze(-1)
        processed_coords = processed_coords.expand(batch_size, self.coordinate_channel, 128, 128)

        x_joint = torch.cat((x, processed_coords), dim=1)

        y = self.g_a(x_joint)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        return y, y_hat, self.entropy_bottleneck._quantized_cdf

class FactorizedPriorEncOne(FactorizedPriorBase):
    def __init__(self, cfg, **kwargs):

        super().__init__(cfg, input_channels=int(1 +  cfg['preprocessing']['coordinate_embedding_dim']), entropy_channels=cfg['compressai_model']['M'], **kwargs)
                
        self.coordinate_channel = cfg['preprocessing']['coordinate_embedding_dim']
        self.coordinate_preprocessor = CoordinatePreprocessor(cfg)

        self.g_s = self.create_g_s([
            deconv(self.M, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, 3),
        ])
    
    def forward(self, x, crs=None, date=None):
        batch_size = x.shape[0]
        processed_coords = self.coordinate_preprocessor(crs)
        processed_coords = processed_coords.unsqueeze(-1).unsqueeze(-1)
        processed_coords = processed_coords.expand(batch_size, self.coordinate_channel, 128, 128)

        x_joint = torch.cat((x[:,0].unsqueeze(1), processed_coords), dim=1)

        y = self.g_a(x_joint)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        return self.forward_common(x, y_hat, y_likelihoods)

    def compress(self, x, crs=None, date=None):
        batch_size = x.shape[0]
        processed_coords = self.coordinate_preprocessor(crs)
        processed_coords = processed_coords.unsqueeze(-1).unsqueeze(-1)
        processed_coords = processed_coords.expand(batch_size, self.coordinate_channel, 128, 128)

        x_joint = torch.cat((x[:,0].unsqueeze(1), processed_coords), dim=1)

        y = self.g_a(x_joint)
        return self.compress_common(y)
    
    def embedding(self, x, crs=None, date=None):
        batch_size = x.shape[0]
        processed_coords = self.coordinate_preprocessor(crs)
        processed_coords = processed_coords.unsqueeze(-1).unsqueeze(-1)
        processed_coords = processed_coords.expand(batch_size, self.coordinate_channel, 128, 128)

        x_joint = torch.cat((x[:,0].unsqueeze(1), processed_coords), dim=1)

        y = self.g_a(x_joint)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        return y, y_hat, self.entropy_bottleneck._quantized_cdf
    
class FactorizedPriorCRSDec(FactorizedPriorBase):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, input_channels=3, entropy_channels=cfg['compressai_model']['M'], **kwargs)
                
        self.coordinate_channel = cfg['preprocessing']['coordinate_embedding_dim']
        self.coordinate_preprocessor = CoordinatePreprocessor(cfg)

        self.g_s = self.create_g_s([
            deconv(self.M + 3, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, 3),
        ])
        
    def forward(self, x, crs=None, date=None):
        batch_size = x.shape[0]

        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)

        processed_coords = self.coordinate_preprocessor(crs)
        processed_coords = processed_coords.unsqueeze(-1).unsqueeze(-1)
        processed_coords = processed_coords.expand(batch_size, self.coordinate_channel, 8, 8)

        y_joint = torch.cat((y_hat, processed_coords), dim=1)
        x_hat = self.g_s(y_joint)
        return {"x_hat": x_hat, "likelihoods": {"y": y_likelihoods}}
    
    def decompress(self, strings, shape, crs, date):
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)

        batch_size = y_hat.shape[0]
        processed_coords = self.coordinate_preprocessor(crs)
        processed_coords = processed_coords.unsqueeze(-1).unsqueeze(-1)
        processed_coords = processed_coords.expand(batch_size, self.coordinate_channel, 8, 8)

        y_joint = torch.cat((y_hat, processed_coords), dim=1)

        x_hat = self.g_s(y_joint).clamp_(0, 1)
        return {"x_hat": x_hat}



class FactorizedPriorEncDec(FactorizedPriorBase):
    def __init__(self, cfg, **kwargs):

        super().__init__(cfg, input_channels=int(3 +  cfg['preprocessing']['coordinate_embedding_dim']), entropy_channels=cfg['compressai_model']['M'], **kwargs)
                
        self.coordinate_channel = cfg['preprocessing']['coordinate_embedding_dim']
        self.coordinate_preprocessor = CoordinatePreprocessor(cfg)

        self.g_s = self.create_g_s([
            deconv(self.M, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, 3),
        ])
        self.g_s = self.create_g_s([
            deconv(self.M + 3, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, 3),
        ])
    
    def forward(self, x, crs=None, date=None):
        batch_size = x.shape[0]
        processed_coords = self.coordinate_preprocessor(crs)
        processed_coords = processed_coords.unsqueeze(-1).unsqueeze(-1)
        processed_coords1 = processed_coords.expand(batch_size, self.coordinate_channel, 128, 128)

        x_joint = torch.cat((x, processed_coords1), dim=1)

        y = self.g_a(x_joint)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)

        processed_coords2 = processed_coords.expand(batch_size, self.coordinate_channel, 8, 8)
        y_joint = torch.cat((y_hat, processed_coords2), dim=1)
        x_hat = self.g_s(y_joint)
        return {"x_hat": x_hat, "likelihoods": {"y": y_likelihoods}}

    def compress(self, x, crs=None, date=None):
        batch_size = x.shape[0]
        processed_coords = self.coordinate_preprocessor(crs)
        processed_coords = processed_coords.unsqueeze(-1).unsqueeze(-1)
        processed_coords = processed_coords.expand(batch_size, self.coordinate_channel, 128, 128)

        x_joint = torch.cat((x, processed_coords), dim=1)

        y = self.g_a(x_joint)
        return self.compress_common(y)
    
    def embedding(self, x, crs=None, date=None):
        batch_size = x.shape[0]
        processed_coords = self.coordinate_preprocessor(crs)
        processed_coords = processed_coords.unsqueeze(-1).unsqueeze(-1)
        processed_coords = processed_coords.expand(batch_size, self.coordinate_channel, 128, 128)

        x_joint = torch.cat((x, processed_coords), dim=1)

        y = self.g_a(x_joint)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        return y, y_hat, self.entropy_bottleneck._quantized_cdf
    
    def decompress(self, strings, shape, crs, date):
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)

        batch_size = y_hat.shape[0]
        processed_coords = self.coordinate_preprocessor(crs)
        processed_coords = processed_coords.unsqueeze(-1).unsqueeze(-1)
        processed_coords = processed_coords.expand(batch_size, self.coordinate_channel, 8, 8)

        y_joint = torch.cat((y_hat, processed_coords), dim=1)

        x_hat = self.g_s(y_joint).clamp_(0, 1)
        return {"x_hat": x_hat}


# class FactorizedPriorCRSDateEnc(FactorizedPriorBase):

# class FactorizedPriorCRSDateDec(FactorizedPriorBase):

# class FactorizedPriorCRSDateEncDec(FactorizedPriorBase):


# ---------------------------------------------------------------------------------------------

class ScaleHyperpriorCRSOnly(ScaleHyperpriorBase):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg,input_channels=3, **kwargs)
        self.coordinate_preprocessor = CoordinatePreprocessor(cfg)
        
        self.h_a_vec = nn.Sequential(
            nn.Linear(self.embedding_size, self.N),
            nn.ReLU(inplace=True),
            nn.Linear(self.N, self.N)
        )

    def forward(self, x, crs, date):
        y = self.g_a(x)
        processed_coords = self.coordinate_preprocessor(crs)
        z_vec = self.h_a_vec(processed_coords)
        embedding_2d = reshape_to_4d(z_vec, self.N, 2)

        z_hat, z_likelihoods = self.entropy_bottleneck(embedding_2d)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        return {"x_hat": self.g_s(y_hat), "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}

    def compress(self, x, crs, date):
        y = self.g_a(x)
        processed_coords = self.coordinate_preprocessor(crs)
        z_vec = self.h_a_vec(processed_coords)
        embedding_2d = reshape_to_4d(z_vec, self.N, 2)

        z_strings = self.entropy_bottleneck.compress(embedding_2d)
        z_hat = self.entropy_bottleneck.decompress(z_strings, embedding_2d.size()[-2:])
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": embedding_2d.size()[-2:]}

class ScaleHyperpriorCRSAddLater(ScaleHyperpriorCRSOnly):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

    def forward(self, x, crs, date):
        y = self.g_a(x)
        z= self.h_a(y)

        processed_coords = self.coordinate_preprocessor(crs)
        processed_coords = processed_coords.view(-1, 64, 1).squeeze()

        z_vec = self.h_a_vec(processed_coords)

        embedding_2d = reshape_to_4d(z_vec, self.N, 2)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        combined_features = torch.add(z_hat, embedding_2d)
        scales_hat = self.h_s(combined_features)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        return {"x_hat": self.g_s(y_hat), "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}

    def compress(self, x, crs, date):
        y = self.g_a(x)
        z= self.h_a(y)
        processed_coords = self.coordinate_preprocessor(crs)
        processed_coords = processed_coords.view(-1, 64, 1).squeeze()
        z_vec = self.h_a_vec(processed_coords)
        embedding_2d = reshape_to_4d(z_vec, self.N, 2)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        combined_features = torch.add(z_hat, embedding_2d)
        scales_hat = self.h_s(combined_features)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
    
    def decompress(self, strings, shape, crs, date):
        processed_coords = self.coordinate_preprocessor(crs)
        processed_coords = processed_coords.view(-1, 64, 1).squeeze()
        z_vec = self.h_a_vec(processed_coords)
        embedding_2d = reshape_to_4d(z_vec, self.N, 2)
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        combined_features = torch.add(z_hat, embedding_2d)
        scales_hat = self.h_s(combined_features)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
    
class ScaleHyperpriorCRSCatLater(ScaleHyperpriorCRSOnly):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

    def forward(self, x, crs, date):
        y = self.g_a(x)
        z = self.h_a(y)
        processed_coords = self.coordinate_preprocessor(crs)
        z_vec = self.h_a_vec(processed_coords)
        embedding_2d = reshape_to_4d(z_vec, self.embedding_size, 2)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        combined_features = torch.cat((z_hat, embedding_2d), dim = 1)
        scales_hat = self.h_s(combined_features)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        return {"x_hat": self.g_s(y_hat), "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}

    def compress(self, x, crs, date):
        y = self.g_a(x)
        z = self.h_a(y)
        processed_coords = self.coordinate_preprocessor(crs)
        z_vec = self.h_a_vec(processed_coords)
        embedding_2d = reshape_to_4d(z_vec, self.embedding_size, 2)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        combined_features = torch.cat((z_hat, embedding_2d), dim = 1)
        scales_hat = self.h_s(combined_features)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
    
    def decompress(self, strings, shape, crs, date):
        processed_coords = self.coordinate_preprocessor(crs)
        z_vec = self.h_a_vec(processed_coords)
        embedding_2d = reshape_to_4d(z_vec, self.embedding_size, 2)
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        combined_features = torch.cat((z_hat, embedding_2d), dim = 1)
        scales_hat = self.h_s(combined_features)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
    

# class ScaleHyperpriorMeta(ScaleHyperpriorBase):
#     def __init__(self, cfg, **kwargs):
#         super().__init__(cfg, input_channels=3, **kwargs)

#         self.coordinate_preprocessor = CoordinatePreprocessor(cfg)

#         self.h_a_img = nn.Sequential(
#             conv(self.M, self.N, stride=1, kernel_size=3),
#             nn.ReLU(inplace=True),
#             conv(self.N, self.N),
#         )

#         self.h_a_vec = nn.Sequential(
#             nn.Linear(self.embedding_size, self.V),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.V, self.N),
#             nn.ReLU(inplace=True),
#         )

#         self.h_a_joint = nn.Sequential(
#             conv(self.N, self.N),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x, v, crs):

#         y = self.g_a(x)
#         z_img = self.h_a_img(y)
#         z_vec = self.h_a_vec(v)
#         z_vec = reshape_to_4d(z_vec, self.N, 4)
#         z_joint = z_img * z_vec
#         z = self.h_a_joint(z_joint)
#         z_hat, z_likelihoods = self.entropy_bottleneck(z)
#         scales_hat = self.h_s(z_hat)
#         y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
#         return {"x_hat": self.g_s(y_hat), "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}

#     def compress(self, x, v, crs):
#         y = self.g_a(x)
#         z_img = self.h_a_img(y)
#         z_vec = self.h_a_vec(v)
#         z_vec = reshape_to_4d(z_vec, self.N, 4)
#         z_joint = z_img * z_vec
#         z = self.h_a_joint(z_joint)

#         z_strings = self.entropy_bottleneck.compress(z)
#         z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
#         scales_hat = self.h_s(z_hat)
#         indexes = self.gaussian_conditional.build_indexes(scales_hat)
#         y_strings = self.gaussian_conditional.compress(y, indexes)
#         return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}


# class ScaleHyperpriorMetaOnly(ScaleHyperpriorBase):
#     def __init__(self, cfg, **kwargs):
#         super().__init__(cfg, **kwargs)
#         self.coordinate_preprocessor = CoordinatePreprocessor(cfg)

#         self.h_a_joint = nn.Sequential(
#             conv(self.N, self.N),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x, v, crs):
#         y = self.g_a(x)
#         z_vec = self.h_a_vec(v)
#         z_vec = reshape_to_4d(z_vec, self.N, 4)
#         z = self.h_a_joint(z_vec)
#         z_hat, z_likelihoods = self.entropy_bottleneck(z)
#         scales_hat = self.h_s(z_hat)
#         y_hat, y_likelihoods = self.gaussian_conditional(self.g_a(x), scales_hat)
#         return {"x_hat": self.g_s(y_hat), "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}

#     def compress(self, x, v, crs):
#         y = self.g_a(x)
#         z_vec = self.h_a_vec(v)
#         z_vec = reshape_to_4d(z_vec, self.N, 4)
#         z = self.h_a_joint(z_vec)
#         z_strings = self.entropy_bottleneck.compress(z)
#         z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
#         scales_hat = self.h_s(z_hat)
#         indexes = self.gaussian_conditional.build_indexes(scales_hat)
#         y_strings = self.gaussian_conditional.compress(y, indexes)
#         return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

#     def decompress(self, strings, shape,crs):
#         z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
#         scales_hat = self.h_s(z_hat)
#         indexes = self.gaussian_conditional.build_indexes(scales_hat)
#         y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
#         x_hat = self.g_s(y_hat).clamp_(0, 1)
#         return {"x_hat": x_hat}


# class ScaleHyperpriorCRSAdd(ScaleHyperpriorMeta):
#     def __init__(self, cfg, **kwargs):
#         super().__init__(cfg, **kwargs)
        
#         self.h_a_vec = nn.Sequential(
#             nn.Linear(self.embedding_size, self.N),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.N, self.N)
#         )

#         self.h_a_joint = nn.Sequential(
#             conv(self.N, self.N, stride=1, kernel_size=3),
#             nn.ReLU(inplace=True),
#             conv(self.N, self.N),
#         )

#     def forward(self, x, v, crs):
#         y = self.g_a(x)
#         z_img = self.h_a_img(y)
#         processed_coords = self.coordinate_preprocessor(crs)
#         z_vec = self.h_a_vec(processed_coords)
#         embedding_2d = reshape_to_4d(z_vec, self.N, 4)

#         combined_features = torch.add(z_img, embedding_2d)
#         z = self.h_a_joint(combined_features)

#         z_hat, z_likelihoods = self.entropy_bottleneck(z)
#         scales_hat = self.h_s(z_hat)
#         y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
#         return {"x_hat": self.g_s(y_hat), "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}

#     def compress(self, x, v, crs):
#         y = self.g_a(x)
#         z_img = self.h_a_img(y)
#         processed_coords = self.coordinate_preprocessor(crs)
#         z_vec = self.h_a_vec(processed_coords)
#         embedding_2d = reshape_to_4d(z_vec, self.N, 4)
#         combined_features = torch.add(z_img, embedding_2d)
#         z = self.h_a_joint(combined_features)
#         z_strings = self.entropy_bottleneck.compress(z)
#         z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
#         scales_hat = self.h_s(z_hat)
#         indexes = self.gaussian_conditional.build_indexes(scales_hat)
#         y_strings = self.gaussian_conditional.compress(y, indexes)
#         return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
    
# class ScaleHyperpriorCRSCat(ScaleHyperpriorMeta):
#     def __init__(self, cfg, **kwargs):
#         super().__init__(cfg, **kwargs)
        
#         self.h_a_vec = nn.Sequential(
#             nn.Linear(self.embedding_size, self.N),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.N, self.embedding_size)
#         )

#         self.h_a_joint = nn.Sequential(
#             conv(self.N+self.embedding_size, self.N, stride=1, kernel_size=3),
#             nn.ReLU(inplace=True),
#             conv(self.N, self.N),
#         )

#     def forward(self, x, v, crs):
#         y = self.g_a(x)
#         z_img = self.h_a_img(y)
#         processed_coords = self.coordinate_preprocessor(crs)
#         z_vec = self.h_a_vec(processed_coords)
#         embedding_2d = reshape_to_4d(z_vec, self.embedding_size, 4)

#         combined_features = torch.cat((z_img, embedding_2d), dim = 1)
#         z = self.h_a_joint(combined_features)

#         z_hat, z_likelihoods = self.entropy_bottleneck(z)
#         scales_hat = self.h_s(z_hat)
#         y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
#         return {"x_hat": self.g_s(y_hat), "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}

#     def compress(self, x, v, crs):
#         y = self.g_a(x)
#         z_img = self.h_a_img(y)
#         processed_coords = self.coordinate_preprocessor(crs)
#         z_vec = self.h_a_vec(processed_coords)
#         embedding_2d = reshape_to_4d(z_vec, self.embedding_size, 4)
#         combined_features = torch.cat((z_img, embedding_2d), dim =1)
#         z = self.h_a_joint(combined_features)
#         z_strings = self.entropy_bottleneck.compress(z)
#         z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
#         scales_hat = self.h_s(z_hat)
#         indexes = self.gaussian_conditional.build_indexes(scales_hat)
#         y_strings = self.gaussian_conditional.compress(y, indexes)
#         return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

# class ScaleHyperpriorCRSAddOnly(ScaleHyperpriorMeta):
#     def __init__(self, cfg, **kwargs):
#         super().__init__(cfg, **kwargs)
        
#         self.h_a_vec = nn.Sequential(
#             nn.Linear(self.embedding_size, self.N),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.N, self.N)
#         )


#     def forward(self, x, v, crs):
#         y = self.g_a(x)
#         processed_coords = self.coordinate_preprocessor(crs)
#         z_vec = self.h_a_vec(processed_coords)
#         embedding_2d = reshape_to_4d(z_vec, self.N, 2)

#         z_hat, z_likelihoods = self.entropy_bottleneck(embedding_2d)
#         scales_hat = self.h_s(z_hat)
#         y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
#         return {"x_hat": self.g_s(y_hat), "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}

#     def compress(self, x, v, crs):
#         y = self.g_a(x)
#         processed_coords = self.coordinate_preprocessor(crs)
#         z_vec = self.h_a_vec(processed_coords)
#         embedding_2d = reshape_to_4d(z_vec, self.N, 2)
#         z_strings = self.entropy_bottleneck.compress(embedding_2d)
#         z_hat = self.entropy_bottleneck.decompress(z_strings, embedding_2d.size()[-2:])
#         scales_hat = self.h_s(z_hat)
#         indexes = self.gaussian_conditional.build_indexes(scales_hat)
#         y_strings = self.gaussian_conditional.compress(y, indexes)
#         return {"strings": [y_strings, z_strings], "shape": embedding_2d.size()[-2:]}
    
    
# class ScaleHyperpriorCRSOnly(ScaleHyperpriorBase):
#     def __init__(self, cfg, **kwargs):
#         super().__init__(cfg, input_channels=3, **kwargs)
#         self.coordinate_preprocessor = CoordinatePreprocessor(cfg)

#         self.h_a_vec = nn.Sequential(
#             nn.Linear(self.embedding_size, self.V),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.V, self.N )
#         )

#         self.h_a_joint = nn.Sequential(
#             conv(self.N, self.N, stride=1, kernel_size=3),
#             nn.ReLU(inplace=True),
#             conv(self.N, self.N),
#         )

#     def forward(self, x, v, crs):
#         processed_coords = self.coordinate_preprocessor(crs)

#         z_vec = self.h_a_vec(processed_coords)
#         embedding_2d = reshape_to_4d(z_vec, self.N, 4)
#         z = self.h_a_joint(embedding_2d)
#         z_hat, z_likelihoods = self.entropy_bottleneck(z)
#         scales_hat = self.h_s(z_hat)
#         y_hat, y_likelihoods = self.gaussian_conditional(self.g_a(x), scales_hat)
#         return {"x_hat": self.g_s(y_hat), "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}

#     def compress(self, x, v, crs):
#         processed_coords = self.coordinate_preprocessor(crs)
#         z_vec = self.h_a_vec(processed_coords)
#         embedding_2d = reshape_to_4d(z_vec, self.N, 4)
#         z = self.h_a_joint(embedding_2d)
#         z_strings = self.entropy_bottleneck.compress(z)
#         z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
#         scales_hat = self.h_s(z_hat)
#         indexes = self.gaussian_conditional.build_indexes(scales_hat)
#         y_strings = self.gaussian_conditional.compress(self.g_a(x), indexes)
#         return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

#     def decompress(self, strings, shape):
#         z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
#         scales_hat = self.h_s(z_hat)
#         indexes = self.gaussian_conditional.build_indexes(scales_hat)
#         y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
#         x_hat = self.g_s(y_hat).clamp_(0, 1)
#         return {"x_hat": x_hat}