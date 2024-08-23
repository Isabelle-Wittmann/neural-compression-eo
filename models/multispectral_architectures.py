import math
import torch.nn.functional as f
from .base_architectures import *
from torch import nn

class ConvolutionalAutoencoder1D(nn.Module):
    """
    Title:
        1D-CONVOLUTIONAL AUTOENCODER BASED HYPERSPECTRAL DATA COMPRESSION
    Authors:
        Kuester, Jannick and Gross, Wolfgang and Middelmann, Wolfgang
    Paper:
        https://doi.org/10.5194/isprs-archives-XLIII-B1-2021-15-2021
    Cite:
        @article{kuester20211d,
            title={1D-convolutional autoencoder based hyperspectral data compression},
            author={Kuester, Jannick and Gross, Wolfgang and Middelmann, Wolfgang},
            journal={International Archives of Photogrammetry, Remote Sensing and Spatial Information Sciences},
            volume={43},
            pages={15--21},
            year={2021},
            publisher={Copernicus GmbH}
        }
    """

    def __init__(self, src_channels=103):
        super(ConvolutionalAutoencoder1D, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=64,
                kernel_size=11,
                stride=1,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=64,
                out_channels=32,
                kernel_size=11,
                stride=1,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=32,
                out_channels=16,
                kernel_size=9,
                stride=1,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=16,
                out_channels=1,
                kernel_size=7,
                stride=1,
                padding="same",
            ),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=9,
                stride=1,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Upsample(
                scale_factor=2
            ),
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=11,
                stride=1,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Upsample(
                scale_factor=2
            ),
            nn.Conv1d(
                in_channels=64,
                out_channels=1,
                kernel_size=11,
                stride=1,
                padding="same",
            ),
            nn.Sigmoid(),
        )

        self.src_channels = src_channels

        self.spectral_downsamplings = 2
        self.spectral_downsampling_factor_estimated = 2 ** self.spectral_downsamplings

        self.spatial_downsamplings = 0
        self.spatial_downsampling_factor = 2 ** self.spatial_downsamplings

        self.latent_channels = int(math.ceil(self.src_channels / 2 ** self.spectral_downsamplings))
        self.spectral_downsampling_factor = self.src_channels / self.latent_channels
        self.compression_ratio = self.spectral_downsampling_factor * self.spatial_downsampling_factor ** 2
        self.bpppc = 32.0 / self.compression_ratio

        self.padding_amount = 0 if self.src_channels % self.spectral_downsampling_factor_estimated == 0 \
            else self.spectral_downsampling_factor_estimated - self.src_channels % self.spectral_downsampling_factor_estimated

    def compress(self, x):
        print(x.shape)
        if self.padding_amount > 0:
            x = f.pad(x, (self.padding_amount, 0))
        x = x.unsqueeze(1)
        print(x.shape)
        y = self.encoder(x)
        y = y.squeeze(1)

        return y

    def decompress(self, y):
        y = y.unsqueeze(1)
        x_hat = self.decoder(y)

        if self.padding_amount > 0:
            x_hat = x_hat[:, :, self.padding_amount:]
        x_hat = x_hat.squeeze(1)

        return x_hat

    def forward(self, x):

        y = self.compress(x)
        x_hat = self.decompress(y)
        return x_hat

    @classmethod
    def from_state_dict(cls, state_dict):
        net = cls()
        net.load_state_dict(state_dict)
        return net



class ConvolutionalAutoencoder3D(nn.Module):
    """
    Title:
        END-TO-END JOINT SPECTRAL-SPATIAL COMPRESSION AND RECONSTRUCTION OF HYPERSPECTRAL IMAGES USING A 3D CONVOLUTIONAL AUTOENCODER
    Authors:
        Chong, Yanwen and Chen, Linwei and Pan, Shaoming
    Paper:
        https://doi.org/10.1117/1.JEI.30.4.041403
    Cite:
        @article{chong2021end,
            title={End-to-end joint spectral--spatial compression and reconstruction of hyperspectral images using a 3D convolutional autoencoder},
            author={Chong, Yanwen and Chen, Linwei and Pan, Shaoming},
            journal={Journal of Electronic Imaging},
            volume={30},
            number={4},
            pages={041403},
            year={2021},
            publisher={SPIE}
        }
    """

    def __init__(self, src_channels=12, latent_channels=16):
        super(ConvolutionalAutoencoder3D, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=16,
                kernel_size=(2, 5, 5),
                stride=(1, 2, 2),
                padding=(0, 2, 2),
            ),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=(5, 5, 5),
                stride=(2, 2, 2),
                padding=(2, 2, 2),
            ),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            _ResBlock(32),
            _ResBlock(32),
            _ResBlock(32),
            nn.Conv3d(
                in_channels=32,
                out_channels=latent_channels,
                kernel_size=(5, 5, 5),
                stride=(2, 2, 2),
                padding=(2, 2, 2),
            ),
            nn.BatchNorm3d(latent_channels),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=(2, 4, 4)),
            nn.Conv3d(
                in_channels=latent_channels,
                out_channels=32,
                kernel_size=(5, 5, 5),
                stride=(2, 2, 2),
                padding=(2, 2, 2),
            ),
            # nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            _ResBlock(32),
            _ResBlock(32),
            _ResBlock(32),
            nn.Upsample(scale_factor=(4, 2, 2)),
            nn.Conv3d(
                in_channels=32,
                out_channels=16,
                kernel_size=(5, 3, 3),
                stride=(2, 1, 1),
                padding=(2, 1, 1),
            ),
            # nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=(4, 2, 2)),
            nn.Conv3d(
                in_channels=16,
                out_channels=1,
                kernel_size=(2, 3, 3),
                stride=(2, 1, 1),
                padding=(0, 1, 1),  # (2, 1, 1)
            ),
            # nn.BatchNorm3d(1),
            nn.LeakyReLU(),
        )

        self.src_channels = src_channels
        self.latent_channels = latent_channels


        self.spectral_downsamplings = 2
        self.spectral_downsampling_factor_estimated = 2 ** self.spectral_downsamplings
        self.spectral_downsampling_factor = self.spectral_downsampling_factor_estimated

        # self.padding_amount = 0 if self.src_channels % self.spectral_downsampling_factor_estimated == 0 \
        #     else self.spectral_downsampling_factor_estimated - self.src_channels % self.spectral_downsampling_factor_estimated

        # self.spectral_downsampling_factor = self.src_channels / ((self.src_channels + self.padding_amount) / self.spectral_downsampling_factor_estimated)

        self.spatial_downsamplings = 3
        self.spatial_downsampling_factor = 2 ** self.spatial_downsamplings

        self.compression_ratio = self.spectral_downsampling_factor * self.spatial_downsampling_factor ** 2 / self.latent_channels
        self.bpppc = 32.0 / self.compression_ratio


    def forward(self, x):

        # if self.padding_amount > 0:
        #     x = f.pad(x, (0, 0, 0, 0, self.padding_amount, 0))
        x = x.unsqueeze(1)

        y = self.encoder(x)
        x_hat = self.decoder(y)
        # if self.padding_amount > 0:
        #     x_hat = x_hat[:, :, self.padding_amount:]
        x_hat = x_hat.squeeze(1)

        return x_hat

    def compress(self, x):
        # if self.padding_amount > 0:
        #     x = f.pad(x, (0, 0, 0, 0, self.padding_amount, 0))
        x = x.unsqueeze(1)
        y = self.encoder(x)
        y = y.squeeze(1)
        return y

    def decompress(self, y):
        x_hat = self.decoder(y)
        # if self.padding_amount > 0:
        #     x_hat = x_hat[:, :, self.padding_amount:]
        x_hat = x_hat.squeeze(1)
        return x_hat


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
            nn.BatchNorm3d(channels),
            self.act,
            nn.Conv3d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(channels),
        ])

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.act(out)
        return out


class HyperspectralCompressionTransformer(nn.Module):
    def __init__(
            self,
            src_channels=202,
            target_compression_ratio=4,
            patch_depth=4,
            hidden_dim=1024,
            dim=64,
            depth=5,
            heads=4,
            mlp_dim=8,
            dim_head=16,
            dropout=0.,
            emb_dropout=0.,
        ):
        super().__init__()

        self.src_channels = src_channels

        self.dim = dim

        latent_channels = int(math.ceil(src_channels / target_compression_ratio))
        self.latent_channels = latent_channels

        self.compression_ratio = src_channels / latent_channels
        self.bpppc = 32 / self.compression_ratio

        self.delta_pad = int(math.ceil(src_channels / patch_depth)) * patch_depth - src_channels

        num_patches = (src_channels + self.delta_pad) // patch_depth
        self.num_patches = num_patches

        patch_dim = (src_channels + self.delta_pad) // num_patches
        self.patch_dim = patch_dim
        
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.comp_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, 'ViT')

        self.to_latent = nn.Sequential(
            nn.Linear(
                in_features=dim,
                out_features=hidden_dim,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=hidden_dim,
                out_features=latent_channels,
            ),
            nn.Sigmoid(),
        )

        self.patch_deembed = nn.Sequential(
            nn.Linear(
                in_features=latent_channels,
                out_features=hidden_dim,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=hidden_dim,
                out_features=src_channels,
            ),
            nn.Sigmoid(),
        )

    def compress(self, x):
        _, _, h, w = x.shape

        if self.delta_pad > 0:
            x = f.pad(x, (0, 0, 0, 0, self.delta_pad, 0))

        x = rearrange(x, 'b (n pd) w h -> (b w h) n pd',
                      n = self.num_patches,
                      pd = self.patch_dim,
                      )

        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        # concat compression tokens
        comp_tokens = repeat(self.comp_token, '() n d -> b n d', b = b)
        x = torch.cat((comp_tokens, x), dim = 1)

        # add position embedding
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x = self.transformer(x)

        # extract transformed comp_tokens
        y = x[:, 0]
        
        y = self.to_latent(y)

        y = rearrange(y, '(b w h) d -> b d w h',
                      d = self.latent_channels,
                      w = w,
                      h = h,
                      )

        return y

    def decompress(self, y):
        y = rearrange(y, 'b d w h -> b w h d')
        
        x_hat = self.patch_deembed(y)
        
        x_hat = rearrange(x_hat, 'b w h c -> b c w h')

        return x_hat
    
    def forward(self, x):
        y = self.compress(x)
        x_hat = self.decompress(y)
        return x_hat


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask = None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = f.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout = dropout)))
            ]))

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth-2):
            self.skipcat.append(nn.Conv2d(num_channel+1, num_channel+1, [1, 2], 1, 0))

    def forward(self, x, mask = None):
        if self.mode == 'ViT':
            for attn, ff in self.layers:
                x = attn(x, mask = mask)
                x = ff(x)
        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:           
                last_output.append(x)
                if nl > 1:             
                    x = self.skipcat[nl-2](torch.cat([x.unsqueeze(3), last_output[nl-2].unsqueeze(3)], dim=3)).squeeze(3)
                x = attn(x, mask = mask)
                x = ff(x)
                nl += 1

        return x



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
