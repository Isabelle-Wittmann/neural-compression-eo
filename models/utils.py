import torch
import torch.nn as nn
import numpy as np 

# BigEarthNet Range
lat_min, lat_max = 36.838 , 70.092 
lon_min, lon_max = -10.474, 31.586
lat_mean, lat_std = 51.787555, 9.31327816 
lon_mean, lon_std = 13.57538875, 14.51931266

def standardize_lat_lon(lat, lon):
    standardized_lat = (lat - lat_mean) / lat_std
    standardized_lon = (lon - lon_mean) / lon_std
    return standardized_lat, standardized_lon

class LatLongEmbedding(nn.Module):
    def __init__(self, lat_bins, lon_bins, embedding_dim):
        super(LatLongEmbedding, self).__init__()

        self.lat_embedding = nn.Embedding(lat_bins, embedding_dim)
        self.lon_embedding = nn.Embedding(lon_bins, embedding_dim)
    
    def forward(self, latitudes, longitudes):
        lat_embeds = self.lat_embedding(latitudes)
        lon_embeds = self.lon_embedding(longitudes)
        
        combined_embeds = torch.cat([lat_embeds, lon_embeds], dim=-1) 
        
        return combined_embeds

def input_fn(lat, lon):
    # latitude and longitude ranges for BigEarthNet

    bins = 100  # Number of buckets

    # latitude and longitude buckets
    latitude_buckets = np.linspace(lat_min, lat_max, bins-1)
    longitude_buckets = np.linspace(lon_min, lon_max, bins-1)

    lat_indices = np.digitize(lat, latitude_buckets)
    lon_indices = np.digitize(lon, longitude_buckets)

    # indices to tensors
    lat_indices = torch.tensor(lat_indices, dtype=torch.long)
    lon_indices = torch.tensor(lon_indices, dtype=torch.long)

    return lat_indices, lon_indices

def reshape_to_4d(input_tensor, channel_dim, spatial_dim):
    if input_tensor.dim() > 2:
        raise ValueError("Input tensor must be 2d")

    output_tensor = input_tensor.view(-1, channel_dim, 1, 1)
    return output_tensor.repeat(1, 1, spatial_dim, spatial_dim)

import numpy as np

def lat_lon_to_radians(lat, lon):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    return lat_rad, lon_rad

def positional_encoding(lat_rad, lon_rad, d_model=64):
    """
    d_model (int): The dimension of the output encoding (must be even).
    """
    assert d_model % 2 == 0
    
    # arrays of positions and frequencies
    positions = np.arange(d_model // 4)
    frequencies = 1 / (10000 ** (2 * positions / d_model))
    lat_rad = np.expand_dims(lat_rad, 1)
    lon_rad = np.expand_dims(lon_rad,1)
    
    # Apply sine and cosine functions to lat/lon with different frequencies
    lat_enc = np.concatenate([np.sin(lat_rad * frequencies), np.cos(lat_rad * frequencies)], axis=1)
    lon_enc = np.concatenate([np.sin(lon_rad * frequencies), np.cos(lon_rad * frequencies)], axis=1)

    encoding = np.concatenate([lat_enc, lon_enc], axis=1)
    
    return encoding


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, lat, lon, device):
        lat_rad, lon_rad = lat_lon_to_radians(lat, lon)
        encoding = positional_encoding(lat_rad, lon_rad, self.embedding_dim)

        return torch.Tensor(encoding).to(device)


class PositionalEncodingRandom(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, lat, lon, device):
        lat, lon = np.random.normal(size=2)
        lat_rad, lon_rad = lat_lon_to_radians(lat, lon)
        encoding = positional_encoding(lat_rad, lon_rad, self.embedding_dim)

        return torch.Tensor(encoding).to(device)
    
class SinCosEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lat, lon, device):
  
        lon_sin = np.sin(2 * np.pi * lon / 360)
        lon_cos = np.cos(2 * np.pi * lon / 360)
        
        coord_vec = np.stack((lat, lon_sin, lon_cos), axis=1)
        # coord_vec = torch.swapaxes(torch.tensor(coord_vec), 0, 1)
        return torch.tensor(coord_vec).to(device)

class EmbeddingLayer(nn.Module):
    def __init__(self, num_bins, embedding_dim):
        super().__init__()
        self.bins = num_bins
        self.embedding_dim = embedding_dim
        self.embed = LatLongEmbedding(self.bins, self.bins, self.embedding_dim)
        
    def forward(self, lat, lon, device):
        lat_indices, lon_indices = input_fn(lat, lon)

        embeddings = self.embed(lat_indices.to(device), lon_indices.to(device))
        
        return embeddings


class CoordinatePreprocessor(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        method = cfg['preprocessing']['coordinate_encoding']
        num_bins = cfg['preprocessing']['coordinate_num_bins']
        embedding_dim = cfg['preprocessing']['coordinate_embedding_dim']

        if method == 'sincos':
            self.preprocessor = SinCosEncoding()
        elif method == 'embedding':
            self.preprocessor = EmbeddingLayer(num_bins=num_bins, embedding_dim=embedding_dim)
        elif method == 'positional':
            self.preprocessor = PositionalEncoding(embedding_dim=embedding_dim)
        elif method == 'positional_random':
            self.preprocessor = PositionalEncodingRandom(embedding_dim=embedding_dim)
        
        else:
            raise ValueError(f"Unknown method: {method}")

    def forward(self, crs):
        device = crs.device
        lon = np.array(crs.cpu()[:, 0])
        lat = np.array(crs.cpu()[:, 1])
        lat, lon = standardize_lat_lon(lat, lon)

        return self.preprocessor(lat, lon, device)