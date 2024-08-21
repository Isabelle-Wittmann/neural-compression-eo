import torch
import torch.nn as nn
import numpy as np 

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
    lon_min, lon_max = 36.838 , 70.092 
    lat_min, lat_max = -10.474, 31.586
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

def reshape_to_4d(input_tensor):
    if input_tensor.dim() > 2:
        raise ValueError("Input tensor must have shape (batch, x)")

    output_tensor = input_tensor.view(-1, 192, 1, 1)
    return output_tensor.repeat(1, 1, 4, 4)

# Coordinate Preprocessing Methods

class SinCosEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, crs):
        lon = np.array(crs.cpu()[:, 0])
        lat = np.array(crs.cpu()[:, 1])
        
        lon_sin = np.sin(2 * np.pi * lon / 360)
        lon_cos = np.cos(2 * np.pi * lon / 360)
        
        coord_vec = np.stack((lat, lon_sin, lon_cos), axis=1)
        # coord_vec = torch.swapaxes(torch.tensor(coord_vec), 0, 1)
        return torch.tensor(coord_vec).to(crs.device)

class EmbeddingLayer(nn.Module):
    def __init__(self, num_bins, embedding_dim):
        super().__init__()
        self.bins = num_bins
        self.embedding_dim = embedding_dim
        self.embed = LatLongEmbedding(self.bins, self.bins, self.embedding_dim)
        
    def forward(self, crs):
        lon = np.array(crs.cpu()[:, 0])
        lat = np.array(crs.cpu()[:, 1])
        lat_indices, lon_indices = input_fn(lat, lon)

        embeddings = self.embed(lat_indices.to(crs.device), lon_indices.to(crs.device))
        
        return embeddings

class CoordinatePreprocessor(nn.Module):
    def __init__(self, method='sincos', num_bins=100, embedding_dim=64):
        super().__init__()
        if method == 'sincos':
            self.preprocessor = SinCosEncoding()
        elif method == 'embedding':
            self.preprocessor = EmbeddingLayer(num_bins=num_bins, embedding_dim=embedding_dim)
        else:
            raise ValueError(f"Unknown method: {method}")

    def forward(self, crs):
        return self.preprocessor(crs)