from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):

    def __init__(self, in_features, latent_size):
        super().__init__()

        self.in_features = in_features
        self.encoder = nn.Sequential(
            nn.Linear(in_features, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size // 2),
            nn.ReLU(),
            nn.Linear(latent_size // 2, latent_size // 4),
            nn.ReLU(),
            nn.Linear(latent_size // 4, latent_size // 8),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size // 8, latent_size // 4),
            nn.ReLU(),
            nn.Linear(latent_size // 4, latent_size // 2),
            nn.ReLU(),
            nn.Linear(latent_size // 2, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, in_features)
        )
    
    def forward(self, cell_feats):

        hidden_feats = self.encoder(cell_feats)
        recon_feats = self.decoder(hidden_feats)
        
        return recon_feats, hidden_feats