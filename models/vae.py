import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectrumVAE(nn.Module):
    def __init__(self, L=8192, latent_dim=32):
        """
        1D Convolutional VAE for Gamma Spectra.
        L must be a multiple of 1024 (e.g., 4096, 8192, 16384).
        """
        super(SpectrumVAE, self).__init__()
        self.L = L

        # Calculate the dynamic bottleneck spatial length
        # 5 layers with stride=4 means total downsampling is 4^5 = 1024
        assert L % 1024 == 0, "Input length L must be cleanly divisible by 1024."
        self.bottleneck_len = L // 1024

        # ==========================================
        # 1. ENCODER (Conv1D)
        # ==========================================
        self.enc1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=4, stride=4)
        self.enc2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4, stride=4)
        self.enc3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=4)
        self.enc4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4, stride=4)
        self.enc5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=4, stride=4)

        # Dynamic flatten size: 256 channels * bottleneck length
        self.flatten_size = 256 * self.bottleneck_len

        # Latent Space Projections
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

        # ==========================================
        # 2. DECODER (ConvTranspose1D)
        # ==========================================
        # First project the latent vector back to the dynamic flattened size
        self.dec_fc = nn.Linear(latent_dim, self.flatten_size)

        self.dec1 = nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=4, stride=4)
        self.dec2 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=4, stride=4)
        self.dec3 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=4, stride=4)
        self.dec4 = nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=4, stride=4)
        self.dec5 = nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=4, stride=4)

    def encode(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        h = F.relu(self.enc3(h))
        h = F.relu(self.enc4(h))
        h = F.relu(self.enc5(h))

        # Flatten the spatial dimensions for the dense layers
        h = h.view(h.size(0), -1)

        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Map latent variable back to spatial tensor
        h = F.relu(self.dec_fc(z))

        # Dynamically reshape using the calculated bottleneck length
        h = h.view(h.size(0), 256, self.bottleneck_len)

        # Upsample
        h = F.relu(self.dec1(h))
        h = F.relu(self.dec2(h))
        h = F.relu(self.dec3(h))
        h = F.relu(self.dec4(h))

        # Final layer uses ReLU to enforce strictly positive physical counts
        out = F.relu(self.dec5(h))
        return out

    def forward(self, x):
        # Add the 'channel' dimension required by Conv1D: (N, L) -> (N, 1, L)
        x = x.unsqueeze(1)

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)

        # Remove the 'channel' dimension so it matches the original target: (N, 1, L) -> (N, L)
        reconstructed_x = reconstructed_x.squeeze(1)

        return reconstructed_x, mu, logvar


def vae_loss_function(reconstructed_x, x, mu, logvar, beta = 1.0):
    MSE = F.mse_loss(reconstructed_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + (beta * KLD)