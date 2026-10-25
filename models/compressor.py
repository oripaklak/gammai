import os
import torch
import torch.nn as nn
from models.vae import SpectrumVAE


class IdentityCompressor(nn.Module):
    """Passthrough for diffusion directly in raw 1D signal space."""

    def __init__(self):
        super().__init__()

    def encode(self, x):
        return x

    def decode(self, z):
        return z


class VAECompressor(nn.Module):
    """
    Wrapper for pre-trained 1D VAE.
    Freezes parameters and extracts latent representation mu.
    """

    def __init__(self, vae_model: nn.Module):
        super().__init__()
        self.vae = vae_model
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()

    @torch.no_grad()
    def encode(self, x):
        # Add channel dimension if missing: (B, L) -> (B, 1, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        res = self.vae.encode(x)

        # Extract mean vector mu if encode returns a tuple (mu, logvar)
        z = res[0] if isinstance(res, tuple) else res

        # Ensure latent representation is 2D: (B, latent_dim)
        if z.dim() == 3 and z.size(1) == 1:
            z = z.squeeze(1)

        return z

    @torch.no_grad()
    def decode(self, z):
        x = self.vae.decode(z)

        # Ensure decoded output is 2D: (B, L)
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)

        return x


def create_compressor(config: dict, device: torch.device) -> nn.Module:
    vae_cfg = config.get("vae", {})
    if not vae_cfg.get("enabled", False):
        print(">> Compressor: IdentityCompressor (Operating in raw signal space)")
        return IdentityCompressor().to(device)

    ckpt_path = vae_cfg.get("checkpoint_path",{})
    latent_dim = vae_cfg.get("latent_dim", 2048)
    raw_signal_len = config["paths"].get("raw_signal_len", 16384)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"VAE checkpoint not found at: {ckpt_path}")

    print(f">> Loading frozen VAE Compressor (latent_dim={latent_dim}) from: {ckpt_path}")
    vae = SpectrumVAE(L=raw_signal_len, latent_dim=latent_dim)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    vae.load_state_dict(state_dict)

    compressor = VAECompressor(vae).to(device)
    print(">> VAE Compressor initialized and weights frozen.")
    return compressor