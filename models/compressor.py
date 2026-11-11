import os
import torch
import torch.nn as nn
from diffusers import AutoencoderKLKVAEAudio
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


class KVAEAudioCompressor(nn.Module):
    """
    Continuous 1D Audio VAE wrapper for 1D signals.
    Encodes raw 1D spectrum -> latent vector z.
    """

    def __init__(self, local_model_path: str = "models/pretrained/KVAE-Audio"):
        super().__init__()

        # Load directly from your local directory without connecting to HF
        self.vae = AutoencoderKLKVAEAudio.from_pretrained(
            local_model_path,
            subfolder="diffusers",
            local_files_only=True,
            torch_dtype=torch.float32,
        )

        # Freeze all VAE weights
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()

    @torch.no_grad()
    def encode(self, x):
        # Input shape: (B, L) -> (B, 1, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        posterior = self.vae.encode(x).latent_dist
        z = posterior.mode()  # Shape: (B, 64, L_latent)

        # Flatten across latent dimensions to fit DiffWave's 1D signal space (B, -1)
        return z.reshape(z.shape[0], -1)

    @torch.no_grad()
    def decode(self, z):
        # Reshape flat 1D latent (B, 64 * L_latent) back to (B, 64, L_latent)
        if z.dim() == 2:
            z = z.view(z.shape[0], 64, -1)

        x_rec = self.vae.decode(z).sample  # (B, 1, L)

        if x_rec.dim() == 3 and x_rec.size(1) == 1:
            x_rec = x_rec.squeeze(1)

        return x_rec


def create_compressor(config: dict, device: torch.device) -> nn.Module:
    vae_cfg = config.get("vae", {})
    if not vae_cfg.get("enabled", False):
        print(">> Compressor: IdentityCompressor (Operating in raw signal space)")
        return IdentityCompressor().to(device)

    vae_type = vae_cfg.get("type", "spectrum_vae").lower()

    # Pre-trained Audio VAE from local Hugging Face download
    if vae_type in ("audio_vae", "kvae_audio"):
        model_path = vae_cfg.get("local_path", "models/pretrained/KVAE-Audio")
        print(f">> Loading frozen KVAE Audio Compressor from: {model_path}")
        compressor = KVAEAudioCompressor(local_model_path=model_path).to(device)
        print(">> Audio VAE Compressor initialized and weights frozen.")
        return compressor

    # Custom 1D SpectrumVAE checkpoint via VAECompressor
    elif vae_type == "spectrum_vae":
        ckpt_path = vae_cfg.get("checkpoint_path", "")
        latent_dim = vae_cfg.get("latent_dim", 2048)
        raw_signal_len = config["paths"].get("raw_signal_len", 16384)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"VAE checkpoint not found at: {ckpt_path}")

        print(f">> Loading frozen SpectrumVAE Compressor (latent_dim={latent_dim}) from: {ckpt_path}")
        vae = SpectrumVAE(L=raw_signal_len, latent_dim=latent_dim)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        vae.load_state_dict(state_dict)

        compressor = VAECompressor(vae).to(device)
        print(">> Custom VAE Compressor initialized and weights frozen.")
        return compressor

    else:
        raise ValueError(f"Unsupported vae.type '{vae_type}'. Choose 'kvae_audio' or 'spectrum_vae'.")