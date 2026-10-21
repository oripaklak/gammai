import torch
import torch.nn as nn

class IdentityCompressor(nn.Module):
    """Passthrough when running diffusion directly in raw 1D signal space."""
    def encode(self,x):
        return x
    
    def decode(self,z):
        return z
    
class VAECompressor(nn.Module):
    """
    Wrapper for pre-trained 1D VAE.
    Freeze weights to keep the compressed hypothesis space fixed.
    """
    def __init__(self, vae_model):
        super().__init__()
        self.vae = vae_model
        # Freeze all parameters
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()
        
    @torch.no_grad
    def encode(self, x):
        z = self.vae.encode(x)
        return z
    
    @torch.no_grad
    def decode(self, z):
        x = self.vae.decode(z)
        return x