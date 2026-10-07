import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

from models.film import FiLMGenerator

Linear = nn.Linear


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step] # [B,128]
        else:
            x = self._lerp_embedding(diffusion_step) # [B,128]
        x = self.projection1(x) # [B,512]
        x = silu(x)
        x = self.projection2(x) # [B,512]
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long() # [B]
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]   # [B,128]
        high = self.embedding[high_idx] # [B,128]
        scale = (t - low_idx).unsqueeze(-1) # [B,1]
        return low + (high - low) * scale

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)  # [1,64]
        table = steps * 10.0 ** (dims * 4.0 / 63.0)  # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation, film_location='before_output_proj'):
        '''
    :param residual_channels: signal conv channels
    :param dilation: signal conv dilation
    '''
        super().__init__()
        self.film_location = film_location
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(512, residual_channels)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)


    def forward(self, x, diffusion_step, gamma=None, beta=None):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step

        # Unconditional processing only
        y = self.dilated_conv(y) # [B,2C,L]

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter) # [B,C,L]

        # Apply FiLM before Output Projection (Channels = C)
        if self.film_location == 'before_output_proj' and gamma is not None and beta is not None:
            y = (1.0 + gamma.unsqueeze(-1)) * y + beta.unsqueeze(-1)

        y = self.output_projection(y) # [B,2C,L]

        # Apply FiLM after Output Projection (Channels = 2C)
        if self.film_location == 'after_output_proj' and gamma is not None and beta is not None:
            y = (1.0 + gamma.unsqueeze(-1)) * y + beta.unsqueeze(-1)

        residual, skip = torch.chunk(y, 2, dim=1) # residual [B,C,L], skip [B,C,L]
        return (x + residual) / sqrt(2.0), skip


class DiffWave(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        mod_cfg = config['model']  # Access the model block
        calib_cfg = config.get('calibration', {})

        cond_method = calib_cfg.get('conditioning_method', 'concat')
        self.use_film = calib_cfg.get('enabled', False) and cond_method == 'film'
        self.num_calib_params = calib_cfg.get('num_params', 4)
        self.film_location = calib_cfg.get('film_location', 'before_output_proj')

        self.input_projection = Conv1d(1, mod_cfg['residual_channels'], 1)
        self.diffusion_embedding = DiffusionEmbedding(mod_cfg['T'])

        # FiLM channels per layer depend on insertion point
        film_channels = mod_cfg['residual_channels'] if self.film_location == 'before_output_proj' else 2 * mod_cfg['residual_channels']

        if self.use_film:
            self.film_generator = FiLMGenerator(
                num_params=self.num_calib_params,
                num_layers=mod_cfg['residual_layers'],
                channels_per_layer=film_channels
            )
        else:
            self.film_generator = None

        self.residual_layers = nn.ModuleList([
            ResidualBlock(residual_channels=mod_cfg['residual_channels'],
                          dilation=2 ** (i % mod_cfg['dilation_cycle_length']),
                          film_location=self.film_location
                          )
            for i in range(mod_cfg['residual_layers'])
        ])
        self.skip_projection = Conv1d(mod_cfg['residual_channels'], mod_cfg['residual_channels'], 1)
        self.output_projection = Conv1d(mod_cfg['residual_channels'], 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, signal, diffusion_step, conditioner=None):
        x = signal.unsqueeze(1)
        x = self.input_projection(x)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)

        # Generate all layer modulation globally
        if self.use_film and conditioner is not None:
            gammas, betas = self.film_generator(conditioner) # Shapes: [B,N,C]
        else:
            gammas,betas = None, None

        skip = None
        for i, layer in enumerate(self.residual_layers):
            g_i = gammas[:, i, :] if gammas is not None else None # [B,C]
            b_i = betas[:, i, :] if betas is not None else None #[B,C]

            x, skip_connection = layer(x, diffusion_step, gamma=g_i, beta=b_i) # [B,C,L] for each of them
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers)) # [B,C,L]
        x = self.skip_projection(x) # [B,C,L]
        x = F.relu(x)
        x = self.output_projection(x) # [B,1,L]
        return x