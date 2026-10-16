import torch
import torch.nn as nn

class FiLMGenerator(nn.Module):
    def __init__(self, num_params, num_layers, channels_per_layer, hidden_dim=128):
        super().__init__()
        self.num_layers = num_layers
        self.channels = channels_per_layer

        # Centralized MLP to generate scale (gamma) and shift (beta) for all layers
        self.net = nn.Sequential(
            nn.Linear(num_params, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * num_layers * channels_per_layer)
        )

        # Zero-initialize the final output so FiLM starts as an identity function
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, conditioner):
        # conditioner: (B, num_params)
        B = conditioner.shape[0]
        params = self.net(conditioner)                              # [B,2*N*C]
        params = params.view(B, self.num_layers, 2, self.channels)  # [B,N,2,C]

        gamma = params[:, :, 0, :] # [B,N,C]
        beta  = params[:, :, 1, :] # [B,N,C]
        return gamma, beta



