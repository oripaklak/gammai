import torch
import torch.nn as nn

class LimitedDataLoss(nn.Module):
    def __init__(self, eta: float = 100.0):
        super().__init__()
        self.eta = eta

    def forward(self, pred_clean, target_clean, ood_sample = None, use_nda = False):
        B = pred_clean.shape[0]
        ood_loss = 0
        diff_real = (pred_clean - target_clean).reshape(B, -1)

        # 1. Standard Denoising L2 loss: || D_theta(x) - y ||_2^2
        recon_loss = torch.sum(diff_real ** 2, dim =-1).mean()

        # 2. OOD Regularization: 1 / (||D_theta(x) - y_hat||_2^2 + eta)
        if use_nda and ood_sample is not None:
            diff_ood = (pred_clean - ood_sample).reshape(B, -1)
            dist_to_ood_sq = torch.sum(diff_ood ** 2, dim=-1)
            ood_loss = (1.0 / (dist_to_ood_sq + self.eta)).mean()

        total_loss = recon_loss + ood_loss
        return total_loss, recon_loss, ood_loss

class Loss(nn.Module):
    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        loss_type = loss_type.lower()
        if loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type in ("l2", "mse"):
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss_type '{loss_type}'. Choose 'l1' or 'l2'/'mse'.")

    def forward(self, pred_noise, target_noise):
        # Align shapes if one has a singleton channel dimension (e.g., [B, 1, L] vs [B, L])
        if pred_noise.shape != target_noise.shape:
            pred_noise = pred_noise.view_as(target_noise)

        return self.loss_fn(pred_noise, target_noise)
        