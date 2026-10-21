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
