import os
import torch
import torch.nn as nn
import numpy as np


class CheckpointManager:
    """
    Unified manager for saving, restoring, and maintaining rolling checkpoints
    """

    def __init__(
            self,
            checkpoint_dir: str,
            model: nn.Module,
            optimizer: torch.optim.Optimizer = None,
            scaler: torch.amp.GradScaler = None,
            config: dict = None,
            keep_last_n: int = 5,
            is_master: bool = True
    ):
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.config = config
        self.keep_last_n = keep_last_n
        self.is_master = is_master
        self.saved_checkpoints = []

        if self.is_master:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _unwrap_model(self):
        """Unwraps DistributedDataParallel or DataParallel wrappers."""
        if hasattr(self.model, "module") and isinstance(self.model.module, nn.Module):
            return self.model.module
        return self.model

    def get_state_dict(self, step_or_epoch: int) -> dict:
        """Packages weights, optimizer, scaler, config, and all RNG states."""
        unwrapped_model = self._unwrap_model()
        model_state = unwrapped_model.state_dict()

        state = {
            "step": step_or_epoch,
            "model": {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in model_state.items()
            },
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
        }

        if self.optimizer is not None:
            state["optimizer"] = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in self.optimizer.state_dict().items()
            }

        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()

        if self.config is not None:
            state["config"] = self.config

        return state

    def load_state_dict(self, state_dict: dict) -> int:
        """Restores model, optimizer, scaler, and random generators."""
        unwrapped_model = self._unwrap_model()
        unwrapped_model.load_state_dict(state_dict["model"])

        if self.optimizer is not None and "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])

        if self.scaler is not None and "scaler" in state_dict:
            self.scaler.load_state_dict(state_dict["scaler"])

        if "torch_rng_state" in state_dict:
            torch.set_rng_state(state_dict["torch_rng_state"])
            torch.cuda.set_rng_state(state_dict["cuda_rng_state"])
            np.random.set_state(state_dict["numpy_rng_state"])

        return state_dict.get("step", 0)

    def save(self, step_or_epoch: int, filename: str = "weights") -> str:
        """Saves a checkpoint and purges old checkpoints exceeding keep_last_n."""
        if not self.is_master:
            return ""

        save_basename = f"{filename}-{step_or_epoch}.pt"
        save_path = os.path.join(self.checkpoint_dir, save_basename)
        link_path = os.path.join(self.checkpoint_dir, f"{filename}.pt")

        torch.save(self.get_state_dict(step_or_epoch), save_path)

        # Update alias / symlink pointing to latest
        if os.name == "nt":
            torch.save(self.get_state_dict(step_or_epoch), link_path)
        else:
            if os.path.islink(link_path) or os.path.exists(link_path):
                os.unlink(link_path)
            os.symlink(save_basename, link_path)

        # Rolling retention
        self.saved_checkpoints.append(save_path)
        if len(self.saved_checkpoints) > self.keep_last_n:
            oldest_ckpt = self.saved_checkpoints.pop(0)
            if os.path.exists(oldest_ckpt):
                os.remove(oldest_ckpt)

        return save_path

    def restore(self, filename: str = "weights") -> tuple[bool, int]:
        """Restores the latest weights alias (e.g. weights.pt). Returns (success, step)."""
        ckpt_path = os.path.join(self.checkpoint_dir, f"{filename}.pt")
        if not os.path.exists(ckpt_path):
            return False, 0

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        step = self.load_state_dict(checkpoint)
        return True, step

    def load_pretrained(self, filepath: str) -> bool:
        """Loads weights from an explicit checkpoint path."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Weight path not found: {filepath}")

        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        model_state = checkpoint.get("model", checkpoint)

        # Clean any module prefix if needed
        model_state = {k.replace("module.", ""): v for k, v in model_state.items()}

        unwrapped_model = self._unwrap_model()
        unwrapped_model.load_state_dict(model_state, strict=False)
        return True