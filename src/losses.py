import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DistributionBalancedLoss(nn.Module):
    def __init__(
        self,
        class_freq: torch.Tensor,
        beta: float,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        map_param: float = 0.1,
        neg_scale: float = 2.0,
    ):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.neg_scale = neg_scale
        self.map_param = map_param

        freq = class_freq.float()
        total = freq.sum()

        effective_num = 1.0 - torch.pow(beta, freq)
        cb_weights = (1.0 - beta) / (effective_num + 1e-8)
        cb_weights = cb_weights / cb_weights.sum() * len(cb_weights)
        self.register_buffer("cb_weights", cb_weights)

        inv_freq = total / (freq + 1e-8)
        rebalance_weight = inv_freq ** map_param
        rebalance_weight = rebalance_weight / rebalance_weight.mean()
        self.register_buffer("rebalance_weight", rebalance_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        neg_logits = logits * self.neg_scale
        asym_logits = targets * logits + (1 - targets) * neg_logits

        probs = torch.sigmoid(asym_logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1.0 - pt) ** self.focal_gamma
        alpha_t = targets * self.focal_alpha + (1 - targets) * (1 - self.focal_alpha)

        bce = F.binary_cross_entropy_with_logits(asym_logits, targets, reduction="none")

        weight = focal_weight * alpha_t * self.cb_weights * self.rebalance_weight
        loss = (weight * bce).mean()
        return loss


def create_loss(
    loss_name: str,
    class_freq: Optional[torch.Tensor] = None,
    loss_config: Optional[dict] = None,
) -> nn.Module:
    loss_config = loss_config

    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()

    if loss_name == "db_loss":
        if class_freq is None:
            raise ValueError("db_loss requires class_freq")
        return DistributionBalancedLoss(
            class_freq=class_freq,
            beta=loss_config.get("beta"),
            focal_gamma=loss_config.get("focal_gamma"),
            focal_alpha=loss_config.get("focal_alpha"),
            map_param=loss_config.get("map_param"),
            neg_scale=loss_config.get("neg_scale"),
        )

    raise ValueError(f"Unknown loss: {loss_name}")
