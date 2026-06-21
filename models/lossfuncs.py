from typing import Optional

from flow_matching.path import PathSample

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Literal

from models.transforms import Transform

def get_loss_func(loss_name: Literal["MSE", "Channel_MSE", "Recon_MSE", "Combined"], transform: Optional[Transform] = None, level_indexes: Optional[List[int]] = None, alpha: Optional[float] = None) -> nn.Module:
    loss_name_lower = loss_name.lower()
    if loss_name_lower == "mse":
        return MSE()
    elif loss_name_lower == "recon_mse":
        if transform is None:
            raise ValueError("Transform must be provided for Recon_MSE")
        return Recon_MSE(transform)
    elif loss_name_lower == "new_recon_loss":
        if transform is None:
            raise ValueError("Transform must be provided for New_Recon_loss")
        return New_Recon_loss(transform)
    elif loss_name == "Channel_MSE":
        return Channel_MSE()
    elif loss_name_lower == "new_combined_loss":
        if transform is None:
            raise ValueError("Transform must be provided for New_Combined_loss")
        if alpha is None:
            raise ValueError("Alpha must be provided for New_Combined_loss")
        recon_loss = New_Recon_loss(transform)
        mse = MSE()
        return New_Combined_loss(transform, mse, alpha)
    elif loss_name_lower.startswith("combined"):
        if transform is None:
            raise ValueError("Transform must be provided for Combined_loss")
        if alpha is None:
            raise ValueError("Alpha must be provided for Combined_loss")
        recon_loss = Recon_MSE(transform)
        mse = MSE()
        return Combined_loss(recon_loss, mse, alpha)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

class WeightedWaveletLoss(nn.Module):
    def __init__(self, level_indexes: List[int] | int) -> None:
        super().__init__()
        self.level_indexes = level_indexes
        self.toString = f"WeightedWaveletLoss(splits={level_indexes})"

    def weighted_loss(self, input: torch.Tensor, target: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
        eps_t = target.new_tensor(eps)
        denom = target.abs().mean(dim=2, keepdim=True).clamp_min(eps_t)
        rel_se = (input - target).pow(2) / denom.pow(2)
        return rel_se.mean()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        inputs = torch.split(input, self.level_indexes, dim=2)
        targets = torch.split(target, self.level_indexes, dim=2)
        zero = input.new_tensor(0.0)
        loss = sum((self.weighted_loss(i, t) for i, t in zip(inputs, targets)), start=zero)
        loss = loss/float(len(inputs))
        return loss


class WeightedWaveletSparseLoss(nn.Module):
    def __init__(self, level_indexes: List[int], sparse_weight: float = 1e-3) -> None:
        super().__init__()
        self.wavelet = WeightedWaveletLoss(level_indexes)
        self.sparse_weight = float(sparse_weight)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        wloss = self.wavelet(input, target)
        # L1 acts as a regularizer on the wavelet loss (multiplicative)
        #
        sparse_penalty = torch.abs(input).sum()
        return wloss * (1.0 + self.sparse_weight * sparse_penalty)

class Recon_MSE(nn.Module):
    def __init__(self, transform : Transform) -> None:
        super().__init__()
        self.toString = "Recon_MSE"
        self.transform = transform

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = self.transform.inverse_transform(input) 
        target = self.transform.inverse_transform(target) 
        return F.mse_loss(input, target)

class Channel_MSE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # input and target are (B, C, L)
        total_loss = torch.tensor(0.0, device=input.device)
        for i in range(input.shape[1]):
            tmp = F.mse_loss(input[:, i, :], target[:, i, :]) 
            print(f'dim {i} loss : {tmp.item()}')
            total_loss += tmp
        return total_loss / input.shape[1]

class Scale_normalized_MSE(nn.Module):
    def __init__(self, eps: float = 1e-2) -> None:
        super().__init__()
        self.toString = "Scale_normalized_MSE"
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps_t = target.new_tensor(self.eps)
        denom = target.abs().mean(dim=2, keepdim=True).clamp_min(eps_t)
        rel_se = (input - target).pow(2) / denom.pow(2)
        return rel_se.mean()

class MSE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.toString = "MSE"
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((input - target) ** 2).mean()

class L1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.toString = "L1"
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.abs(input - target).mean()

class Cosine_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.toString = "Cosine_loss"
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # compute the cosine similarity between input and target, then return 1 - cosine_similarity
        cosine_similarity = F.cosine_similarity(input, target, dim=2)
        return 1 - cosine_similarity.mean()

class General_Recon_loss(nn.Module):
    def __init__(self, transform : Transform, loss_fn : nn.Module) -> None:
        super().__init__()
        self.toString = f"recon({loss_fn.toString})"
        self.transform = transform
        self.loss_fn = loss_fn

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = self.transform.inverse_transform(input) 
        target = self.transform.inverse_transform(target) 
        return self.loss_fn(input, target)

class New_Recon_loss(nn.Module):
    def __init__(self, transform : Transform) -> None:
        super().__init__()
        self.toString = "New_Recon_loss"
        self.transform = transform

    def forward(self, v_pred: torch.Tensor, x_target: torch.Tensor, path_sample : PathSample) -> torch.Tensor:
        """
        Recon loss that compares the inverse transformed predicted x_hat 
        with the inverse transformed target x_target, 
        THIS ONLY WORKS IF THE PATH IS LINEAR 

        v_pred is the predicted velocity,
        x_t is the current position,
        t is the time step.
        x_target is the target position at time t.
        """
        x_t = path_sample.x_t
        t = path_sample.t
        if t.ndim == 1:
            t = t.unsqueeze(-1).unsqueeze(-1)
        x_hat = x_t + (1-t) * v_pred
        x_hat_inv = self.transform.inverse_transform(x_hat)
        return F.mse_loss(x_hat_inv, x_target)

class New_Combined_loss(nn.Module):
    def __init__(self, transform : Transform, loss_fn : nn.Module, alpha = 0.5) -> None:
        super().__init__()
        self.toString = f"New_Combined(new_recon{loss_fn.toString}_{alpha})"
        self.recon_loss = New_Recon_loss(transform)
        self.loss_fn = loss_fn
        self.alpha = alpha

    def forward(self, v_pred: torch.Tensor, x_target: torch.Tensor, path_sample : PathSample) -> torch.Tensor:
        dx_t = path_sample.dx_t
        recon = self.recon_loss(v_pred, x_target, path_sample)
        loss = self.loss_fn(v_pred, dx_t)
        return self.alpha * recon + loss

class Combined_loss(nn.Module):
    def __init__(self, l1: nn.Module, l2: nn.Module, alpha: float = 0.5) -> None:
        super().__init__()
        self.toString = f"Combined({l1.toString}_{l2.toString}_{alpha})"
        self.l1 = l1
        self.l2 = l2
        self.alpha = alpha

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        recon = self.l1(input, target)
        channel = self.l2(input, target)
        return self.alpha * recon + (1 - self.alpha) * channel
