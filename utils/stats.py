from typing import Protocol
from scipy.sparse import rand
import torch
from torch.utils.data import DataLoader 

from torch.nn.modules import transformer
from typing import Sequence, Literal, Tuple
from models.transforms import Transform

def compute_channel_stats(dataloader : DataLoader,transformer : Transform, type : Literal["z", "minmax"]):
    channel_stats = []
    all_coeffs = []
    for _, y in dataloader:
        coeffs = transformer.transform(y)
        all_coeffs.append(coeffs)
    coeffs = torch.cat(all_coeffs, dim=0)

    print(f"Computing channel stats for {coeffs.shape[1]} channels and {coeffs.shape[0]} samples")
    print(f" with min {coeffs.min().item()} and max {coeffs.max().item()}")
    if type == "z":
        for c in range(coeffs.shape[1]):
            channel_data = coeffs[:, c, :].flatten()
            mean = channel_data.mean().item()
            std = channel_data.std().item()
            channel_stats.append((mean, std))
    elif type == "minmax":
        for c in range(coeffs.shape[1]):
            channel_data = coeffs[:, c, :].flatten()
            min_val = channel_data.min().item()
            max_val = channel_data.max().item()
            channel_stats.append((min_val, max_val))
    print(f"Computed channel stats: {channel_stats}")
    return channel_stats

def compute_channel_stats_per_split(dataloader : DataLoader, transformer : Transform,  split : list[int], type : Literal["z", "minmax"] = "z",):
    all_coeffs = []
    out = []
    for _, y in dataloader:
        coeffs = transformer.transform(y)
        all_coeffs.append(coeffs)
    coeffs = torch.cat(all_coeffs, dim=0)
    print(split)
    split_coeffs = torch.split(coeffs, split, dim=2)
    for split_coeff in split_coeffs:
        channel_stats = []
        for c in range(coeffs.shape[1]):
            channel_data = split_coeff[:, c, :].flatten()
            if type == "z":
                mean = channel_data.mean().item()
                std = channel_data.std().item()
                channel_stats.append((mean, std))
            elif type == "minmax":
                min_val = channel_data.min().item()
                max_val = channel_data.max().item()
                channel_stats.append((min_val, max_val))
        out.append(channel_stats.copy())
        print(f"Computed channel stats per split: {channel_stats}")
    return out
