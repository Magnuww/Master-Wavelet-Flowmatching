from torch import Tensor
from typing import Optional, List, Literal, Sequence, Tuple
import matplotlib.pyplot as plt
import mlflow
import models.transforms as transforms
import numpy as np
import os
import seaborn as sns
import torch

def plot_distribution(ori_data, gen_data, mean=True):
    sample_num = min([1000, len(ori_data), len(gen_data)])
    idx = np.random.permutation(len(ori_data))[:sample_num]
    idx2 = np.random.permutation(len(gen_data))[:sample_num]

    ori_data = ori_data[idx]
    gen_data = gen_data[idx2]

    if mean:
        prep_data = np.mean(ori_data, axis=1)
        prep_data_hat = np.mean(gen_data, axis=1)
    else:
        prep_data = ori_data.reshape(sample_num, -1)
        prep_data_hat = gen_data.reshape(sample_num, -1)

    fig, ax = plt.subplots(1,1)
    sns.kdeplot(prep_data.flatten(), color='C0', linewidth=2, label='Original', ax=ax)

    # Plotting KDE for generated data on the same axes
    sns.kdeplot(prep_data_hat.flatten(), color='C1', linewidth=2, linestyle='--', label='Generated', ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('')
    for pos in ['top','right']:
        ax.spines[pos].set_visible(False)
    ax.legend()
    plt.close(fig )
    return fig


def plot_multiple_multivariate_samples_togheter_subfigure(samples: dict[str,Tensor], keys : List[str] | None, mode : Literal["fan", "lines", "both"] = "both"):
    vals = [v for k,v in samples.items()]
    b, c, l = vals[0].shape
    if keys != None and len(keys) != c:
        raise ValueError(f"Expected number of keys to match the number of channels in the samples, but got {len(keys)} and {c} respectively.")
    out_dict = {}
    for i in range(c):
        sample_slices = {k: v[:,i:i+1,:] for k,v in samples.items()}
        fig = plot_multiple_samples_togheter_subfigure(sample_slices, mode)
        key = keys[i] if keys != None else f"channel_{i}"
        out_dict[key] = fig
    return out_dict

def plot_multiple_samples_togheter_subfigure(samples: dict[str,Tensor], mode : Literal["fan", "lines", "both"] = "both"):
    n_plots = len(samples)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2 * n_plots))
    for idx, (k, v) in enumerate(samples.items()):
        B, C , L = v.shape
        data = v.detach().cpu().numpy()
        lines = min(B, 10)
        percentiles = [
            (0, 100),
            (1, 99),
            (5, 95),
            (20, 80),
            (40, 60),
        ]
        perc_values = {
            p: np.percentile(data, p, axis=0).squeeze(0)
            for pair in percentiles for p in pair
        }
        alphas = [0.10,0.2, 0.3, 0.35, 0.4]
        x = np.arange(L)
        #plot the fan of percentiles
        if mode in ["fan", "both"]:
            for (low, high), alpha in zip(percentiles, alphas):
                label = f"{low}th-{high}th percentile"
                axes[idx].fill_between(
                    x,
                    perc_values[low],
                    perc_values[high],
                    color='blue',
                    alpha=alpha,
                    linewidth=0,
                    label=label if idx == 0 else None
                )
        if C != 1:
            raise ValueError(f"Expects only one channel per batch, Recieved {C}")
        if mode in ["lines", "both"]:
            for i in range(lines):
                axes[idx].plot(x, data[i,0,:])
        axes[idx].set_title(k)
        if idx == 0:
            axes[idx].legend(loc='upper right')
    plt.tight_layout()
    plt.close(fig)
    return fig

def _normalize_batch(t: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        raise TypeError("input must be a torch.Tensor")
    t_np = t.detach().cpu().squeeze().numpy()

    # Normalize to (batch_size, seq_len)
    if t_np.ndim == 1:
        t_np = t_np[None, :]
    elif t_np.ndim == 3 and t_np.shape[2] == 1:
        t_np = t_np[:, :, 0]
    elif t_np.ndim != 2:
        raise ValueError(f"Unsupported tensor shape {t_np.shape}")
    return t_np

def plot_n_batches_vertical(
    *batches: torch.Tensor,
    titles: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (8, 4),
    sharex: bool = True,
    sharey: bool = False,
    axes=None,  # New parameter
    fig=None,   # New parameter
):
    n = len(batches)
    if n == 0:
        raise ValueError("At least one batch tensor must be provided.")

    normalized_batches = [_normalize_batch(batch) for batch in batches]
    created_figure = axes is None or fig is None

    if created_figure:
        fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1] * n / 3), sharex=sharex, sharey=sharey)

    if n == 1:
        axes = [axes]

    if titles is None:
        titles = [f"Batch {i+1}" for i in range(n)]
    else:
        if len(titles) != n:
            raise ValueError(f"titles must have {n} elements.")

    for ax, data, title in zip(axes, normalized_batches, titles):
        batch_size, seq_len = data.shape
        x = range(seq_len)
        for i in range(batch_size):
            ax.plot(x, data[i], label=f"batch_{i}")
        ax.set_xlabel("Index")
        if batch_size > 1:
            ax.legend()
        ax.set_ylabel("Value")
        ax.set_title(title)

    axes[-1].set_xlabel("Index")
    if created_figure:
        plt.tight_layout()
        plt.show()
    return fig, axes

def format_coeffs_for_plotting(coeffs: Tensor, transformer):
    if isinstance(transformer, transforms.SimpleMaskTransform) or isinstance(transformer, transforms.HeuristicTopKMaskTransform):
        transformer = transformer.base_transformer
    match transformer:
        case transforms.RFFTTransformer():
            return torch.linalg.norm(coeffs, axis=1, keepdims=True)
        case transforms.FlatWaveletTransformer():
            return coeffs
        case transforms.IdentityTransformer():
            return coeffs
        case transforms.SimpleMaskTransform():
            return coeffs
        case transforms.HeuristicTopKMaskTransform():
            return coeffs
        case transforms.NormalizedTransformer():
            return coeffs
        case _:
            raise ValueError("Plotting not implemented for this transformer")
