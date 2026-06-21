import ot
from typing import Literal
import torch
import numpy as np


"inspired by the TSFLOW implementation https://arxiv.org/abs/2410.03024"

def get_ot_plan(
    source_samples: torch.Tensor,
    target_samples: torch.Tensor,
    reg: float = 1e-1,
    type: Literal["sinkhorn", "emd"] = "emd",
    squared_cost: bool = True,
) -> np.ndarray:
    source_flat = source_samples.reshape(source_samples.shape[0], -1)
    target_flat = target_samples.reshape(target_samples.shape[0], -1)

    cost_matrix = torch.cdist(source_flat, target_flat, p=2)

    if squared_cost:
        cost_matrix = cost_matrix.pow(2)

    n_source = source_flat.shape[0]
    n_target = target_flat.shape[0]

    a = torch.full((n_source,), 1.0 / n_source).cpu().numpy()
    b = torch.full((n_target,), 1.0 / n_target).cpu().numpy()

    cost_matrix_np = cost_matrix.detach().cpu().numpy()

    if type == "sinkhorn":
        ot_plan_np = ot.sinkhorn(a, b, cost_matrix_np, reg)
    elif type == "emd":
        ot_plan_np = ot.emd(a, b, cost_matrix_np)
    else:
        raise ValueError(f"Unknown OT type: {type}")

    return ot_plan_np

def sample_ot_minibatch(
    source_samples: torch.Tensor,
    target_samples: torch.Tensor,
):
    assert source_samples.shape[0] == target_samples.shape[0], "Source and target must have the same number of samples"

    batch_size = source_samples.shape[0]
    pi = get_ot_plan(source_samples, target_samples, type="sinkhorn")

    source_indices = []
    available_sources = np.arange(batch_size)

    for target_idx in range(batch_size):
        p = pi[available_sources, target_idx]

        if p.sum() <= 0:
            p = np.ones_like(p) / len(p)
        else:
            p = p / p.sum()

        chosen_pos = np.random.choice(
            len(available_sources),
            p=p,
        )

        source_idx = available_sources[chosen_pos]
        source_indices.append(source_idx)

        available_sources = np.delete(available_sources, chosen_pos)

    source_idx = torch.as_tensor(
        source_indices,
        device=source_samples.device,
        dtype=torch.long,
    )

    source_batch = source_samples[source_idx]

    return source_batch


# def sample_ot_minibatch(
#     source_samples: torch.Tensor,
#     target_samples: torch.Tensor,
# ):
#     assert source_samples.shape[0] == target_samples.shape[0], "Source and target must have the same number of samples"
#     batch_size = source_samples.shape[0]
#     pi = get_ot_plan(source_samples, target_samples, type="sinkhorn")
#
#     source_indices = []
#
#     for target_idx in range(batch_size):
#         p = pi[:, target_idx]
#
#         if p.sum() <= 0:
#             p = np.ones_like(p) / len(p)
#         else:
#             p = p / p.sum()
#
#         source_idx = np.random.choice(
#             batch_size,
#             p=p,
#         )
#
#         source_indices.append(source_idx)
#
#     source_idx = torch.as_tensor(source_indices, device=source_samples.device)
#
#     source_batch = source_samples[source_idx]
#
#     return source_batch
