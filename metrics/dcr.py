import numpy as np
import torch
from tqdm import tqdm

def calculate_dcr(ori_data, gen_data):
    closest_distances = []
    for gen_series in tqdm(gen_data):
        distances = []
        for ori_series in ori_data:
            dist = np.linalg.norm(ori_series - gen_series)
            distances.append(dist)

        closest_distances.append(min(distances))
    return np.mean(closest_distances)

def calculate_dcr_torch(ori_data, gen_data, batch_size=64):
    print(ori_data.shape, gen_data.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ori = torch.as_tensor(ori_data, dtype=torch.float32, device=device)
    gen = torch.as_tensor(gen_data, dtype=torch.float32, device=device)

    ori = ori.reshape(ori.shape[0], -1)
    gen = gen.reshape(gen.shape[0], -1)

    closest_distances = []

    with torch.no_grad():
        for i in range(0, gen.shape[0], batch_size):
            gen_batch = gen[i:i + batch_size]

            dists = torch.cdist(gen_batch, ori, p=2)

            min_dists = dists.min(dim=1).values

            closest_distances.append(min_dists)

    closest_distances = torch.cat(closest_distances)

    return closest_distances.cpu().numpy()
