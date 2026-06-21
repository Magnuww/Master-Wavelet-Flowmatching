"""
Code adapted from:

Diffusion-ts: https://arxiv.org/abs/2403.01742
Waveletdiff: https://arxiv.org/abs/2510.11839

The original implementation is based on the Pytorch timegan codebase:

Reimplementation of predictive score from the TimeGAN Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com)

"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error
from tqdm.auto import tqdm
from itertools import cycle


class PredictiveDataset(Dataset):
    def __init__(self, data, dim, window_size=20):
        self.data = data
        self.dim = dim
        self.window_size = window_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.dim > 1:
            x = sample[:-1, : self.dim - 1]
            y = sample[1:, self.dim - 1 :]
        else:
            x = sample[:-self.window_size]
            y = sample[self.window_size:]

            x = np.expand_dims(x, axis=-1)
            y = np.expand_dims(y, axis=-1)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y


class Predictor(nn.Module):
    """
    A simple RNN-based predictor model using GRU and a fully connected layer.
    """

    def __init__(self, input_dim, hidden_dim):
        super(Predictor, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        outputs, _ = self.rnn(x)
        y_hat_logit = self.fc(outputs)
        y_hat = torch.sigmoid(y_hat_logit)
        return y_hat


def predictive_score_metrics(ori_data, generated_data, window_size=20, iterations=5000):
    """Report the performance of Post-hoc RNN one-step ahead prediction.

    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
      - window_size: number of steps ahead to predict in the univariate case

    Returns:
      - predictive_score: MAE of the predictions on the original data
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)

    no, seq_len, dim = ori_data.shape

    # Network parameters
    hidden_dim = max(dim // 2, 2)
    batch_size = 128
    input_dim = (dim - 1) if dim > 1 else 1

    train_dataset = PredictiveDataset(
        generated_data,
        dim=dim,
        window_size=window_size,
    )

    test_dataset = PredictiveDataset(
        ori_data,
        dim=dim,
        window_size=window_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    train_iter = cycle(train_loader)

    model = Predictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters())

    # Training loop
    model.train()

    for _ in tqdm(range(iterations), desc="training", total=iterations):
        X_mb, Y_mb = next(train_iter)

        X_mb = X_mb.to(device)
        Y_mb = Y_mb.to(device)

        optimizer.zero_grad()

        y_pred = model(X_mb)
        loss = criterion(y_pred, Y_mb)

        loss.backward()
        optimizer.step()

    # Evaluation loop
    model.eval()

    mae_total = 0.0
    n_samples = 0

    with torch.no_grad():
        for X_mb, Y_mb in test_loader:
            X_mb = X_mb.to(device)
            Y_mb = Y_mb.to(device)

            pred_Y = model(X_mb)

            pred_Y = pred_Y.cpu().numpy()
            Y_mb = Y_mb.cpu().numpy()

            for i in range(len(pred_Y)):
                mae_total += mean_absolute_error(Y_mb[i], pred_Y[i])
                n_samples += 1

    predictive_score = mae_total / n_samples

    return predictive_score
