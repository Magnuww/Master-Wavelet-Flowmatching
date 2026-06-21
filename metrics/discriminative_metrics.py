"""

Code adapted from:

Diffusion-ts: https://arxiv.org/abs/2403.01742
Waveletdiff: https://arxiv.org/abs/2510.11839

The original implementation is based on the Pytorch timegan codebase:

Reimplementation of discrimintive score from the TimeGAN Codebase.

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
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
from itertools import cycle

from metrics.metric_utils import extract_time, train_test_divide


class TimeSeriesDataset(Dataset):
    def __init__(self, data, time):
        self.data = torch.tensor(np.array(data), dtype=torch.float32)
        self.time = torch.tensor(np.array(time), dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.time[idx]


class Discriminator(nn.Module):
    """
    Discriminator model for distinguishing between real and synthetic time-series data.
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, seq_lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x,
            seq_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.rnn(packed_input)
        logits = self.fc(hidden[-1])

        return logits


def discriminative_score_metrics(ori_data, generated_data, iterations=2000):
    """Use post-hoc RNN to classify original data and synthetic data

    Args:
        - ori_data: original data
        - generated_data: generated synthetic data

    Returns:
        - discriminative_score: np.abs(classification accuracy - 0.5)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    no, seq_len, dim = np.asarray(ori_data).shape
    hidden_dim = max(dim // 2, 2)
    batch_size = 128
    # Prepare the data
    ori_time, _ = extract_time(ori_data)
    generated_time, _ = extract_time(generated_data)

    (
        train_x,
        train_x_hat,
        test_x,
        test_x_hat,
        train_t,
        train_t_hat,
        test_t,
        test_t_hat,
    ) = train_test_divide(ori_data, generated_data, ori_time, generated_time)

    real_dataset = TimeSeriesDataset(train_x, train_t)
    fake_dataset = TimeSeriesDataset(train_x_hat, train_t_hat)

    real_loader = DataLoader(
        real_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    fake_loader = DataLoader(
        fake_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    real_iter = cycle(real_loader)
    fake_iter = cycle(fake_loader)

    discriminator = Discriminator(dim, hidden_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(discriminator.parameters())

    discriminator.train()

    for _ in tqdm(range(iterations), desc="training", total=iterations):
        X_mb, T_mb = next(real_iter)
        X_hat_mb, T_hat_mb = next(fake_iter)

        X_mb = X_mb.to(device)
        T_mb = T_mb.to(device)
        X_hat_mb = X_hat_mb.to(device)
        T_hat_mb = T_hat_mb.to(device)

        optimizer.zero_grad()

        logits_real = discriminator(X_mb, T_mb)
        logits_fake = discriminator(X_hat_mb, T_hat_mb)

        loss_real = criterion(logits_real, torch.ones_like(logits_real))
        loss_fake = criterion(logits_fake, torch.zeros_like(logits_fake))
        loss = loss_real + loss_fake

        loss.backward()
        optimizer.step()

    discriminator.eval()

    test_x = torch.tensor(np.array(test_x), dtype=torch.float32).to(device)
    test_t = torch.tensor(np.array(test_t), dtype=torch.long).to(device)
    test_x_hat = torch.tensor(np.array(test_x_hat), dtype=torch.float32).to(device)
    test_t_hat = torch.tensor(np.array(test_t_hat), dtype=torch.long).to(device)

    with torch.no_grad():
        y_pred_real = torch.sigmoid(discriminator(test_x, test_t)).cpu().numpy()
        y_pred_fake = torch.sigmoid(discriminator(test_x_hat, test_t_hat)).cpu().numpy()

    y_pred_final = np.concatenate((y_pred_real, y_pred_fake), axis=0).squeeze()

    y_label_final = np.concatenate(
        [
            np.ones(len(y_pred_real)),
            np.zeros(len(y_pred_fake)),
        ]
    )

    acc = accuracy_score(y_label_final, y_pred_final > 0.5)
    fake_acc = accuracy_score(np.zeros(len(y_pred_fake)), y_pred_fake > 0.5)
    real_acc = accuracy_score(np.ones(len(y_pred_real)), y_pred_real > 0.5)

    discriminative_score = np.abs(0.5 - acc)
    return discriminative_score, fake_acc, real_acc
