import math
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import metrics


class CovBlock(nn.Module):

    # !padding和初始化与原实现不同
    def __init__(self, in_channels, out_channels, conv_patch: int = 5, pool_patch: int = 2):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=conv_patch,
                                stride=1,
                                padding=conv_patch - 1,
                                padding_mode='circular',
                                )

        self.activation = nn.PReLU()

        self.max_pool = nn.MaxPool3d(kernel_size=pool_patch,
                                     ceil_mode=True)

    def forward(self, x):
        h = self.conv3d(x)
        h = self.activation(h)
        output = self.max_pool(h)
        return output


class Pafnucy(nn.Module):

    def __init__(self, in_chnls=19,
                 conv_patch=5, pool_patch=2,
                 conv_channels: List[int] = [64, 128, 256],
                 dense_sizes: List[int] = [1000, 500, 200],
                 seed=123, keep_prob=1):
        super().__init__()

        np.random.seed(seed)
        torch.random.manual_seed(seed)

        # (batch_size, channel_size, D, H, W) = (batch_size, 19, 21, 21, 21)
        assert len(conv_channels) == 3
        self.conv_3d = nn.Sequential(
            CovBlock(in_chnls, conv_channels[0], conv_patch, pool_patch),
            CovBlock(conv_channels[0], conv_channels[1], conv_patch, pool_patch),
            CovBlock(conv_channels[1], conv_channels[2], conv_patch, pool_patch),
        )
        # output_size=(batch_size, conv_channels[-1], ceil(D/pool_patch), ceil(H/pool_patch), ceil(W/pool_patch))

        assert len(dense_sizes) == 3
        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 3 * 256, dense_sizes[0]),
            nn.PReLU(),
            nn.Dropout(1-keep_prob),
            nn.Linear(dense_sizes[0], dense_sizes[1]),
            nn.PReLU(),
            nn.Dropout(1-keep_prob),
            nn.Linear(dense_sizes[1], dense_sizes[2]),
            nn.PReLU(),
            nn.Dropout(1-keep_prob),
        )

        self.classifier = nn.Sequential(
            nn.Linear(dense_sizes[2], 1),
            # nn.ReLU(),  # remove relu to get better performance in training
        )

    def forward(self, x: torch.Tensor):
        assert x.shape[1:] == (19, 21, 21, 21)
        batch_size = x.shape[0]

        a = self.conv_3d(x)
        b = self.fc(a.view(batch_size, -1))
        y_hat = self.classifier(b)

        return y_hat


def val(model: nn.Module, val_loader: DataLoader, loss_function, device, len_rotation=24):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for idx, (x, y) in enumerate(val_loader):

            x = torch.Tensor(np.vstack(x)).to(device)
            y = torch.Tensor(y).to(device)

            output = model(x)

            val_loss += loss_function(output.view(-1), y.view(-1)).item()

    val_loss /= len(val_loader.dataset)*len_rotation

    return val_loss


def test(model: nn.Module, test_loader: DataLoader, loss_function, device, len_rotation=24) -> Dict:
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for idx, (x, y) in enumerate(test_loader):

            x = torch.Tensor(np.vstack(x)).to(device)
            y = torch.Tensor(y).to(device)

            output = model(x)

            test_loss += loss_function(output.view(-1), y.view(-1)).item()
            outputs.append(output.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))

    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)

    test_loss /= len(test_loader.dataset)*len_rotation

    evaluation = {
        'test_loss': test_loss,
        'c_index': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'CORR': metrics.CORR(targets, outputs),
    }

    return evaluation