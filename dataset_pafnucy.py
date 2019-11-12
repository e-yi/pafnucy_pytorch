from typing import List

import torch

import numpy as np
from torch.utils.data import Dataset

from data import rotate, make_grid


class DatasetPafnucy(Dataset):

    def __init__(self, coords, features, affinity, grid_spacing, max_dist,
                 columns, std_charge, fp16, phase_name, rotations):
        self._coords = coords
        self._features = features
        self._affinity = affinity
        self._grid_spacing = grid_spacing
        self._max_dist = max_dist
        self._columns = columns
        self._std_charge = std_charge
        self._fp16 = fp16
        self._phase_name = phase_name
        self._rotations = rotations

        self.len = len(self._affinity[self._phase_name])

    def __getitem__(self, index):
        x = []
        y = []
        if isinstance(index, slice):
            for ci, fi in zip(self._coords[self._phase_name][index], self._features[self._phase_name][index]):
                for rotation in self._rotations:
                    coords_idx = rotate(ci, rotation)
                    features_idx = fi
                    x.append(make_grid(coords_idx, features_idx,
                                       grid_resolution=self._grid_spacing,
                                       max_dist=self._max_dist))
                    y.append(self._affinity[self._phase_name][index])
        else:
            ci = self._coords[self._phase_name][index]
            fi = self._features[self._phase_name][index]
            for rotation in self._rotations:
                coords_idx = rotate(ci, rotation)
                features_idx = fi
                x.append(make_grid(coords_idx, features_idx,
                                   grid_resolution=self._grid_spacing,
                                   max_dist=self._max_dist))
                y.append(self._affinity[self._phase_name][index])

        x = np.vstack(x)
        x[..., self._columns['partialcharge']] /= self._std_charge
        x = x.transpose(0, 4, 1, 2, 3)
        # x = torch.Tensor(x)

        y = np.concatenate(y)
        # y = torch.Tensor(y)

        if self._fp16:
            x = x.astype(np.float16)
            y = y.astype(np.float16)
        else:
            x = x.astype(np.float32)
            y = y.astype(np.float32)

        return x, y

    def __len__(self) -> int:
        return self.len


class DatasetFactory:

    def __init__(self, coords, features, affinity, grid_spacing, max_dist, columns, fp16):
        self._coords = coords
        self._features = features
        self._affinity = affinity
        self._grid_spacing = grid_spacing
        self._max_dist = max_dist
        self._columns = columns
        self._fp16 = fp16

        # normalize charges
        charges = []
        for feature_data in features['training']:
            charges.append(feature_data[..., columns['partialcharge']])

        charges = np.concatenate([c.flatten() for c in charges])

        self._mean_charge = charges.mean()
        self._std_charge = charges.std()
        print('charges: mean=%s, sd=%s' % (self._mean_charge, self._std_charge))
        print('use sd as scaling factor')

    def get_dataset(self, phase_name: str, rotations: List[int]) -> Dataset:
        return DatasetPafnucy(self._coords, self._features, self._affinity,
                              self._grid_spacing, self._max_dist, self._columns,
                              self._std_charge, self._fp16, phase_name, rotations)
