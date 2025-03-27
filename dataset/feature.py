import pickle
from typing import Callable, Dict, Optional

import numpy as np
import pyproj
import torch
from torch.utils.data import Dataset

EMB_MEAN = np.array(
    [
        0.27947354,
        2.1750474,
        5.5015726,
        0.7596884,
        13.291945,
        -0.19277863,
        -0.5305433,
        7.340386,
    ]
)
EMB_STD = np.array(
    [
        1.1150157,
        1.0691862,
        1.9051138,
        0.8898528,
        4.1091094,
        0.73213243,
        0.72798234,
        2.5926378,
    ]
)
EMB_MAX, EMB_MIN = 22.208868, -2.8313537


class TrajFeatureDataset(Dataset):
    def __init__(
        self,
        root: str,
        norm: bool = True,
        length: Optional[int] = None,
    ):
        super().__init__()
        self.data = np.load(root, allow_pickle=True)
        self.length = length
        self.norm = norm

    def __len__(self) -> int:
        if self.length is not None:
            assert self.length <= len(self.data)
            return self.length
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        data = self.data[index]
        if self.norm:
            # data = (data - EMB_MEAN) / EMB_STD
            # normalize to [-1, 1]
            data = (data - EMB_MIN) / (EMB_MAX - EMB_MIN) * 2 - 1
        return torch.tensor(data, dtype=torch.float32)
