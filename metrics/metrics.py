import pickle
from typing import List, Tuple

import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import jensenshannon


def ks_test(real: List[float], gen: List[float]) -> float:
    res = stats.ks_2samp(real, gen)
    return res.statistic


def merge_same_elements(lst):
    merged = []
    for i in range(len(lst)):
        if i == 0 or lst[i] != lst[i - 1]:
            merged.append(lst[i])
    return torch.tensor(merged)


def travel_distance(idx: torch.Tensor, grid_num: dict) -> List[float]:
    distances = []
    for i in range(idx.shape[0]):
        idx_i = idx[i]
        merged = merge_same_elements(idx_i)
        merged = torch.stack(
            [
                merged // grid_num["y"],
                merged % grid_num["y"],
            ],
            dim=-1,
        )
        distances.extend(np.linalg.norm(merged[1:] - merged[:-1], axis=1).tolist())
    return distances


def gyration_radius(idx: torch.Tensor, grid_num: dict) -> List[float]:
    radiuses = []
    for i in range(idx.shape[0]):
        xy_i = torch.stack(
            [
                idx[i] // grid_num["y"],
                idx[i] % grid_num["y"],
            ],
            dim=-1,
        ).to(torch.float32)
        center = torch.mean(xy_i, dim=0)
        n = xy_i.shape[0]
        radiuses.append(np.sqrt(np.sum(np.linalg.norm(xy_i - center, axis=1) ** 2) / n))
    return radiuses


def duration(idx: torch.Tensor, grid_num: dict) -> List[int]:
    durations = []
    for i in range(idx.shape[0]):
        idx_i = idx[i]
        diff = idx_i[1:] - idx_i[:-1]
        where = torch.where(diff != 0)[0]
        if len(where) == 0:
            durations.append(len(idx_i))
            continue
        durations.append(where[0].item())
        durations.extend((where[1:] - where[:-1]).tolist())
    return durations


def daily_loc(idx: torch.Tensor, grid_num: dict) -> List[int]:
    daily_locs = []
    for i in range(idx.shape[0]):
        idx_i = idx[i]
        daily_locs.append(len(torch.unique(idx_i)))
    return daily_locs


def complete_transition_matrix(idx: torch.Tensor, grid_num: dict) -> torch.Tensor:
    matrix = torch.zeros(grid_num["x"] * grid_num["y"], grid_num["x"] * grid_num["y"])
    interval = (12 * 60) // 30
    for i in range(idx.shape[0]):
        idx_i = idx[i]
        idx_i = np.concatenate([idx_i[::interval], [idx_i[-1]]], axis=0)
        for j in range(idx_i.shape[0] - 1):
            if idx_i[j] == idx_i[j + 1]:
                continue
            matrix[idx_i[j], idx_i[j + 1]] += 1
    return matrix
