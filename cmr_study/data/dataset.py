import numpy as np
import torch
from functools import lru_cache
from torch.utils.data import Dataset
from ..motion import providers


@lru_cache(maxsize=16)
def mapped_cine(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


def motion_kind(method):
    if method.startswith("cineq") or method == "mae_dis":
        return "dis"
    return {"heartmae": "heart", "trackmae": "cotracker", "mae_registration": "registration"}.get(method)


def flip_motion(data, size, grid):
    result = {}
    # New patch j corresponds to old horizontally reflected patch j.
    order = np.arange(grid*grid).reshape(grid, grid)[:, ::-1].copy().reshape(-1)
    for k, value in data.items():
        if k == "weight":
            value = value[:, ::-1].copy()
        elif k == "positions":
            axis = 2 if value.ndim == 5 else 1
            value = np.take(value, order, axis=axis).copy()
            if value.ndim == 5:
                value = value[..., [0, 2, 1, 3, 4], :]
            value[..., 0] = size-1-value[..., 0]
        elif k == "confidence":
            axis = 2 if value.ndim == 4 else 1
            value = np.take(value, order, axis=axis).copy()
            if value.ndim == 4:
                value = value[..., [0, 2, 1, 3, 4]]
        result[k] = value
    return result


class CineDataset(Dataset):
    def __init__(self, rows, settings, method="mae_uniform", training=False, task=None, classes=None):
        self.rows, self.settings, self.method = rows, settings, method
        self.training, self.task, self.classes = training, task, classes
        self.items = [(i, v) for i, r in enumerate(rows) for v in range(r["views"])]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        i, view = self.items[index]
        row, s = self.rows[i], self.settings
        video = np.array(mapped_cine(row["cache_path"])[view], dtype=np.float32)
        kind = None if self.task else motion_kind(self.method)
        motion = providers.load(row, view, kind, s) if kind else {}
        if self.training:
            if np.random.random() < s["data"]["horizontal_flip"]:
                video = video[:, :, ::-1].copy()
                motion = flip_motion(motion, video.shape[-1], video.shape[-1]//s["model"]["patch_size"])
            jitter = s["data"]["intensity_jitter"]
            video = np.clip(video*np.random.uniform(1-jitter, 1+jitter)+np.random.uniform(-jitter/2, jitter/2), 0, 1)
        item = dict(video=torch.from_numpy(video)[None], patient_uid=row["patient_uid"], view=view)
        for key in ("positions", "confidence", "weight"):
            if key in motion:
                item[key] = torch.from_numpy(motion[key].astype(np.float32))
        if self.task == "classification":
            item["target"] = torch.tensor(self.classes.index(row["disease"]), dtype=torch.long)
        elif self.task == "ef_regression":
            item["target"] = torch.tensor(row["lvef"], dtype=torch.float32)
        return item
