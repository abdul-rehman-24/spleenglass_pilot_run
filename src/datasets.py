
import numpy as np
import torch
from torch.utils.data import Dataset

class SliceDataset(Dataset):
    """Preloads selected volumes into RAM as one contiguous (num_slices, 1, 256, 256)
    tensor. Slice order is unchanged (volumes in given order, slices ascending);
    only the memory layout differs from the earlier strided version."""
    def __init__(self, volume_ids, cache_dir="/kaggle/working/cache/preprocessed"):
        imgs, lbls = [], []
        self.samples = []  # (volume_id, slice_idx), same order as before
        for vid in volume_ids:
            data = np.load(f"{cache_dir}/{vid}.npz")
            img = data["image"].astype(np.float32)   # (256,256,S)
            lbl = data["label"].astype(np.float32)
            imgs.append(np.ascontiguousarray(np.transpose(img, (2, 0, 1))))  # (S,256,256)
            lbls.append(np.ascontiguousarray(np.transpose(lbl, (2, 0, 1))))
            for s in range(img.shape[-1]):
                self.samples.append((vid, s))
        self.images = torch.from_numpy(np.concatenate(imgs, axis=0)).unsqueeze(1)  # (N,1,256,256)
        self.labels = torch.from_numpy(np.concatenate(lbls, axis=0)).unsqueeze(1)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class VolumeCache:
    """Loads one full volume for 3D validation-inference/reconstruction.
    Caches in RAM after first load so repeated epochs don't re-hit disk."""
    def __init__(self, cache_dir="/kaggle/working/cache/preprocessed"):
        self.cache_dir = cache_dir
        self._ram_cache = {}

    def load(self, vid):
        if vid not in self._ram_cache:
            data = np.load(f"{self.cache_dir}/{vid}.npz")
            self._ram_cache[vid] = (data["image"].astype(np.float32), data["label"].astype(np.float32))
        return self._ram_cache[vid]
