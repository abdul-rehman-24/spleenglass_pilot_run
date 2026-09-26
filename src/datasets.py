
import numpy as np
import torch
from torch.utils.data import Dataset

class SliceDataset(Dataset):
    """Flattens selected volumes into individual 2D slices for training."""
    def __init__(self, volume_ids, cache_dir="/kaggle/working/cache/preprocessed"):
        self.samples = []
        self.cache_dir = cache_dir
        for vid in volume_ids:
            data = np.load(f"{cache_dir}/{vid}.npz")
            n_slices = data["image"].shape[-1]
            for s in range(n_slices):
                self.samples.append((vid, s))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid, s = self.samples[idx]
        data = np.load(f"{self.cache_dir}/{vid}.npz")
        img = data["image"][:, :, s].astype(np.float32)
        lbl = data["label"][:, :, s].astype(np.float32)
        return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(lbl).unsqueeze(0)


class VolumeCache:
    """Loads one full volume for 3D validation-inference/reconstruction."""
    def __init__(self, cache_dir="/kaggle/working/cache/preprocessed"):
        self.cache_dir = cache_dir
    def load(self, vid):
        data = np.load(f"{self.cache_dir}/{vid}.npz")
        return data["image"].astype(np.float32), data["label"].astype(np.float32)
