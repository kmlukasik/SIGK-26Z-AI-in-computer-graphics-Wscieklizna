"""Data for the denoising task (2.2).

Images are 256x256 PNGs (made earlier), scaled to the range <0.0, 1.0>.
We add gaussian noise with a given sigma (0.01 or 0.03).
"""
import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def add_gaussian_noise(img, sigma, rng):
    """img: float32 array (H, W, 3) in <0, 1>. Returns noisy image clipped to <0, 1>."""
    noise = rng.normal(0.0, sigma, img.shape)   # mean 0, standard deviation = sigma
    noisy = img + noise
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def to_tensor(img):
    """(H, W, 3) numpy -> (3, H, W) torch tensor (the format PyTorch expects)."""
    return torch.from_numpy(img.transpose(2, 0, 1).copy())


class DenoiseDataset(Dataset):
    """Returns pairs (noisy image, clean image).

    folder      - e.g. '/content/drive/MyDrive/sigk/data256/train'
    sigma       - noise level, 0.01 or 0.03
    fixed_noise - False for train (new random noise every time),
                  True for val/test (same noise every run -> results can be repeated)
    """

    def __init__(self, folder, sigma, fixed_noise=False, seed=1234):
        self.paths = sorted(glob.glob(f'{folder}/*.png'))
        if len(self.paths) == 0:
            raise FileNotFoundError(f'No png files in {folder}')
        self.sigma = sigma
        self.fixed_noise = fixed_noise
        self.seed = seed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.paths[idx]), cv2.COLOR_BGR2RGB)  # BGR -> RGB
        clean = img.astype(np.float32) / 255.0                              # 0..255 -> 0..1

        if self.fixed_noise:
            rng = np.random.default_rng(self.seed + idx)   # every image always gets the same noise
        else:
            rng = np.random.default_rng()                  # random noise

        noisy = add_gaussian_noise(clean, self.sigma, rng)
        return to_tensor(noisy), to_tensor(clean)
