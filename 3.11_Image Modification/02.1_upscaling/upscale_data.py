"""Prepare data for the upscaling task (2.1).
Images are 256x256 PNGs, scaled to <0.0, 1.0>.
We downscale them to lr_size (64x64 or 32x32) using OpenCV resize with INTER_AREA.
"""
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def to_tensor(img):
    """(H, W, 3) numpy -> (3, H, W) torch tensor."""
    return torch.from_numpy(img.transpose(2, 0, 1).copy())


class UpscaleDataset(Dataset):
    """Returns pairs (lr_image, hr_image).

    folder  - path to folder with 256x256 PNGs ('train', 'val', or 'test')
    lr_size - input low-resolution size: 64 (for 4x) or 32 (for 8x)
    """

    def __init__(self, folder, lr_size=64):
        self.paths = sorted(glob.glob(f'{folder}/*.png'))
        if len(self.paths) == 0:
            raise FileNotFoundError(f'No png files in {folder}')
        self.lr_size = lr_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_bgr = cv2.imread(self.paths[idx], cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        hr_clean = img_rgb.astype(np.float32) / 255.0  # 0..255 -> 0.0..1.0

        # Downscale to lr_size x lr_size using OpenCV INTER_AREA (documented for report)
        lr_img = cv2.resize(
            hr_clean, (self.lr_size, self.lr_size), interpolation=cv2.INTER_AREA
        )
        lr_img = np.clip(lr_img, 0.0, 1.0).astype(np.float32)

        return to_tensor(lr_img), to_tensor(hr_clean)