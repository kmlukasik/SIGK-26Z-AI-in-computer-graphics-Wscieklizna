
# Dataset for low light enhancement.
# Returns pairs (dark noisy image, ground truth),
# both with shape (3, H, W) and values in range 0..1.
#
# Dataset split:
#   train - first 72 images
#   val   - next 8 images
#   test  - last 20 images
#
# train: random 256x256 crops + flips and rotations
# val:   deterministic 256x256 crops
# test:  whole images without augmentation

import csv
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def read_names(csv_path, limit=100):
    names = []

    with open(csv_path, encoding='utf-8') as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= limit:
                break
            names.append(row['File'])

    return names


def split_names(names):
    return names[:-28], names[-28:-20], names[-20:]


def load_rgb(path):
    img = cv2.imread(path)

    if img is None:
        raise FileNotFoundError(path)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_tensor(img_uint8):
    return torch.from_numpy(
        img_uint8.astype(np.float32).transpose(2, 0, 1) / 255.0
    )


def random_crop_pair(gt, dark, size, rng, augment):
    # The same crop and transformations are applied to both images.
    # This keeps the ground truth and dark image aligned.

    h, w = gt.shape[:2]

    if h < size or w < size:
        raise ValueError(
            f"Image is too small for crop size {size}: "
            f"image size is {w}x{h}"
        )

    y = int(rng.integers(0, h - size + 1))
    x = int(rng.integers(0, w - size + 1))

    gt_c = gt[y:y + size, x:x + size]
    dark_c = dark[y:y + size, x:x + size]

    if augment:
        # Horizontal flip
        if rng.random() < 0.5:
            gt_c = gt_c[:, ::-1]
            dark_c = dark_c[:, ::-1]

        # Vertical flip
        if rng.random() < 0.5:
            gt_c = gt_c[::-1]
            dark_c = dark_c[::-1]

        # Rotation by 0, 90, 180 or 270 degrees
        k = int(rng.integers(0, 4))
        gt_c = np.rot90(gt_c, k)
        dark_c = np.rot90(dark_c, k)

    return (
        np.ascontiguousarray(gt_c),
        np.ascontiguousarray(dark_c)
    )


class LowLightDataset(Dataset):
    """
    mode:
        'train' - random crops and data augmentation
        'val'   - deterministic random crops
        'test'  - complete images without augmentation
    """

    def __init__(
        self,
        data_dir,
        names,
        mode='train',
        crop=256,
        crops_per_image=20,
        seed=1234
    ):
        if mode not in ('train', 'val', 'test'):
            raise ValueError(
                "mode must be 'train', 'val' or 'test'"
            )

        self.gt = [
            load_rgb(os.path.join(data_dir, 'gt', f'{n}_gt.png'))
            for n in names
        ]

        self.dark = [
            load_rgb(os.path.join(data_dir, 'dark', f'{n}_input.png'))
            for n in names
        ]

        self.mode = mode
        self.crop = crop
        self.crops_per_image = crops_per_image
        self.seed = seed

        # Persistent RNG for training.
        # It generates new crops and augmentations on each call,
        # while allowing reproducibility with the same seed.
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        if self.mode == 'test':
            return len(self.gt)

        return len(self.gt) * self.crops_per_image

    def __getitem__(self, i):
        idx = i % len(self.gt)

        gt = self.gt[idx]
        dark = self.dark[idx]

        # Test: use the complete image without random transformations.
        if self.mode == 'test':
            return to_tensor(dark), to_tensor(gt)

        # Train: use persistent RNG and apply augmentation.
        if self.mode == 'train':
            rng = self.rng
            augment = True

        # Validation: use the same crop for a given index every time.
        else:
            rng = np.random.default_rng(self.seed + i)
            augment = False

        gt_c, dark_c = random_crop_pair(
            gt,
            dark,
            self.crop,
            rng,
            augment=augment
        )

        return to_tensor(dark_c), to_tensor(gt_c)