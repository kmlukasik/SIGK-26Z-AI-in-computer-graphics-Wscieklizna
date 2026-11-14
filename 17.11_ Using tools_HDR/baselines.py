# Baseline for low light enhancement: histogram equalization.
import cv2
import numpy as np
import torch


def histogram_equalization(dark_batch):
    # dark_batch: tensor (B, 3, H, W), values from 0 to 1 (the low-light input)
    results = []
    for i in range(dark_batch.shape[0]):
        img = (dark_batch[i].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)   # (H, W, 3), RGB, 8 bit
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])      # equalize only the brightness channel
        out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        results.append(torch.from_numpy(out.astype(np.float32) / 255.0).permute(2, 0, 1))
    return torch.stack(results)
