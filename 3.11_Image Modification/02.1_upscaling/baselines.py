import cv2
import torch

def bicubic_upscale(lr_batch, target_size=256):
    # lr_batch: tensor (B, 3, H_lr, W_lr), values from 0 to 1
    # Upscales each image to (target_size, target_size) using OpenCV INTER_CUBIC
    results = []
    for i in range(lr_batch.shape[0]):
        img = lr_batch[i].detach().cpu().permute(1, 2, 0).numpy()     # (3, H_lr, W_lr) -> (H_lr, W_lr, 3)
        out = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        out = out.clip(0.0, 1.0)
        out = torch.from_numpy(out).permute(2, 0, 1).float()          # back to (3, 256, 256)
        results.append(out)
    return torch.stack(results)