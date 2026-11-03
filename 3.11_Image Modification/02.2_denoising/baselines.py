# Baseline method for denoising (task 2.2): bilateral filter from skimage.
import torch
from skimage.restoration import denoise_bilateral



def bilateral_denoise(noisy_batch, sigma_color=None, sigma_spatial=1):
    # noisy_batch: tensor (B, 3, H, W), values from 0 to 1
    # sigma_color=None means the default value from skimage
    results = []
    for i in range(noisy_batch.shape[0]):
        img = noisy_batch[i].detach().cpu().permute(1, 2, 0).numpy()      # (3, H, W) -> (H, W, 3), skimage wants this
        out = denoise_bilateral(img, sigma_color=sigma_color,
                                sigma_spatial=sigma_spatial, channel_axis=-1)
        out = torch.from_numpy(out).permute(2, 0, 1).float()  # back to (3, H, W)
        results.append(out)
    return torch.stack(results)

