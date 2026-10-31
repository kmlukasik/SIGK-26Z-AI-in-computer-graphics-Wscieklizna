# Image quality metrics: SNE, PSNR, SSIM, LPIPS.
# All functions take tensors of shape (B, 3, H, W) with values from 0 to 1
# and return one number per image.
import torch
from torchmetrics.functional import structural_similarity_index_measure


def sne(pred, target):
    # Squared Norm Error: sum of squared differences of all corresponding pixels (lower is better)
    return ((pred - target) ** 2).sum(dim=(1, 2, 3))


def psnr(pred, target):
    # mean squared error of one image; the highest possible pixel value is 1.0 (higher is better)
    mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))
    return 10 * torch.log10(1.0 / mse)


def ssim(pred, target):
    # reduction='none' gives a separate value for every image (higher is better)
    return structural_similarity_index_measure(pred, target, data_range=1.0, reduction='none')


def lpips_value(lpips_model, pred, target):
    # LPIPS wants values from -1 to 1, not from 0 to 1 (lower is better)
    return lpips_model(pred * 2 - 1, target * 2 - 1).flatten()


def all_metrics(pred, target, lpips_model):
    # all metrics for one batch of images
    return {
        'SNE': sne(pred, target),
        'PSNR': psnr(pred, target),
        'SSIM': ssim(pred, target),
        'LPIPS': lpips_value(lpips_model, pred, target),
    }
