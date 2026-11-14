
# Training of the U-Net for low light enhancement.
#
# Usage:
#   python train.py --data_dir data --csv RAISE_258.csv --out_dir runs/unet_l1
#   python train.py --data_dir data --csv RAISE_258.csv --out_dir runs/unet_lpips --lpips_weight 0.05

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import LowLightDataset, read_names, split_names
from metrics import psnr
from models import UNet


def train(
    data_dir,
    csv_path,
    out_dir,
    epochs=50,
    batch_size=16,
    lr=2e-4,
    lpips_weight=0.0,
    base=32,
    crops_per_image=20,
    workers=0,
    seed=0
):
  
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(out_dir, exist_ok=True)

    # Dataset
    train_names, val_names, _ = split_names(read_names(csv_path))

    train_ds = LowLightDataset(
        data_dir,
        train_names,
        mode='train',
        crops_per_image=crops_per_image,
        seed=seed
    )

    val_ds = LowLightDataset(
        data_dir,
        val_names,
        mode='val',
        crops_per_image=16,
        seed=seed + 1
    )

    # Separate generator for DataLoader shuffling
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        generator=loader_generator
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers
    )

    # Model and optimizer
    model = UNet(base=base).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    loss_fn = torch.nn.L1Loss()

    # Optional LPIPS loss
    lpips_model = None

    if lpips_weight > 0:
        import lpips

        lpips_model = lpips.LPIPS(net='alex').to(device)
        lpips_model.eval()

        for p in lpips_model.parameters():
            p.requires_grad_(False)


    # Training
    print(f'device: {device}')
    print(f'train crops per epoch: {len(train_ds)}')
    print(f'val crops: {len(val_ds)}')
    print(f'seed: {seed}')

    best_val = -float('inf')
    history = []

    best_path = os.path.join(out_dir, 'best.pt')

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        model.train()
        total_loss = 0.0

        for dark, gt in train_dl:
            dark = dark.to(device)
            gt = gt.to(device)

            pred = model(dark)

            loss = loss_fn(pred, gt)

            if lpips_model is not None:
                # LPIPS expects images in [-1, 1].
                pred_lpips = pred.clamp(0, 1) * 2 - 1
                gt_lpips = gt * 2 - 1

                loss = loss + lpips_weight * lpips_model(
                    pred_lpips,
                    gt_lpips
                ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_dl)


        # Validation
        model.eval()
        values = []

        with torch.no_grad():
            for dark, gt in val_dl:
                dark = dark.to(device)
                gt = gt.to(device)

                pred = model(dark).clamp(0, 1)

                batch_psnr = psnr(pred, gt)
                values += batch_psnr.cpu().tolist()

        val_psnr = float(np.mean(values))

        current_lr = optimizer.param_groups[0]['lr']

        epoch_info = {
            'epoch': epoch,
            'loss': avg_loss,
            'val_psnr': val_psnr,
            'lr': current_lr
        }

        history.append(epoch_info)

        elapsed = time.time() - t0

        print(
            f"epoch {epoch}: "
            f"loss {avg_loss:.4f} | "
            f"val PSNR {val_psnr:.2f} | "
            f"lr {current_lr:.6f} | "
            f"{elapsed:.0f}s"
        )

        if val_psnr > best_val:
            best_val = val_psnr

            torch.save(
                model.state_dict(),
                best_path
            )

            print(f'  -> saved best model (PSNR: {best_val:.2f})')

        with open(
            os.path.join(out_dir, 'history.json'),
            'w',
            encoding='utf-8'
        ) as f:
            json.dump(history, f, indent=2)

    # Load the best model before returning
    model.load_state_dict(
        torch.load(best_path, map_location=device, weights_only=True)
    )

    model.eval()

    print(f'\nTraining finished. Best validation PSNR: {best_val:.2f}')
    print(f'Best model saved to: {best_path}')

    return model, history


if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out_dir', required=True)

    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--lpips_weight', type=float, default=0.0)
    ap.add_argument('--base', type=int, default=32)
    ap.add_argument('--crops_per_image', type=int, default=20)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)

    a = ap.parse_args()

    train(
        data_dir=a.data_dir,
        csv_path=a.csv,
        out_dir=a.out_dir,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        lpips_weight=a.lpips_weight,
        base=a.base,
        crops_per_image=a.crops_per_image,
        workers=a.workers,
        seed=a.seed
    )
