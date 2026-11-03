import json
import os
from pathlib import Path
import sys
import lpips
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]  # wskazuje na 3.11_Image Modification
sys.path.append(str(BASE_DIR))

from baselines import bicubic_upscale
from metrics import all_metrics, psnr
from models import EDSR
from upscale_data import UpscaleDataset

#configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = 'data/processed/256x256'  # folder with train (720), val (80), test (100)
out_dir = 'results_upscaling'        # folder for checkpoints, json logs, and plots
os.makedirs(out_dir, exist_ok=True)

lpips_model = lpips.LPIPS(net='alex', verbose=False).to(device)
print(f'Device: {device}')

for folder in ['train', 'val', 'test']:
  print(folder, len(os.listdir(f'{data_dir}/{folder}')))


#downscaling
def plot_downscaling_examples():
  ds_64 = UpscaleDataset(f'{data_dir}/test', lr_size=64)
  ds_32 = UpscaleDataset(f'{data_dir}/test', lr_size=32)

  lr64, clean = ds_64[0]
  lr32, _ = ds_32[0]

  fig, axes = plt.subplots(1, 3, figsize=(12, 4))
  imgs = [clean, lr64, lr32]
  titles = [
      'Original HR (256×256)',
      'Input LR (64×64) - Scale 4x',
      'Input LR (32×32) - Scale 8x',
  ]

  for ax, im, title in zip(axes, imgs, titles):
    ax.imshow(im.permute(1, 2, 0).numpy())
    ax.set_title(title, fontsize=11)
    ax.axis('off')
  plt.tight_layout()
  plt.savefig(f'{out_dir}/01_lr_inputs_example.png', dpi=150)
  plt.close()


# ==========================================
# 3. Helper evaluation function
# ==========================================
def evaluate_loader(predict_fn, loader):
  totals = {'SNE': 0.0, 'PSNR': 0.0, 'SSIM': 0.0, 'LPIPS': 0.0}
  n = 0
  with torch.no_grad():
    for lr_img, hr_img in loader:
      pred = predict_fn(lr_img).to(device).clamp(0.0, 1.0)
      hr_img = hr_img.to(device)
      batch_m = all_metrics(pred, hr_img, lpips_model)
      for k in totals:
        totals[k] += batch_m[k].sum().item()
      n += lr_img.shape[0]
  return {k: totals[k] / n for k in totals}



#training EDSR

def train_edsr(lr_size=64, epochs=100, batch_size=8, lr=1e-4):
  upscale_factor = 256 // lr_size
  weights_path = f'{out_dir}/edsr_{lr_size}to256.pt'
  hist_path = f'{out_dir}/edsr_{lr_size}to256_history.json'

  train_ds = UpscaleDataset(f'{data_dir}/train', lr_size=lr_size)
  val_ds = UpscaleDataset(f'{data_dir}/val', lr_size=lr_size)

  train_loader = DataLoader(
      train_ds, batch_size=batch_size, shuffle=True, num_workers=0
  )
  val_loader = DataLoader(
      val_ds, batch_size=batch_size, shuffle=False, num_workers=0
  )

  model = EDSR(
      upscale_factor=upscale_factor,
      num_channels=256,
      num_res_blocks=32,
      res_scale=0.1,
  ).to(device)

  if os.path.exists(weights_path) and os.path.exists(hist_path):
    print(f'Loading existing checkpoint from {weights_path}')
    model.load_state_dict(torch.load(weights_path, map_location=device))
    with open(hist_path, 'r') as f:
      history = json.load(f)
    return model, history

  print(
      f'\n--- Training EDSR for input {lr_size}x{lr_size} (Scale'
      f' {upscale_factor}x) ---'
  )
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=epochs, eta_min=1e-6
  )
  loss_fn = nn.L1Loss()

  history = {'train_loss': [], 'val_loss': [], 'val_psnr': []}
  best_psnr = -1.0

  for epoch in range(1, epochs + 1):
    model.train()
    tr_loss = 0.0
    for lr_img, hr_img in tqdm(
        train_loader, desc=f'Epoch {epoch}/{epochs}', leave=False
    ):
      lr_img, hr_img = lr_img.to(device), hr_img.to(device)
      optimizer.zero_grad()
      pred = model(lr_img)
      loss = loss_fn(pred, hr_img)
      loss.backward()
      optimizer.step()
      tr_loss += loss.item() * lr_img.shape[0]
    tr_loss /= len(train_ds)

    scheduler.step()

    # Validation (80 images)
    model.eval()
    va_loss = 0.0
    va_psnr = 0.0
    with torch.no_grad():
      for lr_img, hr_img in val_loader:
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)
        pred = model(lr_img)
        va_loss += loss_fn(pred, hr_img).item() * lr_img.shape[0]
        va_psnr += psnr(pred, hr_img).sum().item()
    va_loss /= len(val_ds)
    va_psnr /= len(val_ds)

    history['train_loss'].append(tr_loss)
    history['val_loss'].append(va_loss)
    history['val_psnr'].append(va_psnr)

    if va_psnr > best_psnr:
      best_psnr = va_psnr
      torch.save(model.state_dict(), weights_path)

    if epoch % 5 == 0 or epoch == 1:
      print(
          f'Epoch {epoch:02d}/{epochs} | train L1: {tr_loss:.5f} | val L1:'
          f' {va_loss:.5f} | val PSNR: {va_psnr:.2f} dB (best: {best_psnr:.2f})'
      )

  with open(hist_path, 'w') as f:
    json.dump(history, f)

  model.load_state_dict(torch.load(weights_path, map_location=device))
  return model, history



# trainig curves

def plot_training_curves(hist_64, hist_32):
  #loss curve
  fig, axes = plt.subplots(1, 2, figsize=(13, 4))
  for ax, (lr_size, h) in zip(axes, [(64, hist_64), (32, hist_32)]):
    scale = 256 // lr_size
    ax.plot(h['train_loss'], label='Train L1 loss')
    ax.plot(h['val_loss'], label='Val L1 loss')
    ax.set_title(f'EDSR Training Loss — input {lr_size}×{lr_size} (scale {scale}x)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('L1 Loss')
    ax.legend()
    ax.grid(alpha=0.3)
  plt.tight_layout()
  plt.savefig(f'{out_dir}/02_training_curves_loss.png', dpi=150)
  plt.close()

  # PSNR curve
  fig, axes = plt.subplots(1, 2, figsize=(13, 4))
  for ax, (lr_size, h) in zip(axes, [(64, hist_64), (32, hist_32)]):
    scale = 256 // lr_size
    best_ep = int(np.argmax(h['val_psnr'])) + 1
    best_p = max(h['val_psnr'])

    ax.plot(h['val_psnr'], color='green', label='Val PSNR [dB]')
    ax.scatter(
        [best_ep - 1],
        [best_p],
        color='red',
        zorder=5,
        label=f'Best: {best_p:.2f} dB (ep. {best_ep})',
    )
    ax.set_title(f'Validation PSNR — input {lr_size}×{lr_size} (scale {scale}x)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PSNR [dB]')
    ax.legend()
    ax.grid(alpha=0.3)
  plt.tight_layout()
  plt.savefig(f'{out_dir}/03_training_curves_psnr.png', dpi=150)
  plt.close()



# comparison with zoom patch

def show_comparison(idx, lr_size, model, patch_xy=(90, 90), patch_size=64):
  test_ds = UpscaleDataset(f'{data_dir}/test', lr_size=lr_size)
  lr_img, clean = test_ds[idx]
  lr_b = lr_img.unsqueeze(0)
  clean_b = clean.unsqueeze(0).to(device)

  nearest_b = torch.nn.functional.interpolate(
      lr_b, size=(256, 256), mode='nearest'
  ).to(device)
  bicubic_b = bicubic_upscale(lr_b, target_size=256).to(device)

  model.eval()
  with torch.no_grad():
    edsr_b = model(lr_b.to(device)).clamp(0.0, 1.0)

  imgs = [nearest_b[0].cpu(), bicubic_b[0].cpu(), edsr_b[0].cpu(), clean]
  names = [
      f'Input LR ({lr_size}×{lr_size})',
      'OpenCV Bicubic baseline',
      'EDSR reconstruction',
      'Original HR (256×256)',
  ]

  titles = []
  for name, im in zip(names, imgs):
    if name.startswith('Original'):
      titles.append(name)
    else:
      m = all_metrics(im.unsqueeze(0).to(device), clean_b, lpips_model)
      titles.append(
          f"{name}\nPSNR={m['PSNR'].item():.2f} dB, SSIM={m['SSIM'].item():.3f}"
      )

  x0, y0 = patch_xy
  fig, axes = plt.subplots(2, 4, figsize=(16, 8))
  for col, (im, title) in enumerate(zip(imgs, titles)):
    np_im = im.permute(1, 2, 0).numpy()
    # Row 0: Full image with patch rectangle
    axes[0, col].imshow(np_im)
    axes[0, col].add_patch(
        Rectangle(
            (x0, y0),
            patch_size,
            patch_size,
            edgecolor='red',
            facecolor='none',
            lw=2,
        )
    )
    axes[0, col].set_title(title, fontsize=11)
    axes[0, col].axis('off')

    # Row 1: Zoom patch
    patch = np_im[y0 : y0 + patch_size, x0 : x0 + patch_size]
    axes[1, col].imshow(patch)
    axes[1, col].set_title('Zoom patch', fontsize=10)
    axes[1, col].axis('off')

  plt.suptitle(
      f'Method comparison — input {lr_size}×{lr_size} (test image #{idx})',
      fontsize=13,
  )
  plt.tight_layout()
  plt.savefig(
      f'{out_dir}/04_comparison_{lr_size}x{lr_size}_img{idx}.png', dpi=150
  )
  plt.close()



# 7. Metrics bar charts
def plot_metrics_bars(df):
  fig, axes = plt.subplots(1, 4, figsize=(16, 4))
  metrics_info = [
      ('SNE', 'lower is better'),
      ('PSNR', 'higher is better [dB]'),
      ('SSIM', 'higher is better'),
      ('LPIPS', 'lower is better'),
  ]
  for ax, (m, desc) in zip(axes, metrics_info):
    pivot = df.pivot(index='Input size', columns='Method', values=m)
    pivot = pivot[
        ['Input LR (Nearest)', 'OpenCV Bicubic baseline', 'EDSR reconstruction']
    ]
    pivot.plot(kind='bar', ax=ax, rot=0)
    ax.set_title(f'{m} ({desc})')
    ax.grid(alpha=0.3, axis='y')
    ax.legend(fontsize=8)
  plt.tight_layout()
  plt.savefig(f'{out_dir}/05_metrics_bar_charts.png', dpi=150)
  plt.close()


if __name__ == '__main__':
  plot_downscaling_examples()

  #edsr training
  edsr_64, hist_64 = train_edsr(lr_size=64, epochs=100, batch_size=8, lr=1e-4)
  edsr_32, hist_32 = train_edsr(lr_size=32, epochs=100, batch_size=8, lr=1e-4)

  plot_training_curves(hist_64, hist_32)

  # 2. Evaluation on the TEST set (100 images)
  rows = []
  for lr_size, model in [(64, edsr_64), (32, edsr_32)]:
    test_ds = UpscaleDataset(f'{data_dir}/test', lr_size=lr_size)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    m_near = evaluate_loader(
        lambda x: torch.nn.functional.interpolate(
            x, size=(256, 256), mode='nearest'
        ),
        test_loader,
    )
    m_bic = evaluate_loader(
        lambda x: bicubic_upscale(x, target_size=256), test_loader
    )
    model.eval()
    m_edsr = evaluate_loader(
        lambda x, m=model: m(x.to(device)), test_loader
    )

    for name, m in [
        ('Input LR (Nearest)', m_near),
        ('OpenCV Bicubic baseline', m_bic),
        ('EDSR reconstruction', m_edsr),
    ]:
      rows.append({'Input size': f'{lr_size}×{lr_size}', 'Method': name, **m})

  df_results = pd.DataFrame(rows)
  print('\n=== TEST SET RESULTS (100 IMAGES) ===')
  print(df_results.to_string(index=False))
  df_results.to_csv(f'{out_dir}/test_results_2_1.csv', index=False)

  # 3. Bar charts and visual zoom comparisons
  plot_metrics_bars(df_results)

  for idx in [0, 5, 12]:
    show_comparison(
        idx=idx, lr_size=64, model=edsr_64, patch_xy=(90, 90), patch_size=64
    )
    show_comparison(
        idx=idx, lr_size=32, model=edsr_32, patch_xy=(90, 90), patch_size=64
    )

  print(f'\nAll plots, tables, and images saved to: {out_dir}/')
