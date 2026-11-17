# Download of the RAISE dataset (Nikon D7000, indoor) and generation of image pairs:
#   <name>_gt.png     - properly exposed image (ground truth)
#   <name>_input.png  - dark (-3 EV) image with physical sensor noise (input of the network)
#
# Usage (works on Colab and locally):
#   python raise_data.py --csv RAISE_258.csv --raw_dir raw --out_dir data --limit 100
import argparse
import csv
import os
import zlib

import cv2
import numpy as np
import rawpy
import requests


def read_file_ids(csv_path, limit=None):
    # names of the files in the same order as in the csv (the order matters: last 20 = test set)
    ids = []
    with open(csv_path, encoding='utf-8') as f:
        for i, row in enumerate(csv.DictReader(f)):
            if limit is not None and i >= limit:
                break
            ids.append(row['File'])
    return ids


def download_raise_nef(csv_path, output_dir, limit=None):
    # Downloads NEF files listed in the csv. Files that already exist are skipped.
    os.makedirs(output_dir, exist_ok=True)
    with open(csv_path, mode='r', encoding='utf-8') as file:
        for i, row in enumerate(csv.DictReader(file)):
            if limit is not None and i >= limit:
                break
            filename = row['File']
            if not filename.upper().endswith('.NEF'):
                filename += '.NEF'
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                print(f'Skipping {filename} - already exists.')
                continue
            print(f'Downloading {filename} ({i + 1}/{limit})...')
            response = requests.get(row['NEF'], stream=True, timeout=60)
            response.raise_for_status()
            tmp_path = filepath + '.part'              # a broken download never looks like a finished file
            with open(tmp_path, 'wb') as out_file:
                for chunk in response.iter_content(chunk_size=8192):
                    out_file.write(chunk)
            os.replace(tmp_path, filepath)


def get_ground_truth(raw):
    # properly exposed sRGB image; half_size=True makes the image 2 times smaller (faster)
    return raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=True)


def get_linear_image(raw):
    # linear image (no gamma, no auto brightness), float32 in the range 0..1
    linear_16bit = raw.postprocess(use_camera_wb=True, gamma=(1, 1), no_auto_bright=True,
                                   output_bps=16, half_size=True)
    return linear_16bit.astype(np.float32) / 65535.0


def reduce_exposure(linear_img, ev_shift):
    # -1 EV = half of the light, so the scale factor is 2 ** ev_shift (for -3 EV it is 1/8)
    return linear_img * (2.0 ** ev_shift)


def add_physical_noise(linear_img, alpha=0.001, sigma=0.005):
    # alpha: Poisson (photon) noise, sigma: Gaussian (electronic read) noise
    noisy_img = np.random.poisson(linear_img / alpha) * alpha
    noisy_img += np.random.normal(0, sigma, linear_img.shape)
    return np.clip(noisy_img, 0.0, 1.0)


def apply_gamma_correction(img, gamma=2.2):
    return np.power(np.clip(img, 0.0, 1.0), 1.0 / gamma)


def convert_to_8bit(img):
    return (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def get_low_light(raw, ev_shift=-3.0):
    linear_img = get_linear_image(raw)                         # 1. linear image
    dark_linear = reduce_exposure(linear_img, ev_shift)        # 2. less light
    noisy_dark_linear = add_physical_noise(dark_linear)        # 3. sensor noise (still linear!)
    dark_srgb = apply_gamma_correction(noisy_dark_linear)      # 4. gamma, only for display
    return convert_to_8bit(dark_srgb)                          # 5. 8 bit


def save_png(path, img_rgb):
    cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def process_single_file(file_path, gt_dir, dark_dir, ev_shift):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    gt_path = os.path.join(gt_dir, f'{base_name}_gt.png')
    dark_path = os.path.join(dark_dir, f'{base_name}_input.png')
    if os.path.exists(gt_path) and os.path.exists(dark_path):
        print(f'Skipping {base_name} - already processed.')
        return
    print(f'Processing {base_name}...')
    np.random.seed(zlib.crc32(base_name.encode()) % (2 ** 32))   # the same noise every time the script is run
    with rawpy.imread(file_path) as raw:
        img_gt = get_ground_truth(raw)
        img_dark = get_low_light(raw, ev_shift)
    save_png(gt_path, img_gt)
    save_png(dark_path, img_dark)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--raw_dir', default='raw')
    ap.add_argument('--out_dir', default='data')
    ap.add_argument('--limit', type=int, default=100)
    ap.add_argument('--ev', type=float, default=-3.0)
    ap.add_argument('--skip_download', action='store_true')
    args = ap.parse_args()

    if not args.skip_download:
        download_raise_nef(args.csv, args.raw_dir, limit=args.limit)

    gt_dir = os.path.join(args.out_dir, 'gt')
    dark_dir = os.path.join(args.out_dir, 'dark')
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(dark_dir, exist_ok=True)
    for file_id in read_file_ids(args.csv, args.limit):
        try:
            process_single_file(os.path.join(args.raw_dir, file_id + '.NEF'), gt_dir, dark_dir, args.ev)
        except Exception as e:
            print(f'Error processing {file_id}: {e}')


if __name__ == '__main__':
    main()
