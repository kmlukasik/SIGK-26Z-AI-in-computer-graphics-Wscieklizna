from pathlib import Path
import cv2


def process_single_image(img, target_size=256):
  h, w = img.shape[:2]

  #cropping to square
  min_dim = min(h, w)
  start_x = (w - min_dim) // 2
  start_y = (h - min_dim) // 2
  square_img = img[start_y : start_y + min_dim, start_x : start_x + min_dim]

  #scaling to 256x256
  interpolation = (
      cv2.INTER_AREA if min_dim > target_size else cv2.INTER_LINEAR
  )
  resized_img = cv2.resize(
      square_img, (target_size, target_size), interpolation=interpolation
  )

  return resized_img


def process_split(input_dir, output_dir, file_slice=None, target_size=256):
  input_path = Path(input_dir)
  output_path = Path(output_dir)
  output_path.mkdir(parents=True, exist_ok=True)

  png_files = sorted([
      p
      for p in input_path.iterdir()
      if p.is_file() and p.suffix.lower() == ".png"
  ])

  if not png_files:
    print(f"Nie znalezniono plików .png w folderze: {input_dir}")
    return

  if file_slice is not None:
    png_files = png_files[file_slice]

  processed_count = 0
  for file_path in png_files:
    img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
    if img is None:
      print(f"Błąd wczytywania pliku (pominięto): {file_path.name}")
      continue

    processed_img = process_single_image(img, target_size)
    out_file_path = output_path / file_path.name
    cv2.imwrite(str(out_file_path), processed_img)
    processed_count += 1

  print(
      f"Zakończono {output_path.name}! Przetworzono {processed_count} plików."
  )


if __name__ == "__main__":
  BASE_DIR = Path(__file__).resolve().parent

  raw_train_hr = BASE_DIR / "data" / "raw" / "DIV2K_train_HR"
  raw_valid_hr = BASE_DIR / "data" / "raw" / "DIV2K_valid_HR"

  base_out = BASE_DIR / "data" / "processed" / "256x256"

  print("preparing dataset splits")

  # train set -> 720 pictures from DIV2K_train_HR
  process_split(
      raw_train_hr,
      base_out / "train",
      file_slice=slice(0, 720),
      target_size=256,
  )

  # validation set -> last 80 pictures from DIV2K_train_HR
  process_split(
      raw_train_hr,
      base_out / "val",
      file_slice=slice(720, 800),
      target_size=256,
  )

  # training set -> 100 pictures from DIV2K_valid_HR
  process_split(
      raw_valid_hr, base_out / "test", file_slice=None, target_size=256
  )

  print("\nDone!")