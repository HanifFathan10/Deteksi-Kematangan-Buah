import os
import shutil
import random
from typing import Tuple, Dict, Any, List, Optional
import cv2
import numpy as np


def compute_blur_metric(image: np.ndarray) -> float:
    """
    Menghitung skor ketajaman citra menggunakan varians Laplacian.
    Semakin rendah skor, semakin blur citranya.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(variance)


def is_image_blurry(image: np.ndarray, threshold: float = 80.0) -> Tuple[bool, float]:
    """
    Mengecek apakah gambar blur di bawah ambang batas (threshold).
    Rekomendasi threshold: 70 - 100 untuk citra ponsel.
    """
    metric = compute_blur_metric(image)
    return (metric < threshold, metric)


def resize_keep_aspect(image: np.ndarray, max_dim: int = 1280) -> Tuple[np.ndarray, float]:
    """
    Resize citra dengan mempertahankan aspek rasio sesuai contoh kode modul UTS (Halaman 10-11).
    scale = max_dim / max(h, w)
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image.copy(), 1.0
    scale = float(max_dim) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def letterbox_resize(
    image: np.ndarray,
    target_shape: Tuple[int, int] = (640, 640),
    fill_color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize citra dengan teknik Letterboxing (standar YOLO) tanpa distorsi bentuk buah mangga.
    """
    h, w = image.shape[:2]
    target_w, target_h = target_shape

    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_w - new_w
    pad_h = target_h - new_h

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=fill_color
    )
    return padded, scale, (left, top)


def enhance_contrast_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Menyesuaikan pencahayaan dan kontras menggunakan CLAHE pada kanal L (ruang warna LAB).
    Mempertahankan warna asli mangga sekaligus memperjelas buah di area bayangan tajuk.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    merged = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def extract_tree_canopy_mask(image: np.ndarray) -> np.ndarray:
    """
    Membuat segmentasi kasar tajuk pohon mangga berbasis rentang warna dedaunan (HSV).
    Berguna untuk cropping dan memisahkan latar belakang langit/tanah.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Rentang hijau dedaunan pohon dan kuning-kehijauan buah
    lower_green = np.array([25, 30, 20])
    upper_green = np.array([95, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def preprocess_single_image(
    image: np.ndarray,
    max_dim: int = 1280,
    blur_threshold: float = 80.0,
    enhance: bool = True
) -> Dict[str, Any]:
    """
    Pipeline preprocessing lengkap untuk satu citra sesuai Bab 4 panduan UTS.
    """
    blur_flag, blur_score = is_image_blurry(image, threshold=blur_threshold)
    resized, scale = resize_keep_aspect(image, max_dim=max_dim)

    if enhance:
        processed = enhance_contrast_clahe(resized)
    else:
        processed = resized.copy()

    return {
        "processed_image": processed,
        "is_blurry": blur_flag,
        "blur_score": blur_score,
        "scale": scale,
        "original_shape": image.shape[:2],
        "processed_shape": processed.shape[:2]
    }


def batch_preprocess_folder(
    input_dir: str,
    output_dir: str,
    max_dim: int = 1280,
    blur_threshold: float = 80.0,
    skip_blurry: bool = False
) -> List[Dict[str, Any]]:
    """
    Memproses seluruh citra dalam folder dan menyimpannya ke folder output.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    valid_exts = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".bmp"}
    files = [f for f in os.listdir(input_dir) if os.path.splitext(f.lower())[1] in valid_exts]

    for fname in sorted(files):
        in_path = os.path.join(input_dir, fname)
        img = cv2.imread(in_path)
        if img is None:
            continue

        res = preprocess_single_image(img, max_dim=max_dim, blur_threshold=blur_threshold)
        res["filename"] = fname

        if res["is_blurry"] and skip_blurry:
            res["status"] = "skipped_blurry"
        else:
            out_path = os.path.join(output_dir, fname)
            cv2.imwrite(out_path, res["processed_image"])
            res["status"] = "processed"
            res["saved_path"] = out_path

        results.append(res)

    return results


def build_yolo_dataset_structure(
    base_dir: str,
    images: List[Tuple[str, np.ndarray, Optional[List[str]]]],
    split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    random_seed: int = 42
) -> Dict[str, int]:
    """
    Membangun struktur dataset standar YOLO seperti yang diminta pada Halaman 10 modul UTS:
    MangoSmartphone/
    ├── images/{train, valid, test}
    └── labels/{train, valid, test}
    """
    random.seed(random_seed)
    shuffled = images.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * split_ratios[0])
    n_valid = int(n * split_ratios[1])

    splits = {
        "train": shuffled[:n_train],
        "valid": shuffled[n_train:n_train + n_valid],
        "test": shuffled[n_train + n_valid:]
    }

    counts = {}
    for split_name, split_items in splits.items():
        img_dir = os.path.join(base_dir, "images", split_name)
        lbl_dir = os.path.join(base_dir, "labels", split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        counts[split_name] = len(split_items)
        for base_name, img_arr, label_lines in split_items:
            img_out = os.path.join(img_dir, f"{base_name}.jpg")
            cv2.imwrite(img_out, img_arr)

            lbl_out = os.path.join(lbl_dir, f"{base_name}.txt")
            with open(lbl_out, "w", encoding="utf-8") as f:
                if label_lines:
                    for line in label_lines:
                        f.write(line.strip() + "\n")

    return counts
