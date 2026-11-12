import os
import math
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any


def draw_canopy_background(w: int = 1280, h: int = 720) -> np.ndarray:
    """Membuat latar belakang pohon mangga realistis (langit, tajuk dedaunan, batang)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Langit bergradasi biru-putih
    for y in range(h):
        alpha = y / h
        # Langit biru cerah ke bawah
        color = (int(255 - 40 * alpha), int(220 - 30 * alpha), int(150 + 60 * alpha))
        img[y, :] = color

    # Batang pohon utama
    trunk_center = w // 2
    cv2.rectangle(img, (trunk_center - 45, int(h * 0.45)), (trunk_center + 45, h), (35, 55, 80), -1)
    # Cabang-cabang utama
    cv2.line(img, (trunk_center, int(h * 0.5)), (trunk_center - 250, int(h * 0.25)), (35, 55, 80), 30)
    cv2.line(img, (trunk_center, int(h * 0.5)), (trunk_center + 260, int(h * 0.23)), (35, 55, 80), 28)
    cv2.line(img, (trunk_center, int(h * 0.4)), (trunk_center - 120, int(h * 0.15)), (35, 55, 80), 22)
    cv2.line(img, (trunk_center, int(h * 0.4)), (trunk_center + 140, int(h * 0.16)), (35, 55, 80), 20)

    # Tajuk dedaunan lebat (gabungan elips berbagai gradasi hijau)
    np.random.seed(100)
    for _ in range(45):
        cx = int(np.random.normal(trunk_center, w * 0.22))
        cy = int(np.random.normal(h * 0.35, h * 0.16))
        rx = np.random.randint(70, 180)
        ry = np.random.randint(50, 130)
        green_val = np.random.randint(90, 165)
        leaf_color = (np.random.randint(20, 50), green_val, np.random.randint(25, 65))
        cv2.ellipse(img, (cx, cy), (rx, ry), np.random.randint(0, 180), 0, 360, leaf_color, -1)

    return img


def draw_synthetic_mango(
    canvas: np.ndarray,
    center: Tuple[int, int],
    size: Tuple[int, int],
    stage: str,
    angle: float = 15.0
) -> Tuple[int, int, int, int]:
    """Menggambar buah mangga elipsoid dengan gradasi warna sesuai tahap kematangan."""
    cx, cy = center
    rx, ry = size

    # Tangkai buah
    cv2.line(canvas, (cx, cy - ry), (cx - 5, cy - ry - 25), (30, 70, 45), 3)

    if stage == "Unripe":
        base_bgr = (40, 175, 45)      # Hijau tua
        highlight_bgr = (60, 205, 75)
    elif stage == "Mature":
        base_bgr = (30, 200, 160)     # Hijau kekuningan
        highlight_bgr = (40, 225, 200)
    elif stage == "Ripe":
        base_bgr = (30, 170, 235)     # Kuning jingga
        highlight_bgr = (50, 205, 250)
    else:  # Harvest-ready
        base_bgr = (35, 95, 230)      # Jingga kemerahan
        highlight_bgr = (50, 140, 255)

    # Badan buah mangga (elips miring)
    cv2.ellipse(canvas, (cx, cy), (rx, ry), angle, 0, 360, base_bgr, -1)
    # Highlight lengkung pantulan cahaya
    cv2.ellipse(canvas, (cx - int(rx * 0.25), cy - int(ry * 0.25)), (int(rx * 0.5), int(ry * 0.6)), angle, 0, 360, highlight_bgr, -1)
    # Garis tepi halus
    cv2.ellipse(canvas, (cx, cy), (rx, ry), angle, 0, 360, (20, 40, 30), 2)

    x1 = max(0, cx - rx - 2)
    y1 = max(0, cy - ry - 2)
    x2 = min(canvas.shape[1] - 1, cx + rx + 2)
    y2 = min(canvas.shape[0] - 1, cy + ry + 2)
    return (x1, y1, x2, y2)


def generate_synthetic_multi_angle_dataset(
    output_dir: str = "data/raw/photos_multi_angle",
    num_fruits_in_tree: int = 15
) -> Dict[str, Any]:
    """
    Membuat dataset sintetis foto pohon mangga dari 8 arah mata angin (0°, 45°, 90°, ..., 315°)
    dengan posisi 3D buah yang konsisten, mensimulasikan buah yang tampak dari beberapa sudut.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    # Definisikan posisi buah dalam koordinat bola silindris pohon (radius, azimuth_angle_rad, height_y)
    fruits = []
    stages = ["Unripe", "Mature", "Ripe", "Harvest-ready"]
    for fid in range(1, num_fruits_in_tree + 1):
        radius = np.random.uniform(180, 360)
        azimuth = np.random.uniform(0, 2 * math.pi)
        height = np.random.uniform(160, 420)
        stage = stages[fid % len(stages)]
        rx = np.random.randint(22, 34)
        ry = int(rx * np.random.uniform(1.35, 1.6))
        fruits.append({
            "id": fid,
            "radius": radius,
            "azimuth": azimuth,
            "height": height,
            "stage": stage,
            "size": (rx, ry)
        })

    angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]
    w, h = 1280, 720
    camera_center_x = w // 2

    manifest = {"ground_truth_total": num_fruits_in_tree, "angles": {}}

    for angle_deg in angles_deg:
        bg = draw_canopy_background(w, h)
        cam_rad = math.radians(angle_deg)
        visible_fruits_this_angle = []

        for fruit in fruits:
            # Sudut relatif terhadap kamera saat ini
            rel_angle = fruit["azimuth"] - cam_rad
            # Normalisasi ke [-pi, pi]
            rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi

            # Buah hanya terlihat jika menghadap ke arah kamera (lapangan pandang +/- 70 derajat)
            if abs(rel_angle) <= math.radians(72):
                # Proyeksi perspektif sederhana ke bidang 2D
                proj_x = int(camera_center_x + fruit["radius"] * math.sin(rel_angle) * 1.5)
                proj_y = int(fruit["height"] + np.random.normal(0, 3))
                size = fruit["size"]

                if 80 <= proj_x <= w - 80 and 80 <= proj_y <= h - 80:
                    bbox = draw_synthetic_mango(bg, (proj_x, proj_y), size, fruit["stage"])
                    visible_fruits_this_angle.append({
                        "fruit_id": fruit["id"],
                        "stage": fruit["stage"],
                        "bbox": bbox
                    })

        fname = f"angle_{angle_deg:03d}.jpg"
        save_path = os.path.join(output_dir, fname)
        cv2.imwrite(save_path, bg)
        manifest["angles"][fname] = {
            "angle_deg": angle_deg,
            "visible_count": len(visible_fruits_this_angle),
            "visible_fruits": visible_fruits_this_angle
        }

    return manifest


def generate_synthetic_demo_video(
    output_path: str = "data/raw/video/mango_tree_walkaround.mp4",
    total_frames: int = 120,
    fps: int = 30
) -> str:
    """
    Membuat video sintetis rotasi mengelilingi pohon mangga
    untuk menguji pelacakan ByteTrack dan penyambungan tracklet (ID-Switch removal).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    w, h = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    fruits_3d = [
        {"id": 1, "r": 200, "theta": 0.0, "y": 280, "stage": "Ripe", "size": (28, 42)},
        {"id": 2, "r": 280, "theta": 0.5, "y": 220, "stage": "Harvest-ready", "size": (30, 45)},
        {"id": 3, "r": 240, "theta": 1.2, "y": 320, "stage": "Mature", "size": (25, 38)},
        {"id": 4, "r": 220, "theta": 2.1, "y": 260, "stage": "Unripe", "size": (26, 40)},
        {"id": 5, "r": 260, "theta": 3.4, "y": 300, "stage": "Ripe", "size": (27, 41)},
        {"id": 6, "r": 230, "theta": 4.5, "y": 240, "stage": "Harvest-ready", "size": (29, 44)},
        {"id": 7, "r": 250, "theta": 5.4, "y": 350, "stage": "Mature", "size": (24, 36)}
    ]

    for frame_idx in range(total_frames):
        cam_angle = (frame_idx / total_frames) * 2 * math.pi
        frame = draw_canopy_background(w, h)

        for f in fruits_3d:
            rel = (f["theta"] - cam_angle + math.pi) % (2 * math.pi) - math.pi
            if abs(rel) <= math.radians(65):
                proj_x = int((w // 2) + f["r"] * math.sin(rel) * 1.6)
                proj_y = int(f["y"])

                # Simulasi daun menutupi buah sebentar di frame 40-52 (memicu uji ID switch)
                is_occluded = (f["id"] == 1 and 40 <= frame_idx <= 52)
                if not is_occluded:
                    draw_synthetic_mango(frame, (proj_x, proj_y), f["size"], f["stage"])

        out.write(frame)

    out.release()
    return output_path
