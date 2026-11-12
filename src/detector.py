import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

# Kelas tingkat kematangan buah mangga sesuai Halaman 21 Panduan UTS
MATURITY_CLASSES = {
    0: "Unripe",        # Belum matang (kulit hijau pekat)
    1: "Mature",        # Matang fisiologis (mulai menguning/pucat)
    2: "Ripe",          # Matang lanjut (kuning jingga dominan)
    3: "Harvest-ready"  # Siap panen (kuning jingga kemerahan optimal)
}

MATURITY_COLORS = {
    "Unripe": (46, 204, 113),       # Hijau segar (BGR)
    "Mature": (0, 215, 255),        # Kuning emas
    "Ripe": (34, 126, 230),         # Oranye terang
    "Harvest-ready": (50, 50, 235), # Merah oranye matang
    "mango": (0, 255, 0)            # Default hijau
}


@dataclass
class MangoDetection:
    """Representasi satu buah mangga yang terdeteksi."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    maturity_stage: str
    maturity_confidence: float = 1.0
    crop_image: Optional[np.ndarray] = None
    color_hist: Optional[np.ndarray] = None
    feature_vector: Optional[np.ndarray] = None
    track_id: Optional[int] = None

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


def extract_fruit_color_signature(crop: np.ndarray) -> Tuple[str, float, np.ndarray]:
    """
    Menganalisis kematangan buah dari warna kulit (HSV & LAB) sesuai Bab 8.
    Menghitung distribusi rasio warna hijau, kuning, dan jingga kemerahan.
    """
    if crop is None or crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return "Unripe", 0.5, np.zeros(32, dtype=np.float32)

    # Resize crop untuk standarisasi ekstraksi fitur
    std_crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(std_crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(std_crop, cv2.COLOR_BGR2LAB)

    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    b_channel_lab = lab[:, :, 2]  # Indikator warna kuning-biru (+b = kuning)

    # Masking area buah (abaikan pixel yang sangat gelap/terang akibat bayangan/glare)
    valid_mask = (s_channel > 30) & (hsv[:, :, 2] > 30)
    total_valid = np.count_nonzero(valid_mask)

    if total_valid < 20:
        return "Unripe", 0.5, np.zeros(32, dtype=np.float32)

    valid_h = h_channel[valid_mask]
    valid_b_lab = b_channel_lab[valid_mask]

    # Hitung porsi piksel warna
    green_pixels = np.count_nonzero((valid_h >= 35) & (valid_h <= 85))
    yellow_pixels = np.count_nonzero((valid_h >= 20) & (valid_h < 35))
    orange_red_pixels = np.count_nonzero((valid_h < 20) | (valid_h > 165))

    p_green = green_pixels / total_valid
    p_yellow = yellow_pixels / total_valid
    p_orange = orange_red_pixels / total_valid
    avg_yellow_index = float(np.mean(valid_b_lab))

    # Klasifikasi tingkat kematangan berdasar rasio warna & kematangan fisiologis
    if p_green >= 0.55:
        if p_yellow >= 0.20:
            stage = "Mature"
            conf = float(min(0.92, max(0.60, p_yellow * 1.5 + p_green * 0.5)))
        else:
            stage = "Unripe"
            conf = float(min(0.98, max(0.65, p_green * 1.2)))
    elif p_orange >= 0.25 or (p_yellow >= 0.40 and avg_yellow_index > 150):
        stage = "Harvest-ready"
        conf = float(min(0.99, max(0.65, p_orange * 1.5 + p_yellow * 0.5)))
    elif p_yellow >= 0.35:
        stage = "Ripe"
        conf = float(min(0.95, max(0.60, p_yellow * 1.3 + p_orange * 0.4)))
    else:
        stage = "Mature"
        conf = float(min(0.90, max(0.55, p_yellow * 1.1 + (1.0 - p_green) * 0.5)))
    # Ekstraksi histogram HSV sebagai signature visual
    hist_h = cv2.calcHist([hsv], [0], valid_mask.astype(np.uint8), [16], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], valid_mask.astype(np.uint8), [16], [0, 256])
    hist = np.concatenate([hist_h.flatten(), hist_s.flatten()]).astype(np.float32)
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm

    return stage, conf, hist


class MangoDetector:
    """
    Wrapper Deteksi Mangga berbasis Ultralytics YOLO dengan integrasi
    analisis kematangan buah dan penyiapan ekspor ONNX untuk aplikasi ponsel.
    """
    def __init__(self, model_weight: str = "yolov8n.pt", conf_thresh: float = 0.40):
        self.conf_thresh = conf_thresh
        self.model_weight = model_weight
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_weight)
        except Exception as e:
            print(f"[Warning] Gagal memuat model YOLO ({e}), akan menggunakan detektor visual fallback.")
            self.model = None

    def detect(
        self,
        image: np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float] = None
    ) -> List[MangoDetection]:
        """
        Mendeteksi buah mangga dalam citra dan menentukan tingkat kematangannya.
        """
        threshold = conf if conf is not None else self.conf_thresh
        iou_threshold = iou if iou is not None else 0.45
        detections: List[MangoDetection] = []

        if self.model is not None:
            results = self.model.predict(
                image,
                conf=threshold,
                iou=iou_threshold,
                agnostic_nms=True,
                verbose=False
            )
            h_img, w_img = image.shape[:2]
            total_img_area = h_img * w_img

            for res in results:
                if res.boxes is None or len(res.boxes) == 0:
                    continue
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy()

                for box, score, cls_id in zip(boxes, confs, classes):
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)
                    bw = x2 - x1
                    bh = y2 - y1

                    if bw <= 10 or bh <= 10:
                        continue

                    # Filter Geometri Fisik: Eliminasi false positive (daun raksasa atau bintik noise)
                    area = bw * bh
                    aspect = float(bw) / float(bh)
                    if area > 0.12 * total_img_area or area < 350:
                        continue
                    if not (0.35 <= aspect <= 2.2):
                        continue

                    crop = image[y1:y2, x1:x2].copy()

                    # Analisis warna fisik (HSV & LAB) untuk verifikasi kematangan
                    color_stage, color_conf, color_hist = extract_fruit_color_signature(crop)

                    class_name_orig = self.model.names.get(int(cls_id), "mango").lower()
                    # Jika model multi-class yakin (>0.60), gunakan prediksi model; jika ragu, gunakan warna fisik
                    if float(score) >= 0.60 and any(m.lower() in class_name_orig for m in MATURITY_CLASSES.values()):
                        maturity_stage = class_name_orig.capitalize()
                        mat_conf = float(score)
                    else:
                        maturity_stage = color_stage
                        mat_conf = color_conf

                    det = MangoDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(score),
                        class_id=int(cls_id),
                        class_name="mango",
                        maturity_stage=maturity_stage,
                        maturity_confidence=mat_conf,
                        crop_image=crop,
                        color_hist=color_hist
                    )
                    detections.append(det)
            detections = self._post_process_clean_detections(detections, image)
        else:
            detections = self._heuristic_fruit_detect(image)

        return detections

    def _post_process_clean_detections(
        self,
        detections: List[MangoDetection],
        image: np.ndarray
    ) -> List[MangoDetection]:
        """
        Menyaring false positives spesifik:
        1. Rumput tanah pucat di latar belakang bawah.
        2. Ranting/pucuk daun di bagian atas tanpa buah.
        3. Kotak raksasa yang menelan (enclosing) dua buah atau lebih di dalamnya.
        """
        if not detections:
            return []
        h_img, w_img = image.shape[:2]
        valid = []

        for d in detections:
            crop = d.crop_image
            if crop is None or crop.size == 0:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            s_mean = float(hsv[:, :, 1].mean())
            v_mean = float(hsv[:, :, 2].mean())

            # 1. Filter rumput pucat tanah latar belakang
            if s_mean < 100 and v_mean > 165:
                continue

            # 2. Filter dedaunan pucuk atas tanpa buah (di atas tajuk y2 < 160)
            if d.bbox[3] < 160 and s_mean < 155:
                continue

            # 3. Filter dahan kayu mati / bayangan gelap
            if s_mean < 75:
                continue

            valid.append(d)

        # 4. Filter Enclosure / Kotak raksasa yang menelan 2+ buah di dalamnya
        n = len(valid)
        suppressed = set()
        for i in range(n):
            if i in suppressed:
                continue
            b_i = valid[i].bbox
            area_i = (b_i[2] - b_i[0]) * (b_i[3] - b_i[1])
            contained = 0
            for j in range(n):
                if i == j:
                    continue
                b_j = valid[j].bbox
                cj_x = (b_j[0] + b_j[2]) / 2.0
                cj_y = (b_j[1] + b_j[3]) / 2.0
                if b_i[0] <= cj_x <= b_i[2] and b_i[1] <= cj_y <= b_i[3]:
                    area_j = (b_j[2] - b_j[0]) * (b_j[3] - b_j[1])
                    if area_j < 0.65 * area_i:
                        contained += 1
            if contained >= 2:
                suppressed.add(i)

        return [valid[i] for i in range(n) if i not in suppressed]

    def _heuristic_fruit_detect(self, image: np.ndarray) -> List[MangoDetection]:
        """
        Detektor cadangan berbasis morfologi elips dan kontur warna buah
        jika file model belum terunduh / offline.
        """
        detections = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([15, 40, 40]), np.array([90, 255, 255]))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = (image.shape[0] * image.shape[1]) * 0.0005
        max_area = (image.shape[0] * image.shape[1]) * 0.15

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / float(h)
                if 0.5 <= aspect_ratio <= 1.8:
                    crop = image[y:y+h, x:x+w]
                    stage, conf, hist = extract_fruit_color_signature(crop)
                    det = MangoDetection(
                        bbox=(x, y, x + w, y + h),
                        confidence=0.85,
                        class_id=0,
                        class_name="mango",
                        maturity_stage=stage,
                        maturity_confidence=conf,
                        crop_image=crop,
                        color_hist=hist
                    )
                    detections.append(det)
        return detections

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[MangoDetection],
        show_maturity: bool = True,
        show_track_id: bool = True
    ) -> np.ndarray:
        """
        Visualisasi bounding box profesional dengan warna sesuai tingkat kematangan.
        """
        annotated = image.copy()

        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            color = MATURITY_COLORS.get(det.maturity_stage, (0, 255, 0))

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label teks
            if det.track_id is not None and show_track_id:
                label = f"ID:{det.track_id:03d} | {det.maturity_stage}"
            else:
                label = f"#{idx+1} {det.maturity_stage} ({det.confidence:.2f})"

            # Background label teks
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            badge_y1 = max(0, y1 - text_h - 8)
            badge_y2 = y1
            cv2.rectangle(annotated, (x1, badge_y1), (x1 + text_w + 10, badge_y2), color, -1)
            cv2.putText(
                annotated, label, (x1 + 5, y1 - 4),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
            )

        return annotated

    def export_onnx(self, output_path: Optional[str] = None) -> Optional[str]:
        """
        Ekspor model ke ONNX untuk integrasi aplikasi Android / ponsel sesuai Bab 11 panduan UTS.
        """
        if self.model is None:
            print("[Export] Model YOLO belum siap.")
            return None
        try:
            print("[Export] Mengekspor model ke ONNX...")
            exported = self.model.export(format="onnx")
            if output_path and os.path.exists(str(exported)):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                import shutil
                shutil.copy(str(exported), output_path)
                return output_path
            return str(exported)
        except Exception as e:
            print(f"[Export Error] {e}")
            return None


def summarize_maturity_counts(detections: List[MangoDetection]) -> Dict[str, int]:
    """Menghitung ringkasan jumlah buah per kategori kematangan."""
    summary = {
        "Unripe": 0,
        "Mature": 0,
        "Ripe": 0,
        "Harvest-ready": 0,
        "Total": len(detections)
    }
    for d in detections:
        stage = d.maturity_stage
        if stage in summary:
            summary[stage] += 1
        else:
            summary["Unripe"] += 1
    return summary
