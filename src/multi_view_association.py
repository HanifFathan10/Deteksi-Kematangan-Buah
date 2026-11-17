import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.detector import MangoDetection


@dataclass
class Tracklet:
    """Representasi lintasan sebuah buah dalam urutan frame video."""
    track_id: int
    detections: List[MangoDetection] = field(default_factory=list)
    start_frame: int = 0
    end_frame: int = 0
    avg_hist: Optional[np.ndarray] = None
    merged_into_id: Optional[int] = None

    def update_profile(self):
        """Memperbarui signature visual rata-rata dan rentang frame aktif."""
        if not self.detections:
            return
        hists = [d.color_hist for d in self.detections if d.color_hist is not None]
        if hists:
            mean_hist = np.mean(hists, axis=0)
            norm = np.linalg.norm(mean_hist)
            self.avg_hist = mean_hist / (norm + 1e-7)

    @property
    def total_hits(self) -> int:
        return len(self.detections)

    @property
    def dominant_maturity(self) -> str:
        """Menentukan tingkat kematangan dominan sepanjang lintasan."""
        stages = [d.maturity_stage for d in self.detections]
        if not stages:
            return "Unripe"
        from collections import Counter
        return Counter(stages).most_common(1)[0][0]

    @property
    def last_center(self) -> Tuple[float, float]:
        if self.detections:
            return self.detections[-1].center
        return (0.0, 0.0)

    @property
    def first_center(self) -> Tuple[float, float]:
        if self.detections:
            return self.detections[0].center
        return (0.0, 0.0)


class DisjointSetUnion:
    """Struktur data Disjoint Set (Union-Find) untuk pengelompokan buah lintas sudut/frame."""
    def __init__(self):
        self.parent = {}

    def find(self, i):
        if i not in self.parent:
            self.parent[i] = i
            return i
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j


class TrackletAssociator:
    """
    Mengatasi permasalahan ID-Switch pada video tracking (Bab 6.A & Bab 7):
    Menggabungkan kembali ID yang terputus ketika buah tertutup dedaunan
    menggunakan Hungarian Algorithm dan kesamaan visual (Cosine Distance).
    """
    def __init__(
        self,
        max_frame_gap: int = 45,
        max_pixel_dist: float = 250.0,
        appearance_thresh: float = 0.28,
        min_hits: int = 3
    ):
        self.max_frame_gap = max_frame_gap
        self.max_pixel_dist = max_pixel_dist
        self.appearance_thresh = appearance_thresh  # Ambang batas jarak kosinus (0 = identik)
        self.min_hits = min_hits

    def associate_tracklets(self, tracklets: List[Tracklet]) -> Tuple[List[Tracklet], Dict[int, int]]:
        """
        Menghubungkan tracklet yang terputus ke ID awal.
        Mengembalikan list tracklet terkonfirmasi dan peta penggabungan ID {old_id: new_id}.
        """
        # Filter tracklet pendek (spurious/flicker)
        valid_tracklets = [t for t in tracklets if t.total_hits >= self.min_hits]
        for t in valid_tracklets:
            t.update_profile()

        # Urutkan berdasarkan waktu mulai muncul
        valid_tracklets.sort(key=lambda x: x.start_frame)
        id_map: Dict[int, int] = {t.track_id: t.track_id for t in valid_tracklets}

        dsu = DisjointSetUnion()
        for t in valid_tracklets:
            dsu.find(t.track_id)

        n = len(valid_tracklets)
        for i in range(n):
            for j in range(i + 1, n):
                t1 = valid_tracklets[i]
                t2 = valid_tracklets[j]

                # t2 harus muncul setelah t1 selesai (atau sedikit overlap)
                gap = t2.start_frame - t1.end_frame
                if not (-5 <= gap <= self.max_frame_gap):
                    continue

                # Cek jarak spasial posisi akhir t1 dengan posisi awal t2
                p1 = t1.last_center
                p2 = t2.first_center
                spatial_dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                if spatial_dist > self.max_pixel_dist:
                    continue

                # Cek kemiripan fitur visual histogram (Cosine distance)
                if t1.avg_hist is not None and t2.avg_hist is not None:
                    cosine_sim = float(np.dot(t1.avg_hist, t2.avg_hist))
                    appearance_dist = 1.0 - max(0.0, min(1.0, cosine_sim))
                else:
                    appearance_dist = 0.5

                if appearance_dist <= self.appearance_thresh:
                    # Gabungkan identitas t2 ke t1
                    dsu.union(t1.track_id, t2.track_id)

        # Buat pemetaan ID hasil penggabungan
        for t in valid_tracklets:
            root_id = dsu.find(t.track_id)
            id_map[t.track_id] = root_id
            t.merged_into_id = root_id

        # Konsolidasikan tracklet unik
        consolidated: Dict[int, Tracklet] = {}
        for t in valid_tracklets:
            final_id = id_map[t.track_id]
            if final_id not in consolidated:
                consolidated[final_id] = Tracklet(
                    track_id=final_id,
                    start_frame=t.start_frame,
                    end_frame=t.end_frame,
                    detections=list(t.detections)
                )
            else:
                consolidated[final_id].detections.extend(t.detections)
                consolidated[final_id].start_frame = min(consolidated[final_id].start_frame, t.start_frame)
                consolidated[final_id].end_frame = max(consolidated[final_id].end_frame, t.end_frame)

        for c in consolidated.values():
            c.update_profile()

        return list(consolidated.values()), id_map


@dataclass
class MultiViewFruitCluster:
    """Cluster satu buah unik yang terlihat pada berbagai sudut foto."""
    fruit_id: int
    views: Dict[str, MangoDetection] = field(default_factory=dict)
    dominant_maturity: str = "Unripe"
    representative_image: Optional[np.ndarray] = None


class MultiViewPhotoAssociator:
    """
    Pencocokan Buah Antar-Sudut Pengambilan Citra (0°, 45°, 90°, ..., 315°)
    berbasis SIFT Feature Matching dan Hungarian Algorithm (Bab 6.B & Bab 7).
    """
    def __init__(self, min_inliers: int = 4, match_ratio: float = 0.75):
        self.min_inliers = min_inliers
        self.match_ratio = match_ratio
        try:
            self.sift = cv2.SIFT_create()
        except Exception:
            self.sift = cv2.ORB_create(500)

    def extract_crop_features(self, crop: np.ndarray) -> Tuple[Any, Optional[np.ndarray]]:
        """Ekstraksi keypoints dan deskriptor SIFT dari potongan buah."""
        if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return [], None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        kp, desc = self.sift.detectAndCompute(gray, None)
        return kp, desc

    def match_detections(
        self,
        det_a: MangoDetection,
        det_b: MangoDetection
    ) -> float:
        """
        Menghitung skor kesamaan visual antara dua deteksi mangga.
        Nilai mendekati 0 = sangat mirip / buah yang sama, nilai besar = buah berbeda.
        """
        if det_a.crop_image is None or det_b.crop_image is None:
            return 999.0

        kp_a, desc_a = self.extract_crop_features(det_a.crop_image)
        kp_b, desc_b = self.extract_crop_features(det_b.crop_image)

        # 1. Histogram Appearance Distance
        if det_a.color_hist is not None and det_b.color_hist is not None:
            norm_a = float(np.linalg.norm(det_a.color_hist))
            norm_b = float(np.linalg.norm(det_b.color_hist))
            if norm_a > 1e-6 and norm_b > 1e-6:
                cosine_sim = float(np.dot(det_a.color_hist, det_b.color_hist) / (norm_a * norm_b))
                color_dist = 1.0 - max(0.0, min(1.0, cosine_sim))
            else:
                color_dist = 0.5
        else:
            color_dist = 0.5

        # Relative area / size consistency
        area_a = det_a.area
        area_b = det_b.area
        size_ratio = min(area_a, area_b) / (max(area_a, area_b) + 1e-6)
        size_penalty = 1.0 - size_ratio

        # 2. SIFT Inlier Count
        if desc_a is not None and desc_b is not None and len(desc_a) >= self.min_inliers and len(desc_b) >= self.min_inliers:
            bf = cv2.BFMatcher(cv2.NORM_L2 if isinstance(self.sift, cv2.SIFT) else cv2.NORM_HAMMING)
            matches = bf.knnMatch(desc_a, desc_b, k=2)
            good_matches = []
            for m_pair in matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < self.match_ratio * n.distance:
                        good_matches.append(m)

            if len(good_matches) >= self.min_inliers:
                sift_score = max(0.0, 1.0 - (len(good_matches) / 20.0))
            else:
                sift_score = 0.8
            total_cost = 0.5 * sift_score + 0.35 * color_dist + 0.15 * size_penalty
        else:
            # Fallback ketika tekstur mangga terlalu halus untuk SIFT
            total_cost = 0.7 * color_dist + 0.3 * size_penalty

        return float(total_cost)

    def associate_views(
        self,
        view_detections: Dict[str, List[MangoDetection]],
        cost_threshold: float = 0.55
    ) -> List[MultiViewFruitCluster]:
        """
        Mengasosiasikan deteksi dari berbagai sudut foto (0°, 45°, 90°, ..., 315°)
        secara melingkar (circular wrap-around) dan mengeliminasi penghitungan ganda.
        """
        view_names = list(view_detections.keys())
        num_views = len(view_names)
        dsu = DisjointSetUnion()

        # Buat identifier unik untuk setiap deteksi: "nama_view::indeks_deteksi"
        all_items = []
        for v in view_names:
            for idx, det in enumerate(view_detections[v]):
                token = f"{v}::{idx}"
                all_items.append((token, v, det))
                dsu.find(token)

        if num_views >= 2:
            for i in range(num_views):
                # Bandingkan dengan 2 sudut berikutnya secara melingkar (termasuk 315° -> 0° & 45°)
                for step in [1, 2]:
                    if num_views == 2 and step == 2:
                        continue
                    j = (i + step) % num_views
                    if i == j:
                        continue

                    v1, v2 = view_names[i], view_names[j]
                    dets1 = view_detections[v1]
                    dets2 = view_detections[v2]

                    if not dets1 or not dets2:
                        continue

                    # Bangun Cost Matrix
                    cost_matrix = np.full((len(dets1), len(dets2)), 999.0)
                    for idx1, d1 in enumerate(dets1):
                        for idx2, d2 in enumerate(dets2):
                            cost_matrix[idx1, idx2] = self.match_detections(d1, d2)

                    # Jalankan Hungarian Algorithm untuk optimal bipartite matching
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    for r, c in zip(row_ind, col_ind):
                        if cost_matrix[r, c] <= cost_threshold:
                            tok1 = f"{v1}::{r}"
                            tok2 = f"{v2}::{c}"
                            root1 = dsu.find(tok1)
                            root2 = dsu.find(tok2)
                            if root1 != root2:
                                # Cegah penggabungan 2 buah berbeda dari sudut foto yang sama
                                views_in_1 = {
                                    tok.split("::")[0]
                                    for tok, _, _ in all_items
                                    if dsu.find(tok) == root1
                                }
                                views_in_2 = {
                                    tok.split("::")[0]
                                    for tok, _, _ in all_items
                                    if dsu.find(tok) == root2
                                }
                                if views_in_1.isdisjoint(views_in_2):
                                    dsu.union(tok1, tok2)

        # Kelompokkan hasil clustering ke Disjoint Set
        clusters_map: Dict[str, Dict[str, MangoDetection]] = {}
        for token, v, det in all_items:
            root = dsu.find(token)
            if root not in clusters_map:
                clusters_map[root] = {}
            clusters_map[root][v] = det

        # Bangun list MultiViewFruitCluster
        results: List[MultiViewFruitCluster] = []
        for fruit_idx, (root, views) in enumerate(clusters_map.items(), start=1):
            stages = [d.maturity_stage for d in views.values()]
            from collections import Counter
            dom_stage = Counter(stages).most_common(1)[0][0]

            best_det = max(views.values(), key=lambda x: x.confidence)
            rep_img = best_det.crop_image

            cluster = MultiViewFruitCluster(
                fruit_id=fruit_idx,
                views=views,
                dominant_maturity=dom_stage,
                representative_image=rep_img
            )
            results.append(cluster)

        return results


def track_and_count_video(
    video_path: str,
    detector,
    conf_thresh: float = 0.20,
    frame_step: int = 2,
    max_dist: float = 90.0,
    frame_callback=None
) -> Dict[str, Any]:
    """
    Pipeline Video Tracking lengkap sesuai Bab 6.A & Bab 7:
    1. Ekstraksi frame video ponsel secara berkala.
    2. Deteksi mangga per frame menggunakan YOLO.
    3. Tracking antar-frame (pemberian tracking ID).
    4. Tracklet Association menggunakan Hungarian Algorithm untuk rekoneksi ID-switch.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "success": False,
            "error": f"Gagal membuka file video: {video_path}"
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frame_idx = 0
    active_tracks: Dict[int, Tracklet] = {}
    next_id = 1
    all_finished_tracklets: List[Tracklet] = []
    sample_frames: List[np.ndarray] = []
    processed_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_step != 0:
            continue

        processed_count += 1
        dets = detector.detect(frame, conf=conf_thresh)

        # Bipartite matching sederhana posisi spasial deteksi baru ke active tracks
        matched_dets = set()
        matched_tracks = set()

        for tid, trk in list(active_tracks.items()):
            last_det = trk.detections[-1]
            best_dist = 999.0
            best_d_idx = -1

            for d_idx, d in enumerate(dets):
                if d_idx in matched_dets:
                    continue
                dist = math.hypot(d.center[0] - last_det.center[0], d.center[1] - last_det.center[1])
                if dist < max_dist and dist < best_dist:
                    best_dist = dist
                    best_d_idx = d_idx

            if best_d_idx >= 0:
                d = dets[best_d_idx]
                d.track_id = tid
                trk.detections.append(d)
                trk.end_frame = frame_idx
                matched_dets.add(best_d_idx)
                matched_tracks.add(tid)
            else:
                # Track hilang / terputus di frame ini
                all_finished_tracklets.append(trk)
                del active_tracks[tid]

        # Inisialisasi track baru untuk deteksi yang belum cocok
        for d_idx, d in enumerate(dets):
            if d_idx not in matched_dets:
                d.track_id = next_id
                trk = Tracklet(track_id=next_id, detections=[d], start_frame=frame_idx, end_frame=frame_idx)
                active_tracks[next_id] = trk
                next_id += 1

        # Visualisasikan frame
        annotated_frame = detector.draw_detections(frame, dets, show_track_id=True)
        if frame_callback is not None:
            frame_callback(processed_count, total_frames // frame_step, annotated_frame)

        if len(sample_frames) < 12 and (processed_count % 3 == 0):
            sample_frames.append(annotated_frame)

    all_finished_tracklets.extend(active_tracks.values())
    cap.release()

    raw_track_count = len(all_finished_tracklets)

    # Terapkan Tracklet Association untuk menggabungkan kembali ID-switch
    associator = TrackletAssociator(max_frame_gap=35, max_pixel_dist=180, min_hits=2)
    merged_tracklets, id_map = associator.associate_tracklets(all_finished_tracklets)

    # Hitung distribusi kematangan buah unik
    maturity_summary = {"Unripe": 0, "Mature": 0, "Ripe": 0, "Harvest-ready": 0}
    for t in merged_tracklets:
        stg = t.dominant_maturity
        if stg in maturity_summary:
            maturity_summary[stg] += 1
        else:
            maturity_summary["Unripe"] += 1

    return {
        "success": True,
        "total_frames_in_video": total_frames,
        "processed_frames_count": processed_count,
        "raw_track_count": raw_track_count,
        "unique_fruit_count": len(merged_tracklets),
        "id_map": id_map,
        "merged_tracklets": merged_tracklets,
        "maturity_distribution": maturity_summary,
        "sample_frames": sample_frames
    }
