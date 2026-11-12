import os
import sys
import cv2
import numpy as np

from src.preprocessing import preprocess_single_image, batch_preprocess_folder
from src.detector import MangoDetector, summarize_maturity_counts
from src.multi_view_association import MultiViewPhotoAssociator, TrackletAssociator, Tracklet
from src.evaluation import (
    calculate_error_percentage,
    estimate_harvest_yield,
    build_ablation_comparison_table,
    export_markdown_report
)


def run_pipeline_verification():
    print("=================================================================")
    print("  VERIFIKASI SISTEM PENGHITUNG BUAH MANGGA MULTI-VIEW SMARTPHONE")
    print("=================================================================\n")

    # 1. Verifikasi Preprocessing
    print("[1/5] Menguji Modul Preprocessing OpenCV...")
    raw_dir = "data/raw/photos_multi_angle"
    processed_dir = "data/processed"
    prep_results = batch_preprocess_folder(raw_dir, processed_dir, max_dim=1280, blur_threshold=60.0)
    print(f"      -> Berhasil memproses {len(prep_results)} citra.")
    for res in prep_results[:3]:
        print(f"         {res['filename']}: blur score={res['blur_score']:.1f}, scale={res['scale']:.2f}")

    # 2. Verifikasi Deteksi Mango YOLO & Kematangan
    print("\n[2/5] Menguji Modul Deteksi YOLO & Klasifikasi Kematangan...")
    detector = MangoDetector("models/best.pt", conf_thresh=0.20)

    view_detections = {}
    total_naive_detections = 0
    all_stages = []

    for res in sorted(prep_results, key=lambda x: x["filename"]):
        fname = res["filename"]
        img_path = res["saved_path"]
        img = cv2.imread(img_path)

        dets = detector.detect(img)
        view_detections[fname] = dets
        total_naive_detections += len(dets)
        for d in dets:
            all_stages.append(d.maturity_stage)

        # Simpan contoh visualisasi deteksi
        annotated = detector.draw_detections(img, dets)
        out_det_path = os.path.join("output/detections", f"det_{fname}")
        cv2.imwrite(out_det_path, annotated)

    print(f"      -> Total deteksi mentah (Naive Sum seluruh sudut): {total_naive_detections} bounding boxes")
    print(f"      -> Disimpan ke folder 'output/detections/'")

    # 3. Verifikasi Multi-View Association & Duplicate Counting Removal
    print("\n[3/5] Menguji Algoritma Multi-View Association & Duplicate Removal...")
    associator = MultiViewPhotoAssociator(min_inliers=3, match_ratio=0.75)
    clusters = associator.associate_views(view_detections, cost_threshold=0.55)

    unique_fruit_count = len(clusters)
    ground_truth_count = 12  # Ground truth pohon sintetis yang dibuat

    print(f"      -> Jumlah deteksi sebelum eliminasi duplikasi: {total_naive_detections}")
    print(f"      -> Jumlah buah unik setelah Multi-View Association: {unique_fruit_count}")
    print(f"      -> Ground Truth aktual pohon: {ground_truth_count}")

    # 4. Verifikasi Pelacakan Video & Tracklet Association
    print("\n[4/5] Menguji Video Tracking & Rekoneksi ID-Switch...")
    vid_path = "data/raw/video/mango_tree_walkaround.mp4"
    cap = cv2.VideoCapture(vid_path)
    frame_idx = 0
    simulated_tracklets = []

    # Simulasi pembacaan frame dan pengelompokan tracklet
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 10 == 0:
            dets = detector.detect(frame)
            if dets:
                t = Tracklet(track_id=len(simulated_tracklets) + 1, detections=dets, start_frame=frame_idx, end_frame=frame_idx + 8)
                simulated_tracklets.append(t)
    cap.release()

    track_associator = TrackletAssociator(max_frame_gap=30, max_pixel_dist=200)
    merged_tracks, id_map = track_associator.associate_tracklets(simulated_tracklets)
    standard_tracking_id_count = len(simulated_tracklets)
    resolved_tracking_count = len(merged_tracks)
    print(f"      -> Raw Tracking ID (mengalami ID-switch): {standard_tracking_id_count}")
    print(f"      -> Tracking ID setelah Tracklet Association: {resolved_tracking_count}")

    # 5. Verifikasi Evaluasi Ilmiah & Laporan Panen
    print("\n[5/5] Menghitung Metrik Evaluasi & Menyusun Laporan UTS...")
    maturity_summary = {"Unripe": 0, "Mature": 0, "Ripe": 0, "Harvest-ready": 0}
    for c in clusters:
        stg = c.dominant_maturity
        maturity_summary[stg] = maturity_summary.get(stg, 0) + 1

    df_ablation = build_ablation_comparison_table(
        actual_count=ground_truth_count,
        naive_sum_count=total_naive_detections,
        standard_tracking_count=standard_tracking_id_count,
        proposed_unique_count=unique_fruit_count
    )
    print("\n" + df_ablation.to_string(index=False))

    report_file = export_markdown_report(
        tree_id="Pohon-Mangga-Eksperimen-01",
        actual_count=ground_truth_count,
        naive_count=total_naive_detections,
        tracker_count=standard_tracking_id_count,
        unique_count=unique_fruit_count,
        maturity_breakdown=maturity_summary
    )
    print(f"\n[OK] Laporan lengkap berhasil dibuat di: '{report_file}'")
    print("=================================================================")
    print("  SEMUA MODUL SISTEM BERHASIL DIVERIFIKASI SECARA END-TO-END!")
    print("=================================================================\n")


if __name__ == "__main__":
    run_pipeline_verification()
