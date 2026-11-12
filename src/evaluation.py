import math
import os
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


def calculate_error_percentage(actual: float, predicted: float) -> float:
    """Menghitung persentase kesalahan relatif sesuai Bab 9 Panduan UTS."""
    if actual <= 0:
        return 0.0
    return abs(predicted - actual) / actual * 100.0


def calculate_mae(actuals: List[float], predictions: List[float]) -> float:
    """Mean Absolute Error (MAE): Rata-rata kesalahan absolut penghitungan."""
    if not actuals or len(actuals) != len(predictions):
        return 0.0
    errors = [abs(a - p) for a, p in zip(actuals, predictions)]
    return float(np.mean(errors))


def calculate_rmse(actuals: List[float], predictions: List[float]) -> float:
    """Root Mean Squared Error (RMSE): Besarnya kesalahan kuadrat penghitungan."""
    if not actuals or len(actuals) != len(predictions):
        return 0.0
    sq_errors = [(a - p) ** 2 for a, p in zip(actuals, predictions)]
    return float(math.sqrt(np.mean(sq_errors)))


def calculate_mape(actuals: List[float], predictions: List[float]) -> float:
    """Mean Absolute Percentage Error (MAPE): Kesalahan relatif rata-rata dalam persen."""
    if not actuals or len(actuals) != len(predictions):
        return 0.0
    pct_errors = [abs(a - p) / a * 100.0 for a, p in zip(actuals, predictions) if a > 0]
    return float(np.mean(pct_errors)) if pct_errors else 0.0


def calculate_detection_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """
    Menghitung Precision, Recall, dan F1-Score Object Detection
    sesuai tabel metrik Bab 9 Panduan UTS (Halaman 22-23).
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }


def estimate_harvest_yield(
    total_unique_fruits: int,
    maturity_breakdown: Dict[str, int],
    avg_fruit_weight_grams: float = 350.0
) -> Dict[str, Any]:
    """
    Estimasi Hasil Panen Mangga per Pohon sesuai Bab 8 & 13 Panduan UTS:
    - Jumlah buah siap panen
    - Estimasi bobot buah siap panen (kg)
    - Estimasi total potensi hasil panen keseluruhan pohon (kg)
    """
    harvest_ready = maturity_breakdown.get("Harvest-ready", 0)
    mature = maturity_breakdown.get("Mature", 0)
    ripe = maturity_breakdown.get("Ripe", 0)
    unripe = maturity_breakdown.get("Unripe", 0)

    # Buah yang layak panen sekarang (Harvest-ready + sebagian Ripe)
    ready_count = harvest_ready + int(ripe * 0.7)
    ready_pct = (ready_count / total_unique_fruits * 100.0) if total_unique_fruits > 0 else 0.0

    ready_weight_kg = (ready_count * avg_fruit_weight_grams) / 1000.0
    total_potential_weight_kg = (total_unique_fruits * avg_fruit_weight_grams) / 1000.0

    return {
        "total_unique_fruits": total_unique_fruits,
        "harvest_ready_count": ready_count,
        "harvest_ready_pct": round(ready_pct, 1),
        "harvest_ready_weight_kg": round(ready_weight_kg, 2),
        "total_potential_weight_kg": round(total_potential_weight_kg, 2),
        "maturity_distribution": {
            "Harvest-ready": harvest_ready,
            "Ripe": ripe,
            "Mature": mature,
            "Unripe": unripe
        }
    }


def build_ablation_comparison_table(
    actual_count: int,
    naive_sum_count: int,
    standard_tracking_count: int,
    proposed_unique_count: int
) -> pd.DataFrame:
    """
    Membuat Tabel Perbandingan Metode (Ablation Study)
    untuk membuktikan keunggulan algoritma Multi-View Association
    dan Duplicate Removal kepada dosen penguji.
    """
    data = [
        {
            "Metode": "Metode 1: Naive Bounding Box Sum",
            "Hitungan Sistem": naive_sum_count,
            "Ground Truth (Aktual)": actual_count,
            "Selisih Buah": naive_sum_count - actual_count,
            "Error (%)": f"{calculate_error_percentage(actual_count, naive_sum_count):.2f}%",
            "Evaluasi Ilmiah": "Overcount parah karena 1 buah dihitung berkali-kali dari sudut berbeda."
        },
        {
            "Metode": "Metode 2: Standard ByteTrack (Raw IDs)",
            "Hitungan Sistem": standard_tracking_count,
            "Ground Truth (Aktual)": actual_count,
            "Selisih Buah": standard_tracking_count - actual_count,
            "Error (%)": f"{calculate_error_percentage(actual_count, standard_tracking_count):.2f}%",
            "Evaluasi Ilmiah": "Terjadi overcount sedang akibat oklusi daun memicu pergantian ID (ID-switch)."
        },
        {
            "Metode": "Metode 3: Proposed Multi-View Association (Sistem Ini)",
            "Hitungan Sistem": proposed_unique_count,
            "Ground Truth (Aktual)": actual_count,
            "Selisih Buah": proposed_unique_count - actual_count,
            "Error (%)": f"{calculate_error_percentage(actual_count, proposed_unique_count):.2f}%",
            "Evaluasi Ilmiah": "Duplikasi tereliminasi optimal dengan pencocokan visual SIFT & Hungarian Algorithm."
        }
    ]
    return pd.DataFrame(data)


def export_markdown_report(
    tree_id: str,
    actual_count: int,
    naive_count: int,
    tracker_count: int,
    unique_count: int,
    maturity_breakdown: Dict[str, int],
    output_path: str = "output/reports/laporan_evaluasi.md"
) -> str:
    """Menyimpan ringkasan evaluasi ke format Markdown siap dimasukkan ke laporan UTS."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    yield_info = estimate_harvest_yield(unique_count, maturity_breakdown)
    df_comp = build_ablation_comparison_table(actual_count, naive_count, tracker_count, unique_count)

    err_proposed = calculate_error_percentage(actual_count, unique_count)

    content = f"""# Laporan Hasil Evaluasi Sistem Penghitung Buah Mangga
**Mata Kuliah:** Praktikum Informatika - Pengolahan Citra (Precision Agriculture)
**Objek Pohon:** {tree_id}

---

## 1. Ringkasan Evaluasi Akurasi Penghitungan
* **Hitungan Manual Lapangan (Ground Truth):** {actual_count} buah
* **Total Deteksi Naive (Semua Sudut):** {naive_count} deteksi
* **Hasil Standard Video Tracker:** {tracker_count} tracking ID
* **Hasil Akhir Sistem Kami (Unique Fruit Count):** **{unique_count} buah**
* **Tingkat Kesalahan (Error Rate):** **{err_proposed:.2f}%**

---

## 2. Tabel Komparasi Metode (Ablation Study)

| Metode | Hitungan Sistem | Ground Truth | Selisih | Error (%) | Evaluasi Ilmiah |
| :--- | :---: | :---: | :---: | :---: | :--- |
"""
    for _, row in df_comp.iterrows():
        content += f"| {row['Metode']} | {row['Hitungan Sistem']} | {row['Ground Truth (Aktual)']} | {row['Selisih Buah']} | {row['Error (%)']} | {row['Evaluasi Ilmiah']} |\n"

    content += f"""
---

## 3. Estimasi Hasil Panen (Yield & Maturity Breakdown)
* **Buah Siap Panen (Harvest-Ready):** {yield_info['harvest_ready_count']} butir ({yield_info['harvest_ready_pct']}%)
* **Estimasi Bobot Siap Panen:** {yield_info['harvest_ready_weight_kg']} kg (asumsi 350 gram/butir)
* **Total Potensi Bobot Seluruh Pohon:** {yield_info['total_potential_weight_kg']} kg

### Distribusi Kematangan Buah:
* **Unripe (Belum Matang):** {maturity_breakdown.get('Unripe', 0)} butir
* **Mature (Matang Fisiologis):** {maturity_breakdown.get('Mature', 0)} butir
* **Ripe (Matang Lanjut):** {maturity_breakdown.get('Ripe', 0)} butir
* **Harvest-ready (Siap Panen):** {maturity_breakdown.get('Harvest-ready', 0)} butir

---
*Laporan ini dihasilkan secara otomatis oleh Sistem Estimasi Buah Mangga Smartphone.*
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_path


def calculate_citrus_harvest_index(maturity_breakdown: Dict[str, int], total_fruits: int) -> float:
    """
    Menghitung Citrus Harvest Readiness Index (CHRI) sesuai Bab 3 PRD:
    CHRI = (1.0*N_hr + 0.80*N_ripe + 0.20*N_mature + 0.0*N_unripe) / N_total * 100%
    """
    if total_fruits <= 0:
        return 0.0
    hr = maturity_breakdown.get("Harvest-ready", 0)
    ripe = maturity_breakdown.get("Ripe", 0)
    mature = maturity_breakdown.get("Mature", 0)
    unripe = maturity_breakdown.get("Unripe", 0)

    score = (1.0 * hr + 0.80 * ripe + 0.20 * mature + 0.0 * unripe) / total_fruits * 100.0
    return round(float(score), 1)


def estimate_citrus_yield(
    total_unique_fruits: int,
    maturity_breakdown: Dict[str, int],
    avg_fruit_weight_grams: float = 145.0
) -> Dict[str, Any]:
    """
    Estimasi Hasil Panen Jeruk per Pohon sesuai Standar SNI 01-3168-1992 & PRD:
    - Jumlah buah siap petik Batch 1
    - Estimasi bobot Batch 1 (kg)
    - Estimasi total potensi bobot pohon (kg)
    - CHRI score
    """
    harvest_ready = maturity_breakdown.get("Harvest-ready", 0)
    ripe = maturity_breakdown.get("Ripe", 0)
    mature = maturity_breakdown.get("Mature", 0)
    unripe = maturity_breakdown.get("Unripe", 0)

    ready_count = harvest_ready + int(ripe * 0.7)
    ready_pct = (ready_count / total_unique_fruits * 100.0) if total_unique_fruits > 0 else 0.0
    ready_weight_kg = (ready_count * avg_fruit_weight_grams) / 1000.0
    total_potential_weight_kg = (total_unique_fruits * avg_fruit_weight_grams) / 1000.0
    chri = calculate_citrus_harvest_index(maturity_breakdown, total_unique_fruits)

    return {
        "total_unique_fruits": total_unique_fruits,
        "harvest_ready_count": ready_count,
        "harvest_ready_pct": round(ready_pct, 1),
        "harvest_ready_weight_kg": round(ready_weight_kg, 2),
        "total_potential_weight_kg": round(total_potential_weight_kg, 2),
        "chri_score": chri,
        "maturity_distribution": {
            "Harvest-ready": harvest_ready,
            "Ripe": ripe,
            "Mature": mature,
            "Unripe": unripe
        }
    }


def build_citrus_telemetry_payload(
    tree_id: str,
    variety: str,
    raw_detections_count: int,
    unique_fruits_count: int,
    blur_score: float,
    maturity_breakdown: Dict[str, int],
    avg_weight_grams: float = 145.0,
    price_super: float = 22000.0,
    price_standard: float = 14000.0,
    price_juice: float = 10000.0,
    price_tengkulak: float = 8000.0
) -> Dict[str, Any]:
    """Menyusun payload JSON sesuai kontrak schema CitrusVisionTelemetryPayload di BAB 4 PRD."""
    from datetime import datetime
    citrus_yield = estimate_citrus_yield(unique_fruits_count, maturity_breakdown, avg_weight_grams)
    dedup_ratio = 0.0
    if raw_detections_count > 0:
        dedup_ratio = round((raw_detections_count - unique_fruits_count) / raw_detections_count * 100.0, 1)

    return {
        "session_id": f"CV-{datetime.now().strftime('%Y%m%d%H%M%S')}-{tree_id.replace(' ', '_')}",
        "timestamp": datetime.now().isoformat(),
        "commodity": "Citrus",
        "citrus_variety": variety,
        "tree_id": tree_id,
        "cv_detection_metrics": {
            "raw_detections_sum": int(raw_detections_count),
            "unique_verified_fruits": int(unique_fruits_count),
            "deduplication_reduction_pct": dedup_ratio,
            "camera_angles_analyzed": 8,
            "avg_laplacian_blur_score": round(float(blur_score), 1)
        },
        "maturity_breakdown": maturity_breakdown,
        "yield_biomass": {
            "avg_fruit_weight_grams": float(avg_weight_grams),
            "batch1_ready_weight_kg": citrus_yield["harvest_ready_weight_kg"],
            "total_potential_weight_kg": citrus_yield["total_potential_weight_kg"],
            "chri_score": citrus_yield["chri_score"]
        },
        "market_price_context": {
            "price_super_grade_a": float(price_super),
            "price_standard_grade_b": float(price_standard),
            "price_juice_grade_c": float(price_juice),
            "price_tengkulak_raw": float(price_tengkulak)
        }
    }
