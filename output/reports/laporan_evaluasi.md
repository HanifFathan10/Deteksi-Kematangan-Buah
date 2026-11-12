# Laporan Hasil Evaluasi Sistem Penghitung Buah Mangga
**Mata Kuliah:** Praktikum Informatika - Pengolahan Citra (Precision Agriculture)
**Objek Pohon:** Pohon-Mangga-Eksperimen-01

---

## 1. Ringkasan Evaluasi Akurasi Penghitungan
* **Hitungan Manual Lapangan (Ground Truth):** 12 buah
* **Total Deteksi Naive (Semua Sudut):** 43 deteksi
* **Hasil Standard Video Tracker:** 6 tracking ID
* **Hasil Akhir Sistem Kami (Unique Fruit Count):** **13 buah**
* **Tingkat Kesalahan (Error Rate):** **8.33%**

---

## 2. Tabel Komparasi Metode (Ablation Study)

| Metode | Hitungan Sistem | Ground Truth | Selisih | Error (%) | Evaluasi Ilmiah |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Metode 1: Naive Bounding Box Sum | 43 | 12 | 31 | 258.33% | Overcount parah karena 1 buah dihitung berkali-kali dari sudut berbeda. |
| Metode 2: Standard ByteTrack (Raw IDs) | 6 | 12 | -6 | 50.00% | Terjadi overcount sedang akibat oklusi daun memicu pergantian ID (ID-switch). |
| Metode 3: Proposed Multi-View Association (Sistem Ini) | 13 | 12 | 1 | 8.33% | Duplikasi tereliminasi optimal dengan pencocokan visual SIFT & Hungarian Algorithm. |

---

## 3. Estimasi Hasil Panen (Yield & Maturity Breakdown)
* **Buah Siap Panen (Harvest-Ready):** 4 butir (30.8%)
* **Estimasi Bobot Siap Panen:** 1.4 kg (asumsi 350 gram/butir)
* **Total Potensi Bobot Seluruh Pohon:** 4.55 kg

### Distribusi Kematangan Buah:
* **Unripe (Belum Matang):** 6 butir
* **Mature (Matang Fisiologis):** 1 butir
* **Ripe (Matang Lanjut):** 5 butir
* **Harvest-ready (Siap Panen):** 1 butir

---
*Laporan ini dihasilkan secara otomatis oleh Sistem Estimasi Buah Mangga Smartphone.*
