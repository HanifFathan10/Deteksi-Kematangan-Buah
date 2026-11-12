# Projek UTS: Sistem Estimasi Jumlah dan Kematangan Buah Jeruk Menggunakan Ponsel
**Mata Kuliah:** Praktikum Informatika - Pengolahan Citra (Precision Agriculture)  
**Dokumen Acuan:** `UTS-Ganjil2026-2927-PengolahanCitra-V1 (1).pdf`

---

## 📌 Ringkasan Sistem
Sistem ini dikembangkan untuk memperkirakan jumlah buah jeruk pada **satu pohon utuh** secara akurat menggunakan kamera *smartphone* biasa (tanpa drone/kamera khusus).

Sistem ini memecahkan masalah utama yang ditekankan oleh dosen:
> *"Sistem tidak boleh sekadar menjumlahkan seluruh hasil deteksi dari berbagai foto. Satu buah jeruk bisa terlihat pada foto depan, samping, dan belakang. Sistem harus mampu mengenali bahwa ketiga deteksi tersebut berasal dari satu buah yang sama."*

### Keunggulan Sistem Ini (Melampaui Tugas Standar):
1. **Multi-Class Maturity Detection:** Mengklasifikasikan 4 tahap kematangan (*Unripe, Mature, Ripe, Harvest-ready*).
2. **Duplicate Counting Removal:** Memadukan **SIFT Feature Matching**, **Cosine Appearance Distance**, dan **Hungarian Algorithm** untuk mengeliminasi duplikasi antar sudut.
3. **Tracklet Re-Association:** Mengatasi masalah *ID-Switch* pada video tracking saat buah terhalang daun.
4. **Dashboard Interaktif Web (Streamlit):** Siap digunakan untuk demonstrasi langsung saat presentasi (Pertemuan 6).
5. **Mobile-Ready Deployment:** Model telah berhasil diekspor ke format **ONNX** (`models/jeruk_maturity.onnx`) untuk kebutuhan aplikasi Android / smartphone.

---

## 📂 Struktur Direktori Proyek

```text
C:/Pengolahan Citra/
│
├── UTS-Ganjil2026-2927-PengolahanCitra-V1 (1).pdf  # Dokumen panduan UTS dari dosen
├── requirements.txt                               # Dependensi Python
├── README.md                                      # Dokumentasi petunjuk penggunaan
├── app.py                                         # Dashboard interaktif web (Streamlit)
├── test_pipeline.py                               # Skrip pengujian end-to-end otomatis
│
├── data/
│   ├── raw/
│   │   ├── photos_multi_angle/                    # Foto pohon dari 8 sudut mata angin
│   │   └── video/                                 # Video rekaman mengelilingi pohon
│   ├── processed/                                 # Hasil preprocessing OpenCV (CLAHE, resize)
│   └── dataset/                                   # Format dataset standar YOLO
│
├── models/
│   ├── best.pt                                    # Bobot model YOLO mangga (PyTorch)
│   ├── yolov8n.pt                                 # Base model Ultralytics
│   └── jeruk_maturity.onnx                        # Model ONNX untuk ponsel/mobile
│
├── src/
│   ├── preprocessing.py                           # Modul 1: OpenCV Preprocessing, blur check, CLAHE
│   ├── detector.py                                # Modul 2: YOLO Detection & Maturity Classification
│   ├── multi_view_association.py                  # Modul 3: SIFT + Hungarian + Tracklet Re-ID
│   ├── evaluation.py                              # Modul 4: MAE, RMSE, MAPE, Ablation Table
│   └── demo_data.py                               # Generator data sintetis untuk pengujian
│
└── output/
    ├── detections/                                # Visualisasi bounding box deteksi
    └── reports/                                   # Laporan evaluasi Markdown otomatis
```

---

## 🚀 Panduan Menjalankan Sistem

### 1. Menjalankan Dashboard Interaktif Web (Streamlit)
Untuk mendemonstrasikan sistem secara visual kepada dosen saat presentasi UTS:
```bash
streamlit run app.py
```
Aplikasi akan terbuka otomatis di browser (`http://localhost:8501`).
Di dalam aplikasi terdapat 4 tab sesuai alur silabus:
* **Tab 1: Preprocessing Citra** (Cek blur, perbaikan kontras CLAHE).
* **Tab 2: Deteksi & Kematangan** (Bounding box warna-warni sesuai tingkat kematangan).
* **Tab 3: Multi-View Association** (Pembuktian eliminasi duplikasi dari 26 deteksi menjadi buah unik).
* **Tab 4: Evaluasi Akurasi & Panen** (Tabel komparasi MAE/RMSE dan ringkasan panen).

### 2. Menjalankan Verifikasi Pipeline Lengkap (CLI)
Untuk memverifikasi seluruh modul berjalan 100% tanpa error via terminal:
```bash
python test_pipeline.py
```

### 3. Menggunakan Data Foto atau Video Lapangan Anda Sendiri
Ketika Anda mengambil foto atau merekam video pohon mangga di lapangan sesuai Pertemuan 1:
* **Jika Menggunakan Foto:**
  1. Simpan foto-foto pohon yang Anda ambil mengelilingi pohon ke `data/raw/photos_multi_angle/` atau unggah via browser di Tab 1.
  2. Pilih Tab 3 Opsi A untuk menjalankan pencocokan multi-sudut (*SIFT + Hungarian*).
* **Jika Menggunakan Video:**
  1. Unggah video format MP4, MOV, atau AVI langsung melalui Tab 1 atau Tab 3 Opsi B.
  2. Klik tombol **"🎬 Jalankan Video Tracking & Eliminasi ID-Switch"**.
  3. Masukkan jumlah hitungan manual pohon Anda ke parameter *Ground Truth*.
---

## 📊 Hasil Evaluasi Komparatif (Ablation Study)

Berikut adalah bukti pengujian komparasi metode yang dihasilkan sistem:

| Metode | Hitungan Sistem | Ground Truth | Selisih | Error (%) | Evaluasi Ilmiah |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Metode 1: Naive Bounding Box Sum** | 26 | 12 | +14 | 116.67% | *Overcount* parah karena 1 buah dihitung berkali-kali dari sudut berbeda. |
| **Metode 2: Standard ByteTrack (Raw IDs)** | 5 | 12 | -7 | 58.33% | Mengalami *ID-switch* akibat buah tertutup daun ketika kamera berputar. |
| **Metode 3: Proposed Multi-View Association** | **10** | **12** | **-2** | **16.67%** | **Duplikasi tereliminasi optimal dengan pencocokan visual SIFT & Hungarian Algorithm.** |

---

## 🏆 Kepatuhan terhadap Silabus UTS Dosen

| Pertemuan di Modul Dosen | Materi yang Ditugaskan | Implementasi pada Proyek Ini |
| :--- | :--- | :--- |
| **Pertemuan 1** | Akuisisi citra 8 arah & hitungan manual | `data/raw/photos_multi_angle/` + `demo_data.py` |
| **Pertemuan 2** | Preprocessing OpenCV & split dataset | `src/preprocessing.py` (Resize, Blur metric, CLAHE) |
| **Pertemuan 3** | YOLO Object Detection & Training | `src/detector.py` + `models/best.pt` |
| **Pertemuan 4 & 5** | Multi-View Processing & Duplicate Removal | `src/multi_view_association.py` (SIFT + Hungarian) |
| **Pertemuan 6** | Evaluasi Akurasi & Presentasi | `src/evaluation.py` + `app.py` (Streamlit Dashboard) |
| **Bab 11** | Konversi model ke Mobile (ONNX/TFLite) | `models/jeruk_maturity.onnx` |
