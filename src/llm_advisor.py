"""
Modul AI Advisor Agribisnis Jeruk (CitrusVision)
Terintegrasi dengan DeepSeek v4.1 Flash via AIVO Gateway
Berdasarkan Dokumen Kebutuhan Produk (PRD) v2.1.0-PRODUCTION-READY
"""

import os
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Generator, Optional, Union
from dataclasses import dataclass

DEFAULT_AIVO_API_KEY = os.getenv(
    "AIVO_API_KEY",
    "sk-aivo_gooNMxZOjIUsQzfQiiTBQqtuZuHATPGAMErUsVWGeNQOhypldgcPmcpUrqOJjeKs"
)
DEFAULT_AIVO_BASE_URL = os.getenv("AIVO_BASE_URL", "https://api.aivokey.dev/v1")
DEFAULT_MODEL = os.getenv("AIVO_MODEL", "deepseek-v4.1-flash")


@dataclass
class CitrusAdvisorConfig:
    api_key: str = DEFAULT_AIVO_API_KEY
    base_url: str = DEFAULT_AIVO_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = 0.30
    max_tokens: int = 3500
    timeout: int = 60


def build_citrus_risen_prompt(telemetry: Dict[str, Any]) -> str:
    """
    Menyusun prompt berbasis framework RISEN yang komprehensif
    untuk DeepSeek v4.1 Flash sesuai spesifikasi BAB 3 & BAB 4 PRD.
    """
    telemetry_json = json.dumps(telemetry, indent=2, ensure_ascii=False)

    prompt = f"""Anda adalah Lead Agronomist & Enterprise Horticultural Specialist khusus agribisnis buah jeruk tropis (Citrus sinensis / Citrus reticulata). Anda menguasai fisiologi pasca-panen jeruk non-klimakterik, mitigasi penyakit oleocellosis dan jamur Penicillium, standardisasi mutu SNI 01-3168-1992, serta strategi negosiasi perdagangan jeruk di Indonesia.

Tugas Anda adalah menganalisis data telemetri citra pohon jeruk berikut dan menghasilkan Laporan Konsultasi Bisnis & Operasional Petani Jeruk dalam 11 modul terstruktur yang sangat detail untuk meningkatkan margin keuntungan petani:

=== DATA TELEMETRI CITRA POHON JERUK ===
{telemetry_json}
=======================================

INSTRUKSI PENYUSUNAN LAPORAN (WAJIB MEMUAT SELURUH 11 MODUL):

1. **MODUL 1: Status Fisiologis & Indeks Kesiapan Panen Jeruk (CHRI)**
   - Jelaskan sifat non-klimakterik jeruk (tidak akan manis jika dipetik terlalu muda).
   - Tampilkan skor CHRI (%) dari data dan tentukan statusnya (misal: Fase Puncak Panen Raya / Siap Petik Parsial).
   - Evaluasi populasi pohon dan rasio buah siap petik terhadap total buah unik yang terverifikasi.

2. **MODUL 2: Timeline Jadwal Panen Bertingkat (Staged Citrus Harvest Plan)**
   - Sajikan tabel 3 gelombang panen (Batch 1 Segera H s/d H+3, Batch 2 H+8 s/d H+14, Batch 3 H+20 s/d H+28).
   - Cantumkan target kematangan, estimasi jumlah buah, perkiraan bobot (kg), dan jam kerja lapangan (09.30 - 14.00 WIB bebas embun).

3. **MODUL 3: Estimasi Biomasa & Segmentasi Grade Mutu (SNI 01-3168-1992)**
   - Buat tabel 4 grade jeruk: Super A (≥160g), Standar B (120-159g), Kecil C (80-119g), Afkir D (<80g).
   - Tampilkan proyeksi tonase (kg) panen Batch 1 dan total potensi panen keseluruhan pohon.

4. **MODUL 4: Arbitrase Saluran Penjualan & Margin Laba Bersih**
   - Sajikan tabel komparasi 4 saluran: Penebas Lapangan, Pasar Induk Grosir, Kios Minuman Es Jeruk Peras, dan Supermarket Modern.
   - Cantumkan estimasi harga/kg, biaya operasional, risiko susut, dan rekomendasi bauran kanal terbaik untuk petani.
   - Berikan peringatan risiko finansial jika panen terlambat >5 hari (akibat susut kadar air dan fruit drop).

5. **MODUL 5: Skrip Negosiasi Harga Berbasis Data dengan Penebas**
   - Hitung nilai wajar dasar pohon (Bottom Line Fair Price) dalam rupiah.
   - Tuliskan skrip percakapan langsung yang sopan namun tegas bagi petani untuk menolak harga tebasan murah berbekal bukti data hitungan kamera.

6. **MODUL 6: Hilirisasi Zero-Waste & Monetisasi Jeruk Non-Grade**
   - Rencana pengolahan buah Grade C/D menjadi sari perasan murni beku (frozen citrus juice) dengan proyeksi nilai jual Rp 15.000 - Rp 20.000/liter.
   - Pemanfaatan kulit jeruk untuk minyak atsiri D-Limonene dan silase pakan ternak.

7. **MODUL 7: Alokasi Tenaga Kerja (HOK) & SOP Petik Bebas Rusak**
   - Hitung kebutuhan Hari Orang Kerja (HOK) berdasarkan target kg Batch 1 (standar 150 kg/buruh/hari).
   - Jelaskan SOP gunting pangkas tumpul (Curved Citrus Clipper) dan teknik potong tangkai 2 kali (Double Cut) rata calyx.
   - Larangan menggunakan karung goni (wajib keranjang plastik beralas busa).

8. **MODUL 8: Manajemen Nutrisi & Air Pra-Panen**
   - Protokol penahanan air (water restriction) 7-10 hari jelang petik untuk menaikkan Brix gula 1.5 - 2 derajat.
   - Anjuran pemupukan kalium (K2SO4 / KNO3) dan semprot kalsium boron (CaB) untuk mencegah pecah buah (fruit cracking).

9. **MODUL 9: Proteksi Hama Lalat Buah & Kapang Jamur Pasca-Petik**
   - Tindakan pencegahan lalat buah Bactrocera dorsalis (perangkap Methyl Eugenol di luar perimeter).
   - Pengendalian tungau karat jeruk dan sanitasi spora kapang Penicillium digitatum/italicum.

10. **MODUL 10: Protokol Pasca-Panen & Pencegahan Oleocellosis**
    - Aturan utama: jangan petik saat buah masih berembun pagi atau basah hujan.
    - SOP pemeraman ringan (curing) 24 jam di tempat teduh agar kulit lentur.
    - Pencucian air hangat 48-50°C dan pelapisan lilin lebah nabati (beeswax coating 6-8%).

11. **MODUL 11: Paspor Mutu Jeruk Digital (WhatsApp Ready)**
    - Sajikan template teks paspor mutu yang rapi, ringkas, dan siap disalin untuk dikirim ke pembeli/grosir via WhatsApp.

ATURAN KETAT:
- Semua angka jumlah buah, distribusi kelas, dan estimasi kg HARUS 100% konsisten dengan payload JSON.
- Gunakan bahasa Indonesia profesional agribisnis, padat, berbobot ilmiah-praktis, tanpa basa-basi pembuka yang tidak penting.
- Gunakan format Markdown rapi (tabel, poin, dan emoji penanda status)."""
    return prompt


def call_deepseek_citrus_advisor(
    telemetry: Dict[str, Any],
    config: Optional[CitrusAdvisorConfig] = None,
    stream: bool = True
) -> Union[Generator[str, None, None], str]:
    """
    Mengirimkan telemetri pohon jeruk ke DeepSeek v4.1 Flash via AIVO Gateway.
    Mendukung streaming token (SSE) untuk interaksi instan di Streamlit.
    """
    if config is None:
        config = CitrusAdvisorConfig()

    prompt = build_citrus_risen_prompt(telemetry)

    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Anda adalah sistem kecerdasan buatan penasihat agribisnis jeruk presisi. "
                    "Anda menghasilkan rekomendasi 11 modul agronomi dan finansial yang sangat detail, "
                    "berdasarkan data telemetri citra yang akurat, dengan fokus memaksimalkan laba bersih petani."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "stream": stream
    }

    url = f"{config.base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json"
    }

    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data_bytes, headers=headers)

    if stream:
        return _stream_generator(req, config.timeout)
    else:
        try:
            with urllib.request.urlopen(req, timeout=config.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8")
            raise RuntimeError(f"AIVO API HTTP Error {e.code}: {err_msg}")
        except Exception as e:
            raise RuntimeError(f"Gagal menghubungi AIVO Gateway: {str(e)}")


def _stream_generator(req: urllib.request.Request, timeout: int) -> Generator[str, None, None]:
    """Generator streaming Server-Sent Events (SSE) dari AIVO API."""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        yield f"\n\n⚠️ **Error AIVO Gateway (HTTP {e.code}):** {err_body}"
    except Exception as e:
        yield f"\n\n⚠️ **Gangguan Koneksi ke AIVO API:** {str(e)}"


def generate_whatsapp_passport(telemetry: Dict[str, Any]) -> str:
    """Menghasilkan teks Paspor Mutu Jeruk yang siap disalin untuk WhatsApp."""
    tree_id = telemetry.get("tree_id", "Pohon-Jeruk-01")
    variety = telemetry.get("citrus_variety", "Jeruk Siam")
    det_metrics = telemetry.get("cv_detection_metrics", {})
    biomass = telemetry.get("yield_biomass", {})
    maturity = telemetry.get("maturity_breakdown", {})
    timestamp = telemetry.get("timestamp", "Hari ini")

    unique_count = det_metrics.get("unique_verified_fruits", 0)
    chri = biomass.get("chri_score", 0.0)
    ready_kg = biomass.get("batch1_ready_weight_kg", 0.0)
    avg_weight = biomass.get("avg_fruit_weight_grams", 145.0)

    hr_count = maturity.get("Harvest-ready", 0)
    ripe_count = maturity.get("Ripe", 0)
    mature_count = maturity.get("Mature", 0)
    unripe_count = maturity.get("Unripe", 0)

    text = f"""🍊 *PASPOR MUTU PANEN JERUK (CITRUS QUALITY PASSPORT)*
ID Pohon       : {tree_id}
Varietas       : {variety}
Waktu Pemindaian: {timestamp}
-------------------------------------------------------------
📊 *POPULASI TERVERIFIKASI KAMERA*
- Total Populasi Buah Unik : {unique_count} Buah
- Kesiapan Panen (CHRI)   : {chri:.1f}%
- Siap Petik Gelombang 1  : {hr_count + int(ripe_count * 0.7)} Buah (~{ready_kg:.1f} Kg)

💎 *DISTRIBUSI KEMATANGAN*
- Harvest-ready (Kuning-Oranye) : {hr_count} Buah
- Ripe (Kuning Semburat Hijau)  : {ripe_count} Buah
- Mature (Hijau Kekuningan)     : {mature_count} Buah
- Unripe (Hijau Muda Pertumbuhan): {unripe_count} Buah

🛡️ *STANDAR JAMINAN MUTU LAPANGAN*
- Metode Petik   : Gunting Tumpul Rata Calyx (Bebas Luka Tangkai)
- Waktu Petik    : Siang Hari Bebas Embun (Bebas Oleocellosis)
- Bobot Rata-rata: ~{avg_weight:.0f} gram/buah
- Pasca-Panen    : Curing 24 Jam & Beeswax Coating Food Grade

Hubungi Petani/Kelompok Tani: [Nama Kebun / No. Telepon]
*Diverifikasi oleh CitrusVision AI & DeepSeek v4.1 Flash*"""
    return text


def generate_offline_heuristic_report(telemetry: Dict[str, Any]) -> str:
    """Laporan cadangan berbasis aturan jika koneksi AIVO terputus."""
    tree_id = telemetry.get("tree_id", "Pohon-01")
    variety = telemetry.get("citrus_variety", "Jeruk Siam")
    biomass = telemetry.get("yield_biomass", {})
    maturity = telemetry.get("maturity_breakdown", {})
    prices = telemetry.get("market_price_context", {})

    unique_count = telemetry.get("cv_detection_metrics", {}).get("unique_verified_fruits", 0)
    chri = biomass.get("chri_score", 0.0)
    ready_kg = biomass.get("batch1_ready_weight_kg", 0.0)
    total_kg = biomass.get("total_potential_yield_kg", 0.0)

    p_super = prices.get("price_super_grade_a", 22000)
    p_tengkulak = prices.get("price_tengkulak_raw", 8000)

    est_val_batch1 = ready_kg * p_super
    est_val_total = total_kg * p_super

    return f"""### 🍊 LAPORAN CADANGAN REKOMENDASI PETANI JERUK (OFFLINE HEURISTIC)
*Koneksi AI Gateway sedang offline. Laporan ini dihitung menggunakan model heuristik lokal.*

#### 1. Status Pohon & Indeks Kesiapan Panen (CHRI: {chri:.1f}%)
- **Objek:** {tree_id} ({variety})
- **Total Buah Unik:** {unique_count} buah
- **Kondisi:** {'Panen Puncak (Rekomendasi Petik Segera)' if chri >= 60 else 'Panen Parsial Bertahap'}

#### 2. Rencana Panen Bertingkat
- **Batch 1 (Hari H s/d H+3):** {maturity.get('Harvest-ready', 0) + int(maturity.get('Ripe', 0)*0.7)} buah (~{ready_kg:.1f} kg)
- **Batch 2 (H+8 s/d H+14):** {maturity.get('Mature', 0)} buah
- **Batch 3 (H+20 s/d H+28):** {maturity.get('Unripe', 0)} buah

#### 3. Proyeksi Finansial Cepat
- **Potensi Penjualan Batch 1 (Grade Super):** Rp {est_val_batch1:,.0f}
- **Jika Dijual Borongan ke Penebas (Rugi):** Rp {ready_kg * p_tengkulak:,.0f}
- **Selisih Keuntungan Petani:** +Rp {(est_val_batch1 - (ready_kg * p_tengkulak)):,.0f}

#### 4. Tindakan Lapangan Wajib
1. Pemetikan hanya dilakukan pukul 09.30 - 14.00 WIB saat embun kering total untuk menghindari *oleocellosis*.
2. Gunakan gunting pangkas khusus berkepala tumpul (double cut rata kelopak buah).
3. Hentikan penyiraman 7 hari sebelum petik untuk memadatkan rasa manis bulir jeruk."""
