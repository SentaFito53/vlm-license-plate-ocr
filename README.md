# OCR Plat Nomor Kendaraan menggunakan Visual Language Model (VLM)

Program ini melakukan Optical Character Recognition (OCR) pada plat nomor kendaraan Indonesia menggunakan Visual Language Model (VLM) yang dijalankan melalui LM Studio, diintegrasikan dengan Python. Hasil prediksi dievaluasi menggunakan metrik Character Error Rate (CER).

## Dataset

Dataset yang digunakan: [Indonesian License Plate Recognition Dataset](https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset) (folder `test`).

Struktur folder dataset yang diharapkan:
```
Indonesian License Plate Recognition Dataset/
├── images/test/       # gambar plat nomor
├── labels/test/       # label ground truth format YOLO (per karakter)
└── classes.names      # daftar karakter (mapping class_id -> karakter)
```

## Requirement

- Python 3.10+
- [LM Studio](https://lmstudio.ai/) (untuk menjalankan model VLM secara lokal)
- Model VLM multimodal, contoh: `llava-llama-3-8b-v1_1`

## Instalasi

1. Clone repository ini:
   ```
   git clone https://github.com/SentaFito53/vlm-license-plate-ocr-aas.git
   cd vlm-license-plate-ocr-aas
   ```

2. Install dependency Python:
   ```
   pip install -r requirements.txt
   ```

3. Download dan letakkan dataset di folder project (lihat struktur di atas). Dataset **tidak disertakan** di repo ini karena ukurannya besar — download manual dari link Kaggle di atas.

## Menjalankan LM Studio

1. Buka aplikasi LM Studio.
2. Download model VLM (contoh: `llava-llama-3-8b-v1_1`) lewat menu **Discover/Search**.
3. Buka mode **Developer** → pilih model yang sudah didownload → klik **Start Server**.
4. Pastikan server berjalan di `http://localhost:1234` (default) dan biarkan tetap terbuka selama program dijalankan.

## Menjalankan Program

Sesuaikan path dataset di bagian konfigurasi pada `main.py` jika diperlukan, lalu jalankan:

```
python main.py
```

Program akan:
1. Membaca setiap gambar plat nomor di folder `images/test`.
2. Merekonstruksi ground truth dari label YOLO per karakter di folder `labels/test`.
3. Mengirim gambar ke model VLM melalui LM Studio dengan prompt OCR.
4. Membersihkan hasil prediksi model menggunakan regex.
5. Menghitung CER (Character Error Rate) untuk setiap prediksi.
6. Menyimpan seluruh hasil ke file CSV (`hasil_ocr_plat.csv`) dengan kolom: `image, ground_truth, prediction, CER_score`.

## Formula CER

```
CER = (S + D + I) / N
```
- S = jumlah karakter salah substitusi
- D = jumlah karakter yang dihapus
- I = jumlah karakter yang disisipkan
- N = jumlah karakter pada ground truth

## Output

File `hasil_ocr_plat.csv` berisi hasil prediksi dan skor CER untuk setiap gambar, beserta rata-rata CER keseluruhan yang ditampilkan di terminal setelah program selesai.

## Struktur Repository

```
vlm-license-plate-ocr-aas/
├── main.py              # program utama
├── requirements.txt     # daftar dependency Python
├── README.md            # dokumentasi ini
└── .gitignore
```
