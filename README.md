# Deteksi Deepfake Berbasis Multi-Stream Fusion

Proyek skripsi untuk deteksi deepfake menggunakan arsitektur multi-stream berbasis Xception, menggabungkan fitur RGB, Residual Noise spasial, dan koefisien DCT melalui feature-level fusion.

Dataset yang digunakan adalah **IMSFD (Indonesian Muslim Student Face Dataset)** sebagai data real, dengan data fake yang digenerate menggunakan face-swap berbasis InsightFace.

---

## Struktur Proyek

```
deepfake-detection/
├── data/
│   ├── train/
│   │   ├── real/               # Foto wajah asli dari IMSFD (training)
│   │   └── fake/               # Foto hasil deepfake (training)
│   ├── val/
│   │   ├── real/               # Foto wajah asli dari IMSFD (validasi)
│   │   └── fake/               # Foto hasil deepfake (validasi)
│   └── test/
│       ├── real/               # Foto wajah asli dari IMSFD (testing)
│       └── fake/               # Foto hasil deepfake (testing)
├── config/
│   └── config.py               # Konfigurasi global dan per-method
├── src/
│   ├── models.py               # Arsitektur Xception (Baseline, Residual, Fusion)
│   ├── data_preprocessing.py   # Dataset loader dan augmentasi
│   ├── feature_extraction.py   # Ekstraksi Residual Noise dan DCT
│   ├── train.py                # Training loop dengan AMP dan gradient clipping
│   ├── evaluate.py             # Evaluasi dan visualisasi metrik
│   ├── visualization.py        # Grafik training, confusion matrix, ROC curve
│   └── utils.py                # Helper functions
├── saved_models/               # Model weights terbaik per method
│   └── checkpoints/            # Checkpoint per epoch (untuk resume)
├── output/
│   ├── figures/                # Grafik training curve per epoch
│   ├── visualizations/         # Confusion matrix, ROC, PR curve
│   ├── metrics/                # Hasil evaluasi dalam format JSON
│   └── predictions/            # Prediksi model pada test set
├── logs/                       # Training logs
├── main.py                     # Entry point dengan menu interaktif
├── generate_deepfake.py        # Generator deepfake dari IMSFD
├── split_dataset.py            # Pembagian dataset (70/15/15)
├── diagnose_dataset.py         # Diagnosa dataset dan face detection
├── requirements.txt            # Dependencies
└── README.md
```

---

## Persiapan Awal

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Untuk generator deepfake (InsightFace):

```bash
pip install insightface onnxruntime
# Jika menggunakan GPU:
pip install onnxruntime-gpu
```

### 2. Siapkan Dataset IMSFD

Download dataset dari: https://data.mendeley.com/datasets/f6f3y6ndgw/1

Struktur folder IMSFD yang diharapkan:

```
IMSFD/
├── A1/
│   ├── training/
│   │   └── <id_subjek>/
│   │       └── *.jpg
│   └── testing/
│       └── <id_subjek>/
│           └── *.jpg
├── A2/
├── B1/
└── ...
```

Setiap folder subjek berisi beberapa foto wajah orang yang sama. Semua foto dari IMSFD diperlakukan sebagai data **real (label 0)**.

### 3. Generate Data Deepfake

Sebelum generate, jalankan diagnosa untuk memastikan face detection bekerja:

```bash
python diagnose_dataset.py
```

Kemudian generate deepfake dengan face-swap antar subjek:

```bash
python generate_deepfake.py \
    --imsfd_dir "D:/Dataset/IMSFD" \
    --output_dir "D:/Dataset/IMSFD_deepfake" \
    --num_pairs 5000 \
    --swaps_per_pair 3
```

Foto hasil generate disimpan sebagai data **fake (label 1)**.

### 4. Split Dataset

Atur path di `split_dataset.py` kemudian jalankan:

```bash
python split_dataset.py
```

Dataset akan dibagi dengan rasio **70% train / 15% val / 15% test** dan dibalance otomatis antara kelas real dan fake.

---

## Training

Jalankan menu interaktif:

```bash
python main.py
```

Tersedia 4 method yang bisa dijalankan secara terpisah atau sekaligus:

| No | Method | Deskripsi | Batch Size |
|----|--------|-----------|------------|
| 1 | Baseline | Input RGB biasa, ablation study | 32 |
| 2 | Residual Spatial | Input residual noise domain spasial, ablation study | 32 |
| 3 | Residual DCT | Input koefisien DCT dari residual noise, ablation study | 32 |
| 4 | Multi-Stream Fusion | Gabungan RGB + Residual + DCT, metode utama | 8–16 |

> Method 1–3 adalah **ablation study** untuk memvalidasi kontribusi tiap komponen. Method 4 adalah tujuan utama penelitian.

### Catatan VRAM (GTX 1660 Ti / 6GB)

- Baseline, Residual Spatial, Residual DCT: batch size 32 aman (~3.5GB VRAM)
- Fusion: batch size 8–16 karena menjalankan 3 backbone sekaligus (~4.7GB VRAM)

---

## Konfigurasi

Edit `config/config.py` untuk mengubah parameter:

```python
# Hyperparameter training
BATCH_SIZE    = 32       # Turunkan ke 8-16 untuk Fusion
NUM_EPOCHS    = 12
LEARNING_RATE = 0.0001
WEIGHT_DECAY  = 1e-4

# Ukuran gambar
IMAGE_SIZE = 224

# Mixed Precision — aktifkan untuk GPU (lebih cepat, hemat VRAM)
USE_AMP = True

# Residual Noise
RESIDUAL_SPATIAL_KERNEL_SIZE = 5
RESIDUAL_SPATIAL_SIGMA       = 1.0

# DCT
DCT_BLOCK_SIZE = 8
```

---

## Output Training

Setiap method menghasilkan file berikut setelah training selesai:

- `saved_models/<method>_best_model.pth` — model terbaik berdasarkan val accuracy
- `saved_models/checkpoints/<method>_epoch_N.pth` — checkpoint per epoch
- `output/figures/<method>_epoch_progress/` — grafik loss dan accuracy per epoch
- `output/figures/<method>_training_curves.png` — kurva training lengkap
- `output/visualizations/<method>_confusion_matrix.png`
- `output/visualizations/<method>_roc_curve.png`
- `output/visualizations/<method>_precision_recall_curve.png`
- `output/metrics/<method>_metrics.json` — ringkasan metrik evaluasi

---

## Evaluasi dan Perbandingan

Pilih opsi **[7] Evaluate & Generate Results** di menu utama untuk mengevaluasi semua method sekaligus dan menghasilkan tabel perbandingan.

Pilih opsi **[8] Ablation Study Analysis** untuk melihat kontribusi tiap komponen secara detail.

Hasil perbandingan disimpan di `output/thesis_results.json` dan grafik radar chart di `output/visualizations/methods_comparison_dashboard.png`.

---

## Resume Training

Jika training terputus, pilih opsi **[6] Resume Training** di menu. Checkpoint per epoch otomatis disimpan sehingga training bisa dilanjutkan dari epoch terakhir tanpa mengulang dari awal.

---

## Troubleshooting

**CUDA out of memory (fusion model)**
Turunkan batch size di `FusionConfig` pada `config.py`:
```python
class FusionConfig(Config):
    BATCH_SIZE = 8
```

**Face detection gagal saat generate deepfake**
Jalankan diagnosa terlebih dahulu untuk cek ukuran foto dan det_size yang optimal:
```bash
python diagnose_dataset.py
```

**Dataset tidak ditemukan saat training**
Pastikan `split_dataset.py` sudah dijalankan dan folder `data/train/real/`, `data/train/fake/` sudah berisi gambar.

**Val accuracy tidak naik setelah beberapa epoch**
Model sudah konvergen. Scheduler akan otomatis menurunkan learning rate. Jika sudah 3 epoch berturut-turut tidak ada peningkatan, training bisa dihentikan — model terbaik sudah tersimpan otomatis.

---

## Referensi

- Chollet, F. (2017). Xception: Deep Learning with Depthwise Separable Convolutions.
- Rossler, A. et al. (2019). FaceForensics++: Learning to Detect Manipulated Facial Images.
- Faza, F. et al. (2023). IMSFD: Indonesian Muslim Student Face Dataset. Mendeley Data.
- InsightFace: https://github.com/deepinsight/insightface