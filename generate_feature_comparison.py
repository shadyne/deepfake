"""
Generate Gambar 4.4: Perbandingan Visualisasi Representasi Fitur
Menampilkan citra real dan fake yang sama diproses dengan 3 metode.
Output: output/figures/gambar_4_4_representasi_fitur.png
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import torch
import torchvision.transforms as T

from src.feature_extraction import ResidualNoiseExtractor, DCTExtractor

# ── Load satu citra real + satu citra fake ────────────────────────────────────
def load_tensor(path):
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    img = Image.open(path).convert('RGB')
    return tf(img)

def to_display(tensor):
    """Tensor [-1,1] → numpy [0,1] (H,W,C)."""
    arr = tensor.cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    arr = (arr + 1) / 2.0
    return np.clip(arr, 0, 1)

real_dir = 'data/train/real'
fake_dir = 'data/train/fake'

real_files = sorted(os.listdir(real_dir))
fake_files = sorted(os.listdir(fake_dir))

# Pilih satu citra representatif
real_tensor = load_tensor(os.path.join(real_dir, real_files[10]))
fake_tensor  = load_tensor(os.path.join(fake_dir,  fake_files[10]))

# ── Ekstraksi fitur ───────────────────────────────────────────────────────────
residual_ext = ResidualNoiseExtractor(kernel_size=5, sigma=1.0)
dct_ext       = DCTExtractor(block_size=8)

# Real
real_residual = residual_ext.extract(real_tensor)
real_dct      = dct_ext.extract(real_residual)

# Fake
fake_residual = residual_ext.extract(fake_tensor)
fake_dct      = dct_ext.extract(fake_residual)

# ── Ambil channel pertama residual (untuk heatmap) ────────────────────────────
def residual_heatmap(tensor):
    """Rata-rata 3 channel residual → heatmap 2D."""
    arr = tensor.cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    return np.mean(arr, axis=2)   # (H,W)

def dct_heatmap(tensor):
    """Rata-rata 3 channel DCT → heatmap 2D dengan contrast stretch."""
    arr = tensor.cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    gray = np.mean(arr, axis=2)
    # Histogram equalization manual: stretch ke percentile 2-98 agar detail terlihat
    p2, p98 = np.percentile(gray, 2), np.percentile(gray, 98)
    gray = np.clip((gray - p2) / (p98 - p2 + 1e-8), 0, 1)
    return gray

# ── Buat figure ───────────────────────────────────────────────────────────────
COLS   = ['Citra RGB\n(Baseline)', 'Residual Noise\n(Domain Spasial)', 'Koefisien DCT\n(dari Residual)']
ROWS   = ['Real', 'Deepfake (Fake)']
COLORS = ['#2E86AB', '#1B998B', '#F18F01']
LABEL_COLORS = ['#27AE60', '#C0392B']   # hijau=real, merah=fake

fig = plt.figure(figsize=(13, 7))
fig.patch.set_facecolor('#F0F0F0')

outer = gridspec.GridSpec(2, 3, figure=fig,
                          hspace=0.08, wspace=0.06,
                          left=0.08, right=0.98,
                          top=0.86, bottom=0.04)

data = {
    (0, 0): (to_display(real_tensor),   'viridis', False),
    (0, 1): (residual_heatmap(real_residual), 'RdBu_r', True),
    (0, 2): (dct_heatmap(real_dct),     'inferno',  True),
    (1, 0): (to_display(fake_tensor),   'viridis', False),
    (1, 1): (residual_heatmap(fake_residual), 'RdBu_r', True),
    (1, 2): (dct_heatmap(fake_dct),     'inferno',  True),
}

axes = {}
for (r, c), (img, cmap, is_gray) in data.items():
    ax = fig.add_subplot(outer[r, c])
    axes[(r, c)] = ax

    if is_gray:
        im = ax.imshow(img, cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, format='%.1f')
    else:
        ax.imshow(img, aspect='auto')

    ax.axis('off')

    # Border warna per baris (real/fake)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(LABEL_COLORS[r])
        spine.set_linewidth(2.5)

# ── Column headers (atas) ─────────────────────────────────────────────────────
for c, (label, color) in enumerate(zip(COLS, COLORS)):
    ax = axes[(0, c)]
    ax.set_title(label, fontsize=10, fontweight='bold',
                 color='white', pad=5,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.9))

# ── Row labels (kiri) ────────────────────────────────────────────────────────
for r, (label, color) in enumerate(zip(ROWS, LABEL_COLORS)):
    ax = axes[(r, 0)]
    ax.set_ylabel(label, fontsize=11, fontweight='bold', color=color,
                  rotation=90, labelpad=8)
    ax.yaxis.set_label_position('left')

# ── Judul utama ───────────────────────────────────────────────────────────────
fig.suptitle('Gambar 4.4  Perbandingan Visualisasi Representasi Fitur\n'
             'Citra Real vs Deepfake pada Tiga Metode Ekstraksi',
             fontsize=12, fontweight='bold', y=0.97)

os.makedirs('output/figures', exist_ok=True)
out_path = 'output/figures/gambar_4_4_representasi_fitur.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f'Tersimpan: {out_path}')
