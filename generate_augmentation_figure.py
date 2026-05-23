"""
Generate Gambar 4.3: Contoh Hasil Augmentasi Data
Output: output/figures/gambar_4_3_augmentasi.png
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch

random.seed(0)
torch.manual_seed(0)

# ── Load satu sample citra asli ───────────────────────────────────────────────
img_dir = 'data/train/real'
files   = sorted(os.listdir(img_dir))
img_path = os.path.join(img_dir, files[10])   # pilih 1 citra
orig = Image.open(img_path).convert('RGB').resize((224, 224))

# ── Definisi setiap augmentasi (individual, bukan pipeline penuh) ─────────────
def aug_flip(img):
    return TF.hflip(img)

def aug_rotation(img):
    return TF.rotate(img, angle=8)

def aug_colorjitter(img):
    tf = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05)
    return tf(img)

def aug_crop(img):
    # Resize ke 244 lalu random crop 224
    img244 = TF.resize(img, 244)
    i, j, h, w = T.RandomCrop.get_params(img244, output_size=(224, 224))
    return TF.crop(img244, i, j, h, w)

def aug_affine(img):
    return TF.affine(img, angle=0, translate=(10, 8), scale=1.0, shear=0)

def aug_erasing(img):
    tensor = TF.to_tensor(img)
    # erase area kecil 10-15% lebar gambar
    x0, y0 = 80, 60
    x1, y1 = x0 + 45, y0 + 45
    tensor[:, y0:y1, x0:x1] = 0.5   # fill abu-abu
    return TF.to_pil_image(tensor)

augmentations = [
    ('Original',    orig,               '#555555'),
    ('H-Flip',      aug_flip(orig),     '#2E86AB'),
    ('Rotasi ±10°', aug_rotation(orig), '#1B998B'),
    ('ColorJitter', aug_colorjitter(orig), '#F18F01'),
    ('RandomCrop',  aug_crop(orig),     '#A23B72'),
    ('Affine\n(Translate)', aug_affine(orig), '#6B4226'),
    ('RandomErasing', aug_erasing(orig), '#C73E1D'),
]

# ── Plot ──────────────────────────────────────────────────────────────────────
n   = len(augmentations)
fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3.4))
fig.patch.set_facecolor('#F8F8F8')

for ax, (label, img, color) in zip(axes, augmentations):
    ax.imshow(img)
    ax.set_title(label, fontsize=9, fontweight='bold',
                 color='white', pad=4,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.85))
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2)

fig.suptitle('Gambar 4.3  Contoh Hasil Augmentasi Data Pelatihan',
             fontsize=12, fontweight='bold', y=1.01)

plt.tight_layout(pad=0.5)

os.makedirs('output/figures', exist_ok=True)
out_path = 'output/figures/gambar_4_3_augmentasi.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f'Tersimpan: {out_path}')
