"""
Generate Gambar 4.7: Visualisasi Output Lapisan Konvolusi
Layout bersih: 2 baris (Real/Fake) × 3 group lapisan,
tiap group = thumbnail citra + 6 feature map.
Output: output/figures/gambar_4_7_conv_visualization.png
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import torch
import torchvision.transforms as T

from src.models import XceptionBaseline
from src.feature_extraction import ResidualNoiseExtractor

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = 'saved_models/baseline_best_model.pth'
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_MAPS     = 6

# ── Load model ────────────────────────────────────────────────────────────────
model = XceptionBaseline(num_classes=4, pretrained=False)
ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt.get('model_state_dict', ckpt))
model.to(DEVICE).eval()

# ── Hook layers ───────────────────────────────────────────────────────────────
LAYERS = {
    'block1':  ('backbone.block1.rep.0',  'Lapisan Awal\n(Block 1)'),
    'block5':  ('backbone.block5.rep.1',  'Lapisan Tengah\n(Block 5)'),
    'block12': ('backbone.block12.rep.1', 'Lapisan Dalam\n(Block 12)'),
}

feature_maps = {}

def make_hook(key):
    def fn(m, i, o): feature_maps[key] = o.detach().cpu()
    return fn

hooks = []
for key, (path, _) in LAYERS.items():
    parts = path.split('.')
    layer = model
    for p in parts:
        layer = getattr(layer, p)
    hooks.append(layer.register_forward_hook(make_hook(key)))

# ── Image loading ─────────────────────────────────────────────────────────────
tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])

def load(path):
    return tf(Image.open(path).convert('RGB')).unsqueeze(0).to(DEVICE)

real_files = sorted(os.listdir('data/train/real'))
fake_files = sorted(os.listdir('data/train/fake'))
real_t = load(os.path.join('data/train/real', real_files[10]))
fake_t = load(os.path.join('data/train/fake', fake_files[10]))

with torch.no_grad():
    model(real_t)
real_fmaps = {k: v.clone() for k, v in feature_maps.items()}
feature_maps.clear()

with torch.no_grad():
    model(fake_t)
fake_fmaps = {k: v.clone() for k, v in feature_maps.items()}
feature_maps.clear()

for h in hooks:
    h.remove()

# ── Helpers ───────────────────────────────────────────────────────────────────
def top_channels(fmap, n=N_MAPS):
    fm = fmap[0]
    idx = fm.var(dim=[1, 2]).topk(n).indices
    return fm[idx].numpy()

def norm01(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)

def to_rgb(t):
    arr = t[0].cpu().numpy().transpose(1, 2, 0)
    return np.clip((arr + 1) / 2.0, 0, 1)

# ── Figure ────────────────────────────────────────────────────────────────────
# Layout:
#   Luar: 2 baris (Real, Fake) × 3 kolom (block1, block5, block12)
#   Tiap sel: inner grid 2×(N_MAPS/2+1) = thumb + 6 maps (2 baris × 3+1 cols)

LAYER_KEYS    = list(LAYERS.keys())
LAYER_LABELS  = [LAYERS[k][1] for k in LAYER_KEYS]
ROW_LABELS    = ['Real', 'Deepfake (Fake)']
ROW_COLORS    = ['#27AE60', '#C0392B']
COL_COLORS    = ['#2E86AB', '#1B998B', '#F18F01']
CMAP          = 'viridis'

fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor('#F2F2F2')

outer = gridspec.GridSpec(
    2, 3, figure=fig,
    hspace=0.18, wspace=0.10,
    left=0.07, right=0.98,
    top=0.87, bottom=0.03
)

for row_i, (row_label, row_color, all_fmaps, img_t) in enumerate(zip(
        ROW_LABELS, ROW_COLORS,
        [real_fmaps, fake_fmaps],
        [real_t, fake_t])):

    for col_i, (key, layer_label, col_color) in enumerate(zip(
            LAYER_KEYS, LAYER_LABELS, COL_COLORS)):

        maps = top_channels(all_fmaps[key])   # (N_MAPS, H, W)

        # Inner grid: 2 baris × 4 cols  → [thumb | F1 F2 F3] / [-- | F4 F5 F6]
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 4,
            subplot_spec=outer[row_i, col_i],
            hspace=0.06, wspace=0.05
        )

        # -- Thumbnail (kiri, span 2 baris) --
        ax_thumb = fig.add_subplot(inner[:, 0])
        ax_thumb.imshow(to_rgb(img_t))
        ax_thumb.axis('off')
        lbl = row_label.split()[0]
        ax_thumb.set_title(lbl, fontsize=8, fontweight='bold',
                           color='white', pad=3,
                           bbox=dict(boxstyle='round,pad=0.2',
                                     facecolor=row_color, alpha=0.9))
        for sp in ax_thumb.spines.values():
            sp.set_visible(True); sp.set_edgecolor(row_color); sp.set_linewidth(2)

        # -- Feature maps (3 per baris) --
        positions = [(0,1),(0,2),(0,3),(1,1),(1,2),(1,3)]
        for m_i, (r, c) in enumerate(positions):
            ax = fig.add_subplot(inner[r, c])
            ax.imshow(norm01(maps[m_i]), cmap=CMAP, aspect='auto')
            ax.axis('off')
            ax.text(0.96, 0.96, f'F{m_i+1}',
                    transform=ax.transAxes,
                    fontsize=6.5, color='white', fontweight='bold',
                    ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.1',
                              facecolor='black', alpha=0.55))
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(row_color)
                sp.set_linewidth(0.8)

# ── Column headers ────────────────────────────────────────────────────────────
for col_i, (layer_label, col_color) in enumerate(zip(LAYER_LABELS, COL_COLORS)):
    x = (col_i / 3) + (1 / 6)   # center of each column section
    x_mapped = 0.07 + x * 0.91
    fig.text(x_mapped, 0.90,
             layer_label.replace('\n', '  —  '),
             ha='center', va='bottom',
             fontsize=10, fontweight='bold', color='white',
             bbox=dict(boxstyle='round,pad=0.45',
                       facecolor=col_color, alpha=0.92))

# ── Row labels ────────────────────────────────────────────────────────────────
for row_i, (row_label, row_color) in enumerate(zip(ROW_LABELS, ROW_COLORS)):
    y = 0.72 if row_i == 0 else 0.33
    fig.text(0.01, y, row_label,
             ha='left', va='center', fontsize=11,
             fontweight='bold', color=row_color, rotation=90)

# ── Title ─────────────────────────────────────────────────────────────────────
fig.suptitle(
    'Gambar 4.7  Visualisasi Output Lapisan Konvolusi (Model Xception Baseline)\n'
    '6 Feature Map Paling Aktif pada Citra Real vs Deepfake  —  Awal, Tengah, Dalam',
    fontsize=12, fontweight='bold', y=0.985
)

os.makedirs('output/figures', exist_ok=True)
out = 'output/figures/gambar_4_7_conv_visualization.png'
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f'Tersimpan: {out}')
