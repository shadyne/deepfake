"""
Generate Gambar 4.8: Visualisasi Output Lapisan Pooling
Menampilkan penurunan resolusi spatial secara progresif melalui 4 MaxPool
(55×55 → 28×28 → 14×14 → 7×7) untuk citra Real vs Fake.
Output: output/figures/gambar_4_8_pooling_visualization.png
"""

import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from PIL import Image
import torch
import torchvision.transforms as T

from src.models import XceptionBaseline

# ── Load model ────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = XceptionBaseline(num_classes=4, pretrained=False)
ckpt   = torch.load('saved_models/baseline_best_model.pth', map_location=DEVICE)
model.load_state_dict(ckpt.get('model_state_dict', ckpt))
model.to(DEVICE).eval()

# ── Hook 4 MaxPool + Global AvgPool ──────────────────────────────────────────
POOL_LAYERS = {
    'pool1': ('backbone.block1.rep.5',  'MaxPool #1\n(128 ch, 55×55)',  '#2E86AB'),
    'pool2': ('backbone.block2.rep.6',  'MaxPool #2\n(256 ch, 28×28)',  '#1B998B'),
    'pool3': ('backbone.block3.rep.6',  'MaxPool #3\n(728 ch, 14×14)',  '#F18F01'),
    'pool4': ('backbone.block12.rep.6', 'MaxPool #4\n(1024 ch,  7×7)', '#A23B72'),
}

feature_maps = {}

def make_hook(key):
    def fn(m, i, o): feature_maps[key] = o.detach().cpu()
    return fn

hooks = []
for key, (path, _, _) in POOL_LAYERS.items():
    parts = path.split('.')
    layer = model
    for p in parts: layer = getattr(layer, p)
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

for h in hooks: h.remove()

# ── Helpers ───────────────────────────────────────────────────────────────────
N_MAPS = 5   # feature maps per pool layer

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
# Layout utama: 3 baris
#   Baris 0: diagram alur resolusi (info grafis)
#   Baris 1: Real    — [thumb | 5 fmap per pool × 4 pool]
#   Baris 2: Fake    — [thumb | 5 fmap per pool × 4 pool]

ROW_LABELS  = ['Real', 'Deepfake\n(Fake)']
ROW_COLORS  = ['#27AE60', '#C0392B']
COL_COLORS  = [v[2] for v in POOL_LAYERS.values()]
COL_LABELS  = [v[1] for v in POOL_LAYERS.values()]
SHAPES      = ['55×55', '28×28', '14×14', '7×7']
CHANNELS    = [128, 256, 728, 1024]

fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor('#EBEBEB')

main_gs = gridspec.GridSpec(
    3, 1, figure=fig,
    height_ratios=[0.8, 4, 4],
    hspace=0.18,
    left=0.06, right=0.99,
    top=0.90, bottom=0.03
)

# ── Baris 0: Info grafis resolusi ─────────────────────────────────────────────
ax_info = fig.add_subplot(main_gs[0])
ax_info.set_xlim(0, 1)
ax_info.set_ylim(0, 1)
ax_info.axis('off')
ax_info.set_facecolor('#EBEBEB')

# Input box
input_x = 0.04
ax_info.text(input_x, 0.5, 'Input\n224×224\n(RGB)',
             ha='center', va='center', fontsize=9, fontweight='bold',
             color='white',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#555555', alpha=0.9))

# Pool boxes dengan panah
pool_xs = [0.23, 0.42, 0.61, 0.80]
for i, (px, shape, ch, color) in enumerate(zip(pool_xs, SHAPES, CHANNELS, COL_COLORS)):
    # Panah
    arrow_x0 = input_x + 0.045 if i == 0 else pool_xs[i-1] + 0.065
    ax_info.annotate('', xy=(px - 0.055, 0.5), xytext=(arrow_x0, 0.5),
                     arrowprops=dict(arrowstyle='->', color='#444444', lw=1.5))
    # Box
    ax_info.text(px, 0.5,
                 f'MaxPool #{i+1}\n{ch} ch\n{shape}',
                 ha='center', va='center', fontsize=9, fontweight='bold',
                 color='white',
                 bbox=dict(boxstyle='round,pad=0.45', facecolor=color, alpha=0.9))

# Label "resolusi menurun"
ax_info.annotate('', xy=(0.86, 0.08), xytext=(0.14, 0.08),
                 arrowprops=dict(arrowstyle='->', color='#888888', lw=1.5,
                                 connectionstyle='arc3'))
ax_info.text(0.50, 0.03, '← resolusi spatial menurun, jumlah channel meningkat →',
             ha='center', va='bottom', fontsize=8.5, color='#555555', style='italic')

# ── Baris 1 & 2: Feature maps ─────────────────────────────────────────────────
for row_i, (row_label, row_color, all_fmaps, img_t) in enumerate(zip(
        ROW_LABELS, ROW_COLORS,
        [real_fmaps, fake_fmaps],
        [real_t, fake_t])):

    row_gs = gridspec.GridSpecFromSubplotSpec(
        1, 5,   # thumb + 4 pool groups
        subplot_spec=main_gs[row_i + 1],
        width_ratios=[1, 2.5, 2.5, 2.5, 2.5],
        wspace=0.08
    )

    # Thumbnail
    ax_th = fig.add_subplot(row_gs[0])
    ax_th.imshow(to_rgb(img_t))
    ax_th.axis('off')
    ax_th.set_title(row_label.split('\n')[0], fontsize=10, fontweight='bold',
                    color='white', pad=4,
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor=row_color, alpha=0.9))
    for sp in ax_th.spines.values():
        sp.set_visible(True); sp.set_edgecolor(row_color); sp.set_linewidth(2.5)

    # 4 pool groups
    for col_i, (key, col_color) in enumerate(zip(POOL_LAYERS.keys(), COL_COLORS)):
        maps = top_channels(all_fmaps[key])  # (N_MAPS, H, W)

        inner = gridspec.GridSpecFromSubplotSpec(
            1, N_MAPS,
            subplot_spec=row_gs[col_i + 1],
            wspace=0.04
        )

        for m_i in range(N_MAPS):
            ax = fig.add_subplot(inner[0, m_i])
            ax.imshow(norm01(maps[m_i]), cmap='inferno', aspect='auto')
            ax.axis('off')

            ax.text(0.97, 0.97, f'F{m_i+1}',
                    transform=ax.transAxes,
                    fontsize=6, color='white', fontweight='bold',
                    ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.1',
                              facecolor='black', alpha=0.5))

            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(row_color)
                sp.set_linewidth(0.9)

            # Header ukuran hanya pada baris pertama, map pertama tiap group
            if row_i == 0 and m_i == 0:
                ax.set_title(SHAPES[col_i], fontsize=8, fontweight='bold',
                             color='white', pad=3,
                             bbox=dict(boxstyle='round,pad=0.2',
                                       facecolor=col_color, alpha=0.85))

# ── Column group labels (atas baris feature map pertama) ─────────────────────
# Posisi relatif ke figure
col_xs = [0.20, 0.38, 0.57, 0.76]
for cx, label, color in zip(col_xs, COL_LABELS, COL_COLORS):
    label_clean = label.replace('\n', '  ')
    fig.text(cx + 0.04, 0.91, label_clean,
             ha='center', va='bottom', fontsize=9.5, fontweight='bold',
             color='white',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.92))

# ── Row labels (sisi kiri) ────────────────────────────────────────────────────
fig.text(0.01, 0.69, 'Real', ha='center', va='center',
         fontsize=11, fontweight='bold', color=ROW_COLORS[0], rotation=90)
fig.text(0.01, 0.32, 'Fake', ha='center', va='center',
         fontsize=11, fontweight='bold', color=ROW_COLORS[1], rotation=90)

# ── Title ─────────────────────────────────────────────────────────────────────
fig.suptitle(
    'Gambar 4.8  Visualisasi Output Lapisan Pooling (Model Xception Baseline)\n'
    'Penurunan Resolusi Spatial Progresif: 55×55 → 28×28 → 14×14 → 7×7',
    fontsize=12, fontweight='bold', y=0.985
)

os.makedirs('output/figures', exist_ok=True)
out = 'output/figures/gambar_4_8_pooling_visualization.png'
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f'Tersimpan: {out}')
