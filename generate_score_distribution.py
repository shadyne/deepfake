"""
Generate Gambar 5.14–5.16: Distribusi Skor Prediksi (Softmax Fake)
Tanpa judul gambar pada figure (caption di Word).
Output: output/figures/
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from src.models import XceptionBaseline, XceptionResidualSpatial, XceptionResidualDCT
from src.feature_extraction import ResidualNoiseExtractor, DCTExtractor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

class SimpleDataset(Dataset):
    def __init__(self, root, transform):
        self.samples = []
        for label, cls in enumerate(['real', 'fake']):
            d = os.path.join(root, cls)
            for f in sorted(os.listdir(d)):
                self.samples.append((os.path.join(d, f), label))
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, l = self.samples[i]
        return self.transform(Image.open(p).convert('RGB')), l

test_ds = SimpleDataset('data/test', tf)
loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

residual_ext = ResidualNoiseExtractor()
dct_ext      = DCTExtractor()

METHODS = [
    {
        'label':      'Baseline (RGB)',
        'ModelClass':  XceptionBaseline,
        'ckpt':       'saved_models/baseline_best_model.pth',
        'extractor':  None,
        'color_real': '#2E86AB',
        'color_fake': '#F18F01',
        'fname':      'gambar_5_14_score_dist_baseline.png',
    },
    {
        'label':      'Residual Spatial',
        'ModelClass':  XceptionResidualSpatial,
        'ckpt':       'saved_models/residual_spatial_best_model.pth',
        'extractor':  'spatial',
        'color_real': '#1B998B',
        'color_fake': '#A23B72',
        'fname':      'gambar_5_15_score_dist_residual_spatial.png',
    },
    {
        'label':      'Residual DCT',
        'ModelClass':  XceptionResidualDCT,
        'ckpt':       'saved_models/residual_dct_best_model.pth',
        'extractor':  'dct',
        'color_real': '#2E86AB',
        'color_fake': '#C73E1D',
        'fname':      'gambar_5_16_score_dist_residual_dct.png',
    },
]

os.makedirs('output/figures', exist_ok=True)

for cfg in METHODS:
    model = cfg['ModelClass'](num_classes=4, pretrained=False)
    ckpt  = torch.load(cfg['ckpt'], map_location=DEVICE)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.to(DEVICE).eval()

    scores_real, scores_fake = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            if cfg['extractor'] == 'spatial':
                imgs = torch.stack([residual_ext.extract(imgs[i].cpu())
                                    for i in range(len(imgs))]).to(DEVICE)
            elif cfg['extractor'] == 'dct':
                processed = []
                for i in range(len(imgs)):
                    res = residual_ext.extract(imgs[i].cpu())
                    processed.append(dct_ext.extract(res))
                imgs = torch.stack(processed).to(DEVICE)
            else:
                imgs = imgs.to(DEVICE)

            probs = torch.softmax(model(imgs), dim=1)[:, 1].cpu().numpy()
            for p, l in zip(probs, labels.numpy()):
                (scores_real if l == 0 else scores_fake).append(float(p))

    scores_real = np.array(scores_real)
    scores_fake = np.array(scores_fake)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor('#F8F8F8')
    ax.set_facecolor('#FAFAFA')

    x = np.linspace(0, 1, 500)

    # KDE Real
    kde_r = gaussian_kde(scores_real, bw_method=0.05)
    y_r   = kde_r(x)
    ax.plot(x, y_r, color=cfg['color_real'], lw=2.5, label='Real')
    ax.fill_between(x, y_r, alpha=0.18, color=cfg['color_real'])

    # KDE Fake
    kde_f = gaussian_kde(scores_fake, bw_method=0.05)
    y_f   = kde_f(x)
    ax.plot(x, y_f, color=cfg['color_fake'], lw=2.5, label='Fake')
    ax.fill_between(x, y_f, alpha=0.18, color=cfg['color_fake'])

    # Threshold line
    ax.axvline(0.5, color='gray', linestyle='--', lw=1.5, alpha=0.8, label='Threshold = 0.5')

    ax.set_xlabel('Skor Prediksi (Probabilitas Kelas Fake)', fontsize=12)
    ax.set_ylabel('Densitas', fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    for sp in ax.spines.values():
        sp.set_edgecolor('#CCCCCC')

    # Anotasi overlap / separasi
    overlap = np.minimum(y_r, y_f)
    sep_pct = (1 - np.trapz(overlap, x)) * 100
    ax.text(0.50, ax.get_ylim()[1] * 0.92,
            f'Separasi: {sep_pct:.1f}%',
            ha='center', fontsize=9.5, color='#444444',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    plt.tight_layout(pad=1.5)
    out = f'output/figures/{cfg["fname"]}'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

    real_lo = (scores_real < 0.5).mean() * 100
    fake_hi = (scores_fake >= 0.5).mean() * 100
    print(f'{cfg["label"]}: separasi={sep_pct:.1f}%  '
          f'real<0.5={real_lo:.2f}%  fake>=0.5={fake_hi:.2f}%')
    print(f'Tersimpan: {out}')
