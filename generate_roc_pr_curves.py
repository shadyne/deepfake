"""
Generate Gambar 5.10–5.13: ROC Curve & Precision-Recall Curve
Output: output/figures/
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

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
        'key':       'baseline',
        'label':     'Baseline (RGB)',
        'ModelClass': XceptionBaseline,
        'ckpt':      'saved_models/baseline_best_model.pth',
        'extractor': None,
        'color':     '#2E86AB',
        'fig_num':   '5.10',
        'fname':     'gambar_5_10_roc_baseline.png',
    },
    {
        'key':       'residual_spatial',
        'label':     'Residual Spatial',
        'ModelClass': XceptionResidualSpatial,
        'ckpt':      'saved_models/residual_spatial_best_model.pth',
        'extractor': 'spatial',
        'color':     '#1B998B',
        'fig_num':   '5.11',
        'fname':     'gambar_5_11_roc_residual_spatial.png',
    },
    {
        'key':       'residual_dct',
        'label':     'Residual DCT',
        'ModelClass': XceptionResidualDCT,
        'ckpt':      'saved_models/residual_dct_best_model.pth',
        'extractor': 'dct',
        'color':     '#C73E1D',
        'fig_num':   '5.12',
        'fname':     'gambar_5_12_roc_residual_dct.png',
    },
]

os.makedirs('output/figures', exist_ok=True)

all_results = []

for cfg in METHODS:
    model = cfg['ModelClass'](num_classes=4, pretrained=False)
    ckpt  = torch.load(cfg['ckpt'], map_location=DEVICE)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.to(DEVICE).eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            if cfg['extractor'] == 'spatial':
                imgs = torch.stack([residual_ext.extract(imgs[i].cpu()) for i in range(len(imgs))]).to(DEVICE)
            elif cfg['extractor'] == 'dct':
                processed = []
                for i in range(len(imgs)):
                    res = residual_ext.extract(imgs[i].cpu())
                    processed.append(dct_ext.extract(res))
                imgs = torch.stack(processed).to(DEVICE)
            else:
                imgs = imgs.to(DEVICE)

            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

    y_true  = np.array(all_labels)
    y_score = np.array(all_probs)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc     = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap           = average_precision_score(y_true, y_score)

    all_results.append({**cfg, 'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc,
                        'prec': prec, 'rec': rec, 'ap': ap})
    print(f'{cfg["label"]}: AUC={roc_auc:.6f}, AP={ap:.6f}')

    # ── Individual ROC curve ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#F8F8F8')
    ax.set_facecolor('#FAFAFA')

    ax.plot(fpr, tpr, color=cfg['color'], lw=2.5,
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.08, color=cfg['color'])

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    for sp in ax.spines.values():
        sp.set_edgecolor('#CCCCCC')

    fig.suptitle(
        f'Gambar {cfg["fig_num"]}  ROC Curve — {cfg["label"]}\n'
        f'Area Under Curve (AUC) = {roc_auc:.4f}',
        fontsize=11, fontweight='bold', y=1.03
    )
    plt.tight_layout(pad=1.5)
    out = f'output/figures/{cfg["fname"]}'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'Tersimpan: {out}')

# ── Gambar 5.13: PR Curve ketiga skenario ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor('#F8F8F8')
ax.set_facecolor('#FAFAFA')

for r in all_results:
    ax.plot(r['rec'], r['prec'], color=r['color'], lw=2.5,
            label=f'{r["label"]}  (AP = {r["ap"]:.4f})')
    ax.fill_between(r['rec'], r['prec'], alpha=0.06, color=r['color'])

ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0.60, 1.02)
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.legend(fontsize=10, loc='lower left')
ax.grid(True, alpha=0.3, linestyle='--')
for sp in ax.spines.values():
    sp.set_edgecolor('#CCCCCC')

fig.suptitle(
    'Gambar 5.13  Precision-Recall Curve — Perbandingan Ketiga Skenario',
    fontsize=11, fontweight='bold', y=1.02
)
plt.tight_layout(pad=1.5)
out = 'output/figures/gambar_5_13_pr_curve.png'
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f'Tersimpan: {out}')
