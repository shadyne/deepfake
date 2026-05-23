"""
Generate Gambar 5.7, 5.8, 5.9: Confusion Matrix
- Gambar 5.7: Baseline (RGB)
- Gambar 5.8: Residual Spatial
- Gambar 5.9: Residual DCT
Output: output/figures/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

os.makedirs('output/figures', exist_ok=True)

MATRICES = {
    'baseline': {
        'cm':       np.array([[1824, 0], [9, 1815]]),
        'label':    'Baseline (RGB)',
        'fig_num':  '5.7',
        'fname':    'gambar_5_7_confusion_matrix_baseline.png',
        'color':    '#2E86AB',
    },
    'residual_spatial': {
        'cm':       np.array([[1819, 5], [3, 1821]]),
        'label':    'Residual Spatial',
        'fig_num':  '5.8',
        'fname':    'gambar_5_8_confusion_matrix_residual_spatial.png',
        'color':    '#1B998B',
    },
    'residual_dct': {
        'cm':       np.array([[1774, 50], [4, 1820]]),
        'label':    'Residual DCT',
        'fig_num':  '5.9',
        'fname':    'gambar_5_9_confusion_matrix_residual_dct.png',
        'color':    '#C73E1D',
    },
}

CLASSES = ['Real', 'Fake']

for key, cfg in MATRICES.items():
    cm    = cfg['cm']
    total = cm.sum()
    acc   = (cm[0,0] + cm[1,1]) / total * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#F8F8F8')
    ax.set_facecolor('#FAFAFA')

    # Heatmap manual
    vmax = cm.max()
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=vmax)

    # Anotasi tiap sel
    for i in range(2):
        for j in range(2):
            val   = cm[i, j]
            pct   = val / cm[i].sum() * 100
            color = 'white' if val > vmax * 0.5 else '#222222'
            ax.text(j, i, f'{val}\n({pct:.1f}%)',
                    ha='center', va='center',
                    fontsize=14, fontweight='bold', color=color)

    # Label
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASSES, fontsize=12)
    ax.set_yticklabels(CLASSES, fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12, labelpad=8)
    ax.set_ylabel('True Label', fontsize=12, labelpad=8)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    # Highlight diagonal
    for i in range(2):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor=cfg['color'],
                                   linewidth=2.5))

    # Metrics ringkas di bawah judul
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    prec    = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1      = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0

    fig.suptitle(
        f'Gambar {cfg["fig_num"]}  Confusion Matrix — {cfg["label"]}\n'
        f'Acc: {acc:.2f}%  |  Precision: {prec:.2f}%  |  Recall: {recall:.2f}%  |  F1: {f1:.2f}%',
        fontsize=11, fontweight='bold', y=1.04
    )

    plt.tight_layout(pad=1.5)
    out = f'output/figures/{cfg["fname"]}'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'Tersimpan: {out}')
