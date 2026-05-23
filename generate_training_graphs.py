"""
Generate grafik akurasi dan loss pelatihan (dipisah per file):
  gambar_5_1_akurasi_baseline.png
  gambar_5_2_loss_baseline.png
  gambar_5_3_akurasi_residual_spatial.png
  gambar_5_4_loss_residual_spatial.png
  gambar_5_5_akurasi_residual_dct.png
  gambar_5_6_loss_residual_dct.png
Output: output/figures/
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DEVICE    = 'cpu'
CKPT_DIR  = 'saved_models/checkpoints'
N_EPOCHS  = 12

METHODS = {
    'baseline': {
        'label':      'Baseline (RGB)',
        'color_tr':   '#2E86AB',
        'color_val':  '#F18F01',
        'best_ep':    8,
        'fig_acc':    ('5.1', 'gambar_5_1_akurasi_baseline.png'),
        'fig_loss':   ('5.2', 'gambar_5_2_loss_baseline.png'),
    },
    'residual_spatial': {
        'label':      'Residual Spatial',
        'color_tr':   '#1B998B',
        'color_val':  '#A23B72',
        'best_ep':    8,
        'fig_acc':    ('5.3', 'gambar_5_3_akurasi_residual_spatial.png'),
        'fig_loss':   ('5.4', 'gambar_5_4_loss_residual_spatial.png'),
    },
    'residual_dct': {
        'label':      'Residual DCT',
        'color_tr':   '#6B4226',
        'color_val':  '#C73E1D',
        'best_ep':    12,
        'fig_acc':    ('5.5', 'gambar_5_5_akurasi_residual_dct.png'),
        'fig_loss':   ('5.6', 'gambar_5_6_loss_residual_dct.png'),
    },
}

os.makedirs('output/figures', exist_ok=True)
epochs = list(range(1, N_EPOCHS + 1))


def styled_ax(ax):
    ax.set_facecolor('#FAFAFA')
    ax.set_xlim(0.5, N_EPOCHS + 0.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.grid(True, alpha=0.35, linestyle='--')
    for sp in ax.spines.values():
        sp.set_edgecolor('#CCCCCC')


for key, cfg in METHODS.items():
    last_ckpt = os.path.join(CKPT_DIR, f'{key}_epoch_{N_EPOCHS}.pth')
    ckpt  = torch.load(last_ckpt, map_location=DEVICE)
    hist  = ckpt['history']

    train_acc  = [v * 100 for v in hist['train_acc']]
    val_acc    = [v * 100 for v in hist['val_acc']]
    train_loss = hist['train_loss']
    val_loss   = hist['val_loss']

    best_ep = cfg['best_ep']

    # ── Gambar Akurasi ────────────────────────────────────────────────────────
    fig_num_acc, fname_acc = cfg['fig_acc']
    best_val_acc = val_acc[best_ep - 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#F8F8F8')
    styled_ax(ax)

    ax.plot(epochs, train_acc, 'o-', color=cfg['color_tr'],
            linewidth=2.2, markersize=6, label='Train Accuracy')
    ax.plot(epochs, val_acc,   's--', color=cfg['color_val'],
            linewidth=2.2, markersize=6, label='Validation Accuracy')
    ax.axvline(best_ep, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.scatter([best_ep], [best_val_acc], color='gold', edgecolors='black',
               zorder=5, s=120, label=f'Best Epoch {best_ep} ({best_val_acc:.2f}%)')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Akurasi (%)', fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.legend(fontsize=10, loc='lower right')
    fig.suptitle(
        f'Gambar {fig_num_acc}  Grafik Akurasi Pelatihan dan Validasi — {cfg["label"]}\n'
        f'Epoch 1–{N_EPOCHS}, Best Validation Accuracy pada Epoch {best_ep}',
        fontsize=11, fontweight='bold', y=1.02
    )
    plt.tight_layout(pad=1.5)
    out = f'output/figures/{fname_acc}'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'Tersimpan: {out}')

    # ── Gambar Loss ───────────────────────────────────────────────────────────
    fig_num_loss, fname_loss = cfg['fig_loss']
    best_val_loss = val_loss[best_ep - 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#F8F8F8')
    styled_ax(ax)

    ax.plot(epochs, train_loss, 'o-', color=cfg['color_tr'],
            linewidth=2.2, markersize=6, label='Train Loss')
    ax.plot(epochs, val_loss,   's--', color=cfg['color_val'],
            linewidth=2.2, markersize=6, label='Validation Loss')
    ax.axvline(best_ep, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.scatter([best_ep], [best_val_loss], color='gold', edgecolors='black',
               zorder=5, s=120, label=f'Best Epoch {best_ep} (loss={best_val_loss:.4f})')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    fig.suptitle(
        f'Gambar {fig_num_loss}  Grafik Loss Pelatihan dan Validasi — {cfg["label"]}\n'
        f'Epoch 1–{N_EPOCHS}, Best Validation Loss pada Epoch {best_ep}',
        fontsize=11, fontweight='bold', y=1.02
    )
    plt.tight_layout(pad=1.5)
    out = f'output/figures/{fname_loss}'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'Tersimpan: {out}')
