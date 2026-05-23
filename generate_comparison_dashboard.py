"""
Generate Gambar 5.17: Methods Comparison Dashboard
Bar chart + Radar chart perbandingan ketiga skenario.
Tanpa judul gambar pada figure.
Output: output/figures/gambar_5_17_comparison_dashboard.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

os.makedirs('output/figures', exist_ok=True)

# ── Data (macro-averaged, %) ──────────────────────────────────────────────────
METHODS = ['Baseline\n(RGB)', 'Residual\nSpasial', 'Residual\nDCT']
COLORS  = ['#2E86AB', '#1B998B', '#C73E1D']

METRICS = {
    'Accuracy':  [99.75, 99.78, 98.52],
    'Precision': [99.75, 99.78, 98.55],
    'Recall':    [99.75, 99.78, 98.52],
    'F1-Score':  [99.75, 99.78, 98.52],
    'AUC×100':   [99.98, 99.99, 99.94],
}
METRIC_LABELS = list(METRICS.keys())
METRIC_LABELS[-1] = 'AUC (×100)'

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6))
fig.patch.set_facecolor('#F8F8F8')

gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30,
                       left=0.06, right=0.97, top=0.88, bottom=0.12)

# ── Panel kiri: Grouped Bar Chart ─────────────────────────────────────────────
ax_bar = fig.add_subplot(gs[0])
ax_bar.set_facecolor('#FAFAFA')

n_metrics = len(METRICS)
n_methods = len(METHODS)
x = np.arange(n_metrics)
width = 0.22

for i, (method, color) in enumerate(zip(METHODS, COLORS)):
    vals = [list(METRICS.values())[m][i] for m in range(n_metrics)]
    bars = ax_bar.bar(x + (i - 1) * width, vals,
                      width, label=method.replace('\n', ' '),
                      color=color, alpha=0.88, edgecolor='white', linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=6.5,
                    fontweight='bold', color=color)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(METRIC_LABELS, fontsize=9.5)
ax_bar.set_ylim(97.5, 100.6)
ax_bar.set_ylabel('Nilai (%)', fontsize=11)
ax_bar.set_title('Perbandingan Metrik Evaluasi', fontsize=11, fontweight='bold', pad=8)
ax_bar.legend(fontsize=9, loc='lower right')
ax_bar.grid(True, axis='y', alpha=0.35, linestyle='--')
for sp in ax_bar.spines.values():
    sp.set_edgecolor('#CCCCCC')

# ── Panel kanan: Radar Chart ──────────────────────────────────────────────────
ax_radar = fig.add_subplot(gs[1], polar=True)
ax_radar.set_facecolor('#FAFAFA')

radar_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC (×100)']
radar_vals = [
    [99.75, 99.75, 99.75, 99.75, 99.98],
    [99.78, 99.78, 99.78, 99.78, 99.99],
    [98.52, 98.55, 98.52, 98.52, 99.94],
]

N = len(radar_metrics)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

for vals, method, color in zip(radar_vals, METHODS, COLORS):
    v = vals + vals[:1]
    ax_radar.plot(angles, v, color=color, lw=2.2, label=method.replace('\n', ' '))
    ax_radar.fill(angles, v, color=color, alpha=0.08)

ax_radar.set_theta_offset(np.pi / 2)
ax_radar.set_theta_direction(-1)
ax_radar.set_thetagrids(np.degrees(angles[:-1]), radar_metrics, fontsize=9)
ax_radar.set_ylim(97.5, 100.1)
ax_radar.set_yticks([98.0, 98.5, 99.0, 99.5, 100.0])
ax_radar.set_yticklabels(['98.0', '98.5', '99.0', '99.5', '100.0'], fontsize=7)
ax_radar.set_title('Radar Chart Perbandingan', fontsize=11, fontweight='bold', pad=14)
ax_radar.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.32, 1.12))
ax_radar.grid(True, alpha=0.4)

plt.tight_layout()
out = 'output/figures/gambar_5_17_comparison_dashboard.png'
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f'Tersimpan: {out}')
