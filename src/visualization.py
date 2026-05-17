import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, classification_report
)
import os


def setup_plot_style():
    """Setup matplotlib style"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")


def plot_confusion_matrix(y_true, y_pred, method_name, save_dir, config):
    """Plot confusion matrix tersendiri"""
    setup_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Regular confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'],
                cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
    # Normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', ax=axes[1],
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'],
                cbar_kws={'label': 'Percentage'})
    axes[1].set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    
    fig.suptitle(f'Confusion Matrices - {method_name.upper()}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{method_name}_confusion_matrix.png')
    plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved: {save_path}")
    return save_path


def plot_roc_curve(y_true, y_scores, method_name, save_dir, config):
    """Plot ROC curve tersendiri"""
    setup_plot_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color='darkorange', lw=3, 
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier (AUC = 0.5000)')
    
    # Add some points
    ax.scatter(fpr[::len(fpr)//10], tpr[::len(tpr)//10], 
              color='darkorange', s=50, zorder=5, alpha=0.6)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_title(f'ROC Curve - {method_name.upper()}', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{method_name}_roc_curve.png')
    plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curve saved: {save_path}")
    return roc_auc, save_path


def plot_precision_recall_curve(y_true, y_scores, method_name, save_dir, config):
    """Plot Precision-Recall curve tersendiri"""
    setup_plot_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    
    ax.plot(recall, precision, color='purple', lw=3,
            label=f'PR Curve (AP = {avg_precision:.4f})')
    ax.fill_between(recall, precision, alpha=0.2, color='purple')
    
    # Add baseline
    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(y=baseline, color='red', linestyle='--', lw=2, 
              label=f'Baseline (AP = {baseline:.4f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=13, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
    ax.set_title(f'Precision-Recall Curve - {method_name.upper()}', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc="lower left", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{method_name}_precision_recall_curve.png')
    plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Precision-Recall curve saved: {save_path}")
    return save_path


def plot_class_distribution(y_true, y_pred, method_name, save_dir, config):
    """Plot class distribution tersendiri"""
    setup_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # True distribution
    true_counts = np.bincount(y_true)
    true_labels = ['Real', 'Fake']
    colors1 = ['#2E86AB', '#C73E1D']
    
    axes[0].bar(true_labels, true_counts, color=colors1, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[0].set_title('True Label Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for i, count in enumerate(true_counts):
        axes[0].text(i, count, f'{count}\n({count/len(y_true)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Predicted distribution
    pred_counts = np.bincount(y_pred)
    
    x = np.arange(2)
    width = 0.35
    
    bars1 = axes[1].bar(x - width/2, true_counts, width, label='True', 
                       color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = axes[1].bar(x + width/2, pred_counts, width, label='Predicted', 
                       color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    axes[1].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[1].set_title('True vs Predicted Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Real', 'Fake'])
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    fig.suptitle(f'Class Distribution - {method_name.upper()}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{method_name}_class_distribution.png')
    plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Class distribution saved: {save_path}")
    return save_path


def plot_metrics_bar_chart(y_true, y_pred, y_scores, method_name, save_dir, config):
    """Plot metrics bar chart tersendiri"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    setup_plot_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred) * 100,
        'Precision': precision_score(y_true, y_pred, zero_division=0) * 100,
        'Recall': recall_score(y_true, y_pred, zero_division=0) * 100,
        'F1-Score': f1_score(y_true, y_pred, zero_division=0) * 100,
        'AUC': roc_auc_score(y_true, y_scores) * 100
    }
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A4C93']
    bars = ax.barh(list(metrics.keys()), list(metrics.values()), 
                   color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Score (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Performance Metrics - {method_name.upper()}', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlim([0, 105])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, (name, value)) in enumerate(zip(bars, metrics.items())):
        ax.text(value + 1, i, f'{value:.2f}%', 
                va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{method_name}_metrics_bar_chart.png')
    plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics bar chart saved: {save_path}")
    return save_path


def plot_score_distribution(y_true, y_scores, method_name, save_dir, config):
    """Plot score distribution tersendiri"""
    setup_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    real_scores = y_scores[y_true == 0]
    fake_scores = y_scores[y_true == 1]
    
    # Histogram
    axes[0].hist(real_scores, bins=50, alpha=0.7, label='Real', color='green', 
                density=True, edgecolor='black', linewidth=0.5)
    axes[0].hist(fake_scores, bins=50, alpha=0.7, label='Fake', color='red', 
                density=True, edgecolor='black', linewidth=0.5)
    axes[0].axvline(x=0.5, color='black', linestyle='--', linewidth=2.5, 
                   label='Threshold (0.5)')
    axes[0].set_xlabel('Prediction Score (Fake Probability)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Density', fontsize=12, fontweight='bold')
    axes[0].set_title('Score Distribution - Histogram', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    data_to_plot = [real_scores, fake_scores]
    bp = axes[1].boxplot(data_to_plot, labels=['Real', 'Fake'], patch_artist=True,
                        widths=0.6, showfliers=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7, linewidth=1.5),
                        medianprops=dict(color='red', linewidth=2.5),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
    
    # Color boxes
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('red')
    bp['boxes'][1].set_alpha(0.6)
    
    axes[1].axhline(y=0.5, color='black', linestyle='--', linewidth=2, 
                   label='Threshold (0.5)')
    axes[1].set_ylabel('Prediction Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Score Distribution - Box Plot', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'Prediction Score Distribution - {method_name.upper()}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{method_name}_score_distribution.png')
    plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Score distribution saved: {save_path}")
    return save_path


def plot_comprehensive_metrics(y_true, y_pred, y_scores, method_name, save_dir, config):
    """
    Generate all visualization plots separately
    Returns AUC score
    """
    
    print(f"\n{'='*70}")
    print(f"GENERATING VISUALIZATIONS - {method_name.upper()}".center(70))
    print(f"{'='*70}")
    
    # 1. Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, method_name, save_dir, config)
    
    # 2. ROC Curve
    roc_auc, _ = plot_roc_curve(y_true, y_scores, method_name, save_dir, config)
    
    # 3. Precision-Recall Curve
    plot_precision_recall_curve(y_true, y_scores, method_name, save_dir, config)
    
    # 4. Class Distribution
    plot_class_distribution(y_true, y_pred, method_name, save_dir, config)
    
    # 5. Metrics Bar Chart
    plot_metrics_bar_chart(y_true, y_pred, y_scores, method_name, save_dir, config)
    
    # 6. Score Distribution
    plot_score_distribution(y_true, y_scores, method_name, save_dir, config)
    
    print(f"{'='*70}")
    print("All visualizations generated successfully!".center(70))
    print(f"{'='*70}\n")
    
    return roc_auc


def plot_training_curves(history, method_name, save_dir, config):
    """Plot training curves tersendiri"""
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', 
                   linewidth=2.5, markersize=7)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-s', label='Val Loss', 
                   linewidth=2.5, markersize=7)
    axes[0, 0].set_xlabel('Epoch', fontsize=13, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=13, fontweight='bold')
    axes[0, 0].set_title('Loss Curves', fontsize=15, fontweight='bold')
    axes[0, 0].legend(fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(epochs, history['train_acc'], 'b-o', label='Train Acc', 
                   linewidth=2.5, markersize=7)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-s', label='Val Acc', 
                   linewidth=2.5, markersize=7)
    axes[0, 1].set_xlabel('Epoch', fontsize=13, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    axes[0, 1].set_title('Accuracy Curves', fontsize=15, fontweight='bold')
    axes[0, 1].legend(fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Combined with fill
    axes[1, 0].plot(epochs, history['train_loss'], 'b-', label='Train', 
                   linewidth=3, alpha=0.8)
    axes[1, 0].plot(epochs, history['val_loss'], 'r-', label='Validation', 
                   linewidth=3, alpha=0.8)
    axes[1, 0].fill_between(epochs, history['train_loss'], alpha=0.2, color='blue')
    axes[1, 0].fill_between(epochs, history['val_loss'], alpha=0.2, color='red')
    axes[1, 0].set_xlabel('Epoch', fontsize=13, fontweight='bold')
    axes[1, 0].set_ylabel('Loss', fontsize=13, fontweight='bold')
    axes[1, 0].set_title('Loss Progression', fontsize=15, fontweight='bold')
    axes[1, 0].legend(fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning dynamics
    if len(history['train_acc']) > 1:
        train_improvement = np.array(history['train_acc'][1:]) - np.array(history['train_acc'][:-1])
        val_improvement = np.array(history['val_acc'][1:]) - np.array(history['val_acc'][:-1])
        
        axes[1, 1].plot(range(2, len(epochs) + 1), train_improvement, 'b-o', 
                       label='Train Improvement', linewidth=2.5, markersize=7)
        axes[1, 1].plot(range(2, len(epochs) + 1), val_improvement, 'r-s', 
                       label='Val Improvement', linewidth=2.5, markersize=7)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontsize=13, fontweight='bold')
        axes[1, 1].set_ylabel('Accuracy Change (%)', fontsize=13, fontweight='bold')
        axes[1, 1].set_title('Learning Dynamics', fontsize=15, fontweight='bold')
        axes[1, 1].legend(fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle(f'Training Progress - {method_name.upper()}', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    save_path = os.path.join(save_dir, f'{method_name}_training_curves.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved: {save_path}")


def plot_methods_comparison(results_dict, save_dir, config):    
    """Plot comparison antar methods"""
    setup_plot_style()
    
    methods = list(results_dict.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.ravel()
    
    colors = config.COLOR_PALETTE[:len(methods)]
    
    # Plot each metric
    for idx, (metric, label) in enumerate(zip(metrics_names, metrics_labels)):
        values = [results_dict[m].get(metric, 0) * (100 if metric != 'auc' else 1) 
                  for m in methods]
        
        bars = axes[idx].bar(methods, values, color=colors, alpha=0.8, 
                           edgecolor='black', linewidth=1.5)
        axes[idx].set_ylabel(f'{label} {"(%)" if metric != "auc" else ""}', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_title(f'{label} Comparison', fontsize=14, fontweight='bold')
        axes[idx].set_ylim([0, 105 if metric != 'auc' else 1.05])
        axes[idx].grid(True, alpha=0.3, axis='y')
        axes[idx].tick_params(axis='x', rotation=15)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{val:.2f}{"%" if metric != "auc" else ""}',
                          ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Overall comparison radar chart
    ax_radar = axes[5]
    ax_radar.remove()
    ax_radar = fig.add_subplot(2, 3, 6, projection='polar')
    
    angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
    angles += angles[:1]
    
    for i, method in enumerate(methods):
        values = [results_dict[method].get(m, 0) * (100 if m != 'auc' else 100) 
                  for m in metrics_names]
        values += values[:1]
        
        ax_radar.plot(angles, values, 'o-', linewidth=2.5, label=method, color=colors[i])
        ax_radar.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metrics_labels, fontsize=11)
    ax_radar.set_ylim(0, 100)
    ax_radar.set_title('Overall Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax_radar.grid(True)
    
    fig.suptitle('Methods Comparison Dashboard', fontsize=20, fontweight='bold', y=0.995)
    
    save_path = os.path.join(save_dir, 'methods_comparison_dashboard.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Methods comparison saved: {save_path}")