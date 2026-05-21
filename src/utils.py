import torch
import numpy as np
import os
import json
from datetime import datetime
import random


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    
    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def get_lr(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def count_parameters(model, verbose=True):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if verbose:
        print(f"\nModel Parameters:")
        print(f"  Total:     {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Frozen:    {total_params - trainable_params:,}")
    
    return total_params, trainable_params


def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded: {filepath}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    return checkpoint


def save_metrics(metrics, filepath):
    """Save metrics to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved: {filepath}")


def load_metrics(filepath):
    """Load metrics from JSON file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metrics file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def print_system_info():
    """Print system information"""
    print("\n" + "="*70)
    print("SYSTEM INFORMATION".center(70))
    print("="*70)
    print(f"{'PyTorch Version:':<25} {torch.__version__}")
    print(f"{'CUDA Available:':<25} {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"{'CUDA Version:':<25} {torch.version.cuda}")
        print(f"{'GPU Device:':<25} {torch.cuda.get_device_name(0)}")
        print(f"{'GPU Memory:':<25} {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print("="*70 + "\n")


def print_training_summary(history, method_name):
    """Print training summary"""
    print("\n" + "="*70)
    print(f"TRAINING SUMMARY - {method_name.upper()}".center(70))
    print("="*70)
    
    epochs = len(history['train_loss'])
    
    print(f"{'Total Epochs:':<30} {epochs}")
    print(f"\n{'Initial Training Acc:':<30} {history['train_acc'][0]:.2f}%")
    print(f"{'Final Training Acc:':<30} {history['train_acc'][-1]:.2f}%")
    print(f"{'Training Improvement:':<30} {history['train_acc'][-1] - history['train_acc'][0]:+.2f}%")
    
    print(f"\n{'Initial Validation Acc:':<30} {history['val_acc'][0]:.2f}%")
    print(f"{'Final Validation Acc:':<30} {history['val_acc'][-1]:.2f}%")
    print(f"{'Validation Improvement:':<30} {history['val_acc'][-1] - history['val_acc'][0]:+.2f}%")
    
    print(f"\n{'Best Validation Acc:':<30} {max(history['val_acc']):.2f}%")
    print(f"{'Best Epoch:':<30} {history['val_acc'].index(max(history['val_acc'])) + 1}")
    
    print("="*70 + "\n")


def create_comparison_table(results_dict):
    """Create formatted comparison table"""
    print("\n" + "="*90)
    print("METHODS COMPARISON TABLE".center(90))
    print("="*90)
    
    # Header
    header = f"{'Method':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC':<10}"
    print(header)
    
    # Rows
    for method, metrics in results_dict.items():
        accuracy = metrics.get('accuracy', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        f1 = metrics.get('f1_score', 0)
        auc_score = metrics.get('auc', 0)
        
        row = f"{method:<20} {accuracy:<11.2f}% {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {auc_score:<10.4f}"
        print(row)
    
    print("="*90 + "\n")


def print_method_info(method_name, config):
    """Print method information"""
    descriptions = {
        'baseline': 'Standard Xception with RGB input',
        'residual_spatial': 'Xception with Residual Noise extraction (Spatial Domain)',
        'residual_dct': 'Xception with DCT coefficients from Residual Noise',
    }
    
    print("\n" + "="*70)
    print(f"METHOD: {method_name.upper()}".center(70))
    print("="*70)
    print(f"Description: {descriptions.get(method_name, 'Unknown method')}")
    print("="*70 + "\n")


def get_device_info():
    """Get device information"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    info = {
        'device': str(device),
        'cuda_available': torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        info['device_name'] = torch.cuda.get_device_name(0)
        info['device_count'] = torch.cuda.device_count()
    
    return info


def format_time(seconds):
    """Format seconds to human-readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"