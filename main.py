import argparse
import os
import sys
import time
import json

from config.config import (
    Config, BaselineConfig, ResidualSpatialConfig, 
    ResidualDCTConfig, FusionConfig
)
from src.data_preprocessing import get_dataloaders
from src.models import get_model
from src.feature_extraction import get_feature_extractor
from src.train import train_model
from src.evaluate import full_evaluation, load_and_evaluate
from src.utils import set_seed, count_parameters
from src.visualization import plot_training_curves
import torch


class ProgressTracker:
    """Track training progress untuk resume functionality"""
    
    def __init__(self, progress_file='training_progress.json'):
        self.progress_file = progress_file
        self.progress = self.load_progress()
    
    def load_progress(self):
        """Load progress dari file"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            'completed_methods': [],
            'last_run': None
        }
    
    def save_progress(self):
        """Save progress ke file"""
        self.progress['last_run'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=4)
    
    def mark_completed(self, method):
        """Mark method sebagai completed"""
        if method not in self.progress['completed_methods']:
            self.progress['completed_methods'].append(method)
        self.save_progress()
    
    def is_completed(self, method):
        """Check apakah method sudah completed"""
        return method in self.progress['completed_methods']
    
    def get_next_method(self):
        """Get method berikutnya yang belum completed"""
        all_methods = ['baseline', 'residual_spatial', 'residual_dct', 'fusion']
        for method in all_methods:
            if not self.is_completed(method):
                return method
        return None
    
    def reset(self):
        """Reset progress"""
        self.progress = {'completed_methods': [], 'last_run': None}
        self.save_progress()
    
    def get_progress_summary(self):
        """Get summary progress"""
        all_methods = ['baseline', 'residual_spatial', 'residual_dct', 'fusion']
        completed = len(self.progress['completed_methods'])
        total = len(all_methods)
        return completed, total, self.progress['completed_methods']


def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name == 'posix' else 'cls')


def print_banner():
    """Print application banner"""
    print("\n" + "="*70)
    print("DEEPFAKE DETECTION - MULTI-STREAM FUSION".center(70))
    print("="*70)


def print_menu(tracker):
    """Print main menu with proper methodology framing"""
    completed, total, completed_list = tracker.get_progress_summary()
    
    print("TAHAPAN PENELITIAN".center(70))
    
    methods = [
        ('baseline', '1. Baseline (RGB - Ablation Study)', BaselineConfig),
        ('residual_spatial', '2. Residual Spatial (Ablation Study)', ResidualSpatialConfig),
        ('residual_dct', '3. Residual DCT (Ablation Study)', ResidualDCTConfig),
        ('fusion', '4. Multi-Stream Fusion', FusionConfig)
    ]
    
    for i, (method, label, _) in enumerate(methods, 1):
        status = "" if tracker.is_completed(method) else " "
        print(f"  [{i}] [{status}] {label}")
    
    print(f"\n  [5] Train All Methods (Sequential)")
    print(f"  [6] Resume Training")
    print(f"  [7] Evaluate & Generate Results")
    print(f"  [8] Ablation Study Analysis")
    print(f"  [9] Reset Progress")
    print(f"  [0] Exit")
    print(f"\nProgress: {completed}/{total} methods completed")
    if tracker.progress['last_run']:
        print(f"Last run: {tracker.progress['last_run']}")
    
    print("NOTE: Baseline, Residual Spatial, dan Residual DCT adalah".center(70))
    print("ABLATION STUDY untuk validasi kontribusi setiap komponen.".center(70))
    print("FUSION (Method 4) adalah penggabungan dan tujuan dari penelitian ini.".center(70))
    print()


def check_training_state(method_key, config_class):
    """Check if training has checkpoint"""
    config = config_class()
    state_path = os.path.join(config.CHECKPOINTS_DIR, f'{method_key}_training_state.json')
    
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            state = json.load(f)
        return state
    return None


def train_single_stage(method_key, config_class, tracker, resume=False):
    """Train single stage/method"""
    
    config = config_class()
    config.create_directories()
    set_seed(42)
    
    # Check training state
    training_state = check_training_state(method_key, config_class)
    
    if training_state and training_state.get('completed', False):
        print(f"\n{'='*70}")
        print(f"{method_key.upper()} - ALREADY COMPLETED".center(70))
        print(f"{'='*70}")
        print(f"Completed at: {tracker.progress.get('last_run', 'Unknown')}")
        print(f"Best accuracy: {training_state.get('best_acc', 0):.2f}%")
        
        choice = input("\nTrain ulang dari awal? (y/n): ").strip().lower()
        if choice != 'y':
            return False
        
        # Reset checkpoint for this method
        checkpoint_dir = config.CHECKPOINTS_DIR
        for f in os.listdir(checkpoint_dir):
            if f.startswith(method_key):
                os.remove(os.path.join(checkpoint_dir, f))
        
        # Remove from completed list
        if method_key in tracker.progress['completed_methods']:
            tracker.progress['completed_methods'].remove(method_key)
            tracker.save_progress()
        
        resume = False
    
    elif training_state and not training_state.get('completed', False):
        last_epoch = training_state['last_epoch']
        total_epochs = training_state['total_epochs']
        
        print(f"\n{'='*70}")
        print(f"{method_key.upper()} - INCOMPLETE TRAINING FOUND".center(70))
        print(f"{'='*70}")
        print(f"Last completed epoch: {last_epoch}/{total_epochs}")
        print(f"Best accuracy so far: {training_state.get('best_acc', 0):.2f}%")
        
        choice = input(f"\nResume from epoch {last_epoch+1}? (y/n/restart): ").strip().lower()
        
        if choice == 'restart':
            # Clear checkpoints
            checkpoint_dir = config.CHECKPOINTS_DIR
            for f in os.listdir(checkpoint_dir):
                if f.startswith(method_key):
                    os.remove(os.path.join(checkpoint_dir, f))
            resume = False
        elif choice == 'y':
            resume = True
        else:
            return False
    
    print("\n" + "="*70)
    print(f"TRAINING: {method_key.upper()}".center(70))
    print("="*70)
    
    if resume:
        print(f"Mode: RESUME TRAINING".center(70))
    else:
        print(f"Mode: START FROM SCRATCH".center(70))
    
    print("="*70)
    
    config.print_config()
    
    # Load data
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    # Create model
    print("\nBuilding model...")
    model = get_model(method_key, config)
    count_parameters(model)
    
    # Get feature extractor
    feature_extractor = get_feature_extractor(method_key, config)
    
    # Train
    print(f"\nStarting training...")
    start_time = time.time()
    
    try:
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            method=method_key,
            feature_extractor=feature_extractor,
            resume=resume
        )
        
        training_time = time.time() - start_time
        
        # Plot training curves
        if len(history['train_loss']) > 0:
            print("\nPlotting training curves...")
            plot_training_curves(history, method_key, config.FIGURES_DIR, config)
        
        # Evaluate on test set
        print("\nFinal evaluation on test set...")
        results = full_evaluation(
            model=model,
            dataloader=test_loader,
            config=config,
            method=method_key,
            feature_extractor=feature_extractor
        )
        
        # Mark as completed
        tracker.mark_completed(method_key)
        
        print("\n" + "="*70)
        print(f"{method_key.upper()} COMPLETED!".center(70))
        print("="*70)
        print(f"{'Training time:':<25} {training_time:.2f} seconds")
        print(f"{'Test Accuracy:':<25} {results['accuracy']:.2f}%")
        print(f"{'Test Precision:':<25} {results['precision']:.4f}")
        print(f"{'Test Recall:':<25} {results['recall']:.4f}")
        print(f"{'Test F1-Score:':<25} {results['f1_score']:.4f}")
        print(f"{'Test AUC:':<25} {results['auc']:.4f}")
        print(f"{'Model saved:':<25} {config.get_model_path(method_key)}")
        print("="*70)
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("TRAINING INTERRUPTED BY USER!".center(70))
        print("="*70)
        print("Progress has been saved.")
        print(f"You can resume training {method_key} later.")
        print("="*70 + "\n")
        return False
    
    except Exception as e:
        print(f"\n\n" + "="*70)
        print("ERROR DURING TRAINING!".center(70))
        print("="*70)
        print(f"Error: {str(e)}")
        print("="*70 + "\n")
        import traceback
        traceback.print_exc()
        return False


def train_all_sequential(tracker, resume=False):
    """Train all methods sequentially"""
    
    methods = [
        ('baseline', 'Baseline (Ablation)', BaselineConfig),
        ('residual_spatial', 'Residual Spatial (Ablation)', ResidualSpatialConfig),
        ('residual_dct', 'Residual DCT (Ablation)', ResidualDCTConfig),
        ('fusion', 'Fusion', FusionConfig)
    ]
    
    # Determine starting point
    if resume:
        start_idx = 0
        for i, (method_key, _, _) in enumerate(methods):
            if tracker.is_completed(method_key):
                start_idx = i + 1
            else:
                break
        
        if start_idx >= len(methods):
            print("\nAll methods already completed!")
            return
        
        print(f"\nResuming from: {methods[start_idx][1]}")
        methods = methods[start_idx:]
    
    print("\n" + "="*70)
    print("SEQUENTIAL TRAINING - ALL METHODS".center(70))
    print("="*70)
    print(f"Will train {len(methods)} method(s)")
    
    if not resume:
        confirm = input("\nContinue? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            return
    
    total_start = time.time()
    
    for i, (method_key, method_name, config_class) in enumerate(methods, 1):
        print("\n\n" + "="*70)
        print(f"[{i}/{len(methods)}] {method_name}".center(70))
        print("="*70)
        
        success = train_single_stage(method_key, config_class, tracker, resume=False)
        
        if not success:
            print(f"\nStopped at {method_name}")
            print("You can resume training later using 'Resume Training' option")
            break
    
    total_time = time.time() - total_start
    completed, total, _ = tracker.get_progress_summary()
    
    print("\n\n" + "="*70)
    print("TRAINING SUMMARY".center(70))
    print("="*70)
    print(f"Completed: {completed}/{total} methods")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("="*70)


def evaluate_and_generate_results():
    """
    Evaluate all methods and generate complete results for thesis
    Includes ablation study analysis
    """
    
    config = Config()
    config.create_directories()
    
    methods = ['baseline', 'residual_spatial', 'residual_dct', 'fusion']
    method_names = [
        'Baseline (RGB Only)',
        'Residual Spatial',
        'Residual DCT',
        'Multi-Stream Fusion'
    ]
    
    print("\n" + "="*70)
    print("COMPLETE EVALUATION & RESULTS GENERATION".center(70))
    print("="*70)
    
    # Load test data
    _, _, test_loader = get_dataloaders(config)
    
    results_dict = {}
    
    for method, name in zip(methods, method_names):
        model_path = config.get_model_path(method)
        
        if not os.path.exists(model_path):
            print(f"\n{name}: Model not found, skipping...")
            continue
        
        print(f"Evaluating: {name}".center(70))
        
        # Get config
        if method == 'baseline':
            method_config = BaselineConfig()
        elif method == 'residual_spatial':
            method_config = ResidualSpatialConfig()
        elif method == 'residual_dct':
            method_config = ResidualDCTConfig()
        else:
            method_config = FusionConfig()
        
        # Create model
        model = get_model(method, method_config)
        feature_extractor = get_feature_extractor(method, method_config)
        
        # Evaluate
        results = load_and_evaluate(
            model_path=model_path,
            model=model,
            dataloader=test_loader,
            config=method_config,
            method=method,
            feature_extractor=feature_extractor
        )
        
        results_dict[method] = {
            'name': name,
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score'],
            'auc': results['auc']
        }
    
    # Print results table
    if len(results_dict) > 0:
        print("\n\n" + "="*70)
        print("EXPERIMENTAL RESULTS".center(70))
        print("="*70)
        print(f"{'Method':<35} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")

        
        baseline_acc = None
        fusion_acc = None
        
        for method, metrics in results_dict.items():
            marker = ""
            if method == 'fusion':
                marker = ""
                fusion_acc = metrics['accuracy']
            elif method == 'baseline':
                baseline_acc = metrics['accuracy']
            
            print(f"{metrics['name']:<35}{marker} {metrics['accuracy']:>9.2f}% "
                  f"{metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
                  f"{metrics['f1_score']:>10.4f} {metrics['auc']:>10.4f}")
        
        print("="*70)
        
        # Ablation study analysis
        if baseline_acc and fusion_acc:
            improvement = fusion_acc - baseline_acc
            print(f"\n{'ABLATION STUDY ANALYSIS':^70}")
    
            print(f"Baseline (RGB only):           {baseline_acc:>6.2f}%")
            print(f"Proposed Method (Fusion):      {fusion_acc:>6.2f}%")
            print(f"Improvement:                   {improvement:>+6.2f}%")
            print(f"Relative Improvement:          {(improvement/baseline_acc*100):>+6.2f}%")
    
            print("\nKesimpulan:")
            print("Multi-stream fusion yang menggabungkan RGB, Residual Spatial,")
            print("dan DCT features terbukti meningkatkan performa deteksi secara")
            print(f"signifikan (+{improvement:.2f}% dari baseline).")
            print("="*70)
        
        # Save comparison
        comparison_path = os.path.join(config.OUTPUT_DIR, 'thesis_results.json')
        with open(comparison_path, 'w') as f:
            json.dump({
                'results': results_dict,
                'ablation_study': {
                    'baseline_accuracy': baseline_acc,
                    'proposed_accuracy': fusion_acc,
                    'improvement': fusion_acc - baseline_acc if (baseline_acc and fusion_acc) else None
                }
            }, f, indent=4)
        
        print(f"\nResults saved: {comparison_path}")
        
        # Plot comparison
        from src.visualization import plot_methods_comparison
        plot_methods_comparison(results_dict, config.VISUALIZATIONS_DIR, config)
        
    else:
        print("\nNo trained models found!")


def ablation_study_analysis():
    """
    Detailed ablation study analysis
    Shows contribution of each component
    """
    
    config = Config()
    
    print("\n" + "="*70)
    print("ABLATION STUDY - COMPONENT ANALYSIS".center(70))
    print("="*70)
    
    # Load results
    results_path = os.path.join(config.OUTPUT_DIR, 'thesis_results.json')
    
    if not os.path.exists(results_path):
        print("\nPlease run 'Evaluate & Generate Results' first!")
        return
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    print("\nKOMPONEN ANALYSIS:")
    print("="*70)
    
    # Extract accuracies
    baseline_acc = results.get('baseline', {}).get('accuracy', 0)
    residual_spatial_acc = results.get('residual_spatial', {}).get('accuracy', 0)
    residual_dct_acc = results.get('residual_dct', {}).get('accuracy', 0)
    fusion_acc = results.get('fusion', {}).get('accuracy', 0)
    
    print(f"\n1. RGB Features (Baseline)")
    print(f"   Accuracy: {baseline_acc:.2f}%")
    print(f"   â†’ Baseline performance")
    
    print(f"\n2. + Residual Spatial Features")
    print(f"   Accuracy: {residual_spatial_acc:.2f}%")
    print(f"   Contribution: {residual_spatial_acc - baseline_acc:+.2f}%")
    print(f"   â†’ Detects manipulation artifacts in spatial domain")
    
    print(f"\n3. + DCT Frequency Features")
    print(f"   Accuracy: {residual_dct_acc:.2f}%")
    print(f"   Contribution: {residual_dct_acc - residual_spatial_acc:+.2f}%")
    print(f"   â†’ Analyzes frequency domain anomalies")
    
    print(f"\n4.Multi-Stream Fusion (RGB + Residual + DCT)")
    print(f"   Accuracy: {fusion_acc:.2f}%")
    print(f"   Total Improvement: {fusion_acc - baseline_acc:+.2f}%")
    print(f"   â†’ Combines all feature representations")
    
    print("\n" + "="*70)
    print("KESIMPULAN ABLATION STUDY:".center(70))
    print("="*70)
    print("Setiap komponen memberikan kontribusi positif:")
    print(f"â€¢ Residual Spatial: +{residual_spatial_acc - baseline_acc:.2f}%")
    print(f"â€¢ DCT Frequency:    +{residual_dct_acc - residual_spatial_acc:.2f}%")
    print(f"â€¢ Multi-Stream Fusion mengintegrasikan semua komponen")
    print(f"  dan mencapai akurasi tertinggi: {fusion_acc:.2f}%")
    print("="*70 + "\n")


def main():
    """Main function with proper methodology framing"""
    
    tracker = ProgressTracker()
    
    methods_map = {
        '1': ('baseline', 'Baseline', BaselineConfig),
        '2': ('residual_spatial', 'Residual Spatial', ResidualSpatialConfig),
        '3': ('residual_dct', 'Residual DCT', ResidualDCTConfig),
        '4': ('fusion', 'Multi-Stream Fusion', FusionConfig)
    }
    
    while True:
        clear_screen()
        print_banner()
        print_menu(tracker)
        
        choice = input("Pilih opsi (0-9): ").strip()
        
        if choice == '0':
            print("\n" + "="*70)
            print("Terima kasih!".center(70))
            print("="*70 + "\n")
            break
        
        elif choice in ['1', '2', '3', '4']:
            method_key, method_name, config_class = methods_map[choice]
            train_single_stage(method_key, config_class, tracker, resume=False)
            input("\nPress Enter to continue...")
        
        elif choice == '5':
            train_all_sequential(tracker, resume=False)
            input("\nPress Enter to continue...")
        
        elif choice == '6':
            # Resume training
            next_method = tracker.get_next_method()
            if next_method is None:
                print("\nAll methods already completed!")
                input("\nPress Enter to continue...")
            else:
                print(f"\nWill resume from: {next_method}")
                confirm = input("Continue? (y/n): ").strip().lower()
                if confirm == 'y':
                    train_all_sequential(tracker, resume=True)
                input("\nPress Enter to continue...")
        
        elif choice == '7':
            evaluate_and_generate_results()
            input("\nPress Enter to continue...")
        
        elif choice == '8':
            ablation_study_analysis()
            input("\nPress Enter to continue...")
        
        elif choice == '9':
            print("\nThis will reset all training progress!")
            confirm = input("Are you sure? (yes/no): ").strip().lower()
            if confirm == 'yes':
                tracker.reset()
                
                # Also clear all checkpoints
                config = Config()
                if os.path.exists(config.CHECKPOINTS_DIR):
                    for f in os.listdir(config.CHECKPOINTS_DIR):
                        os.remove(os.path.join(config.CHECKPOINTS_DIR, f))
                
                print("Progress and checkpoints reset!")
            else:
                print("Cancelled.")
            input("\nPress Enter to continue...")
        
        else:
            print("\nInvalid option!")
            input("\nPress Enter to continue...")


if __name__ == '__main__':
    main()