import argparse
import os
import sys
import time
import json

from config.config import (
    Config, BaselineConfig, ResidualSpatialConfig, ResidualDCTConfig
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

    def __init__(self, progress_file='training_progress.json'):
        self.progress_file = progress_file
        self.progress = self.load_progress()

    def load_progress(self):
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {'completed_methods': [], 'last_run': None}

    def save_progress(self):
        self.progress['last_run'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=4)

    def mark_completed(self, method):
        if method not in self.progress['completed_methods']:
            self.progress['completed_methods'].append(method)
        self.save_progress()

    def is_completed(self, method):
        return method in self.progress['completed_methods']

    def get_next_method(self):
        for method in ['baseline', 'residual_spatial', 'residual_dct']:
            if not self.is_completed(method):
                return method
        return None

    def reset(self):
        self.progress = {'completed_methods': [], 'last_run': None}
        self.save_progress()

    def get_progress_summary(self):
        all_methods = ['baseline', 'residual_spatial', 'residual_dct']
        completed = len(self.progress['completed_methods'])
        total = len(all_methods)
        return completed, total, self.progress['completed_methods']


def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')


def print_banner():
    print("\n" + "="*70)
    print("DETEKSI DEEPFAKE BERBASIS RESIDUAL NOISE".center(70))
    print("Perbandingan Fitur Domain Spasial dan DCT Menggunakan CNN".center(70))
    print("="*70)


def print_menu(tracker):
    completed, total, completed_list = tracker.get_progress_summary()

    print("TAHAPAN PENELITIAN".center(70))

    methods = [
        ('baseline',         '1. Baseline (RGB)                    — Pembanding'),
        ('residual_spatial', '2. Residual Spatial                  — Fitur Domain Spasial'),
        ('residual_dct',     '3. Residual DCT                      — Fitur Domain Frekuensi'),
    ]

    for method, label in methods:
        status = "✓" if tracker.is_completed(method) else " "
        print(f"  [{status}] {label}")

    print(f"\n  [4] Train Semua Model (Sekuensial)")
    print(f"  [5] Resume Training")
    print(f"  [6] Evaluasi & Generate Hasil")
    print(f"  [7] Analisis Perbandingan Fitur")
    print(f"  [8] Reset Progress")
    print(f"  [0] Keluar")
    print(f"\nProgress: {completed}/{total} model selesai")
    if tracker.progress['last_run']:
        print(f"Terakhir dijalankan: {tracker.progress['last_run']}")
    print()


def check_training_state(method_key, config_class):
    config = config_class()
    state_path = os.path.join(config.CHECKPOINTS_DIR, f'{method_key}_training_state.json')
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            return json.load(f)
    return None


def train_single_stage(method_key, config_class, tracker, resume=False):

    config = config_class()
    config.create_directories()
    set_seed(42)

    training_state = check_training_state(method_key, config_class)

    if training_state and training_state.get('completed', False):
        print(f"\n{'='*70}")
        print(f"{method_key.upper()} - SUDAH SELESAI".center(70))
        print(f"{'='*70}")
        print(f"Best accuracy: {training_state.get('best_acc', 0):.2f}%")

        choice = input("\nTrain ulang dari awal? (y/n): ").strip().lower()
        if choice != 'y':
            return False

        checkpoint_dir = config.CHECKPOINTS_DIR
        for f in os.listdir(checkpoint_dir):
            if f.startswith(method_key):
                os.remove(os.path.join(checkpoint_dir, f))

        if method_key in tracker.progress['completed_methods']:
            tracker.progress['completed_methods'].remove(method_key)
            tracker.save_progress()

        resume = False

    elif training_state and not training_state.get('completed', False):
        last_epoch = training_state['last_epoch']
        total_epochs = training_state['total_epochs']

        print(f"\n{'='*70}")
        print(f"{method_key.upper()} - TRAINING BELUM SELESAI".center(70))
        print(f"{'='*70}")
        print(f"Epoch terakhir: {last_epoch}/{total_epochs}")
        print(f"Best accuracy: {training_state.get('best_acc', 0):.2f}%")

        choice = input(f"\nLanjut dari epoch {last_epoch+1}? (y/n/restart): ").strip().lower()

        if choice == 'restart':
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

    config.print_config()

    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    print("\nBuilding model...")
    model = get_model(method_key, config)
    count_parameters(model)

    feature_extractor = get_feature_extractor(method_key, config)

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

        if len(history['train_loss']) > 0:
            plot_training_curves(history, method_key, config.FIGURES_DIR, config)

        print("\nEvaluasi pada test set...")
        results = full_evaluation(
            model=model,
            dataloader=test_loader,
            config=config,
            method=method_key,
            feature_extractor=feature_extractor
        )

        tracker.mark_completed(method_key)

        print("\n" + "="*70)
        print(f"{method_key.upper()} SELESAI!".center(70))
        print("="*70)
        print(f"{'Waktu training:':<25} {training_time:.2f} detik")
        print(f"{'Test Accuracy:':<25} {results['accuracy']:.2f}%")
        print(f"{'Test Precision:':<25} {results['precision']:.4f}")
        print(f"{'Test Recall:':<25} {results['recall']:.4f}")
        print(f"{'Test F1-Score:':<25} {results['f1_score']:.4f}")
        print(f"{'Test AUC:':<25} {results['auc']:.4f}")
        print("="*70)

        return True

    except KeyboardInterrupt:
        print("\n\nTraining dihentikan. Progress tersimpan.")
        return False

    except Exception as e:
        print(f"\n\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def train_all_sequential(tracker, resume=False):

    methods = [
        ('baseline',         'Baseline (RGB)',        BaselineConfig),
        ('residual_spatial', 'Residual Spatial',      ResidualSpatialConfig),
        ('residual_dct',     'Residual DCT',          ResidualDCTConfig),
    ]

    if resume:
        start_idx = 0
        for i, (method_key, _, _) in enumerate(methods):
            if tracker.is_completed(method_key):
                start_idx = i + 1
            else:
                break

        if start_idx >= len(methods):
            print("\nSemua model sudah selesai ditraining!")
            return

        methods = methods[start_idx:]

    print("\n" + "="*70)
    print("TRAINING SEMUA MODEL — SEKUENSIAL".center(70))
    print("="*70)
    print(f"Akan melatih {len(methods)} model")

    if not resume:
        confirm = input("\nLanjutkan? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Dibatalkan.")
            return

    total_start = time.time()

    for i, (method_key, method_name, config_class) in enumerate(methods, 1):
        print("\n\n" + "="*70)
        print(f"[{i}/{len(methods)}] {method_name}".center(70))
        print("="*70)

        success = train_single_stage(method_key, config_class, tracker, resume=False)

        if not success:
            print(f"\nBerhenti di {method_name}. Resume nanti dengan opsi [5].")
            break

    total_time = time.time() - total_start
    completed, total, _ = tracker.get_progress_summary()

    print("\n\n" + "="*70)
    print("RINGKASAN TRAINING".center(70))
    print("="*70)
    print(f"Selesai: {completed}/{total} model")
    print(f"Total waktu: {total_time/60:.2f} menit")
    print("="*70)


def evaluate_and_generate_results():

    config = Config()
    config.create_directories()

    methods = [
        ('baseline',         'Baseline (RGB Only)',              BaselineConfig),
        ('residual_spatial', 'Residual Noise — Domain Spasial',  ResidualSpatialConfig),
        ('residual_dct',     'Residual Noise — Domain DCT',      ResidualDCTConfig),
    ]

    print("\n" + "="*70)
    print("EVALUASI & GENERATE HASIL".center(70))
    print("="*70)

    _, _, test_loader = get_dataloaders(config)

    results_dict = {}

    for method, name, config_class in methods:
        model_path = config.get_model_path(method)

        if not os.path.exists(model_path):
            print(f"\n{name}: Model tidak ditemukan, dilewati...")
            continue

        print(f"\nEvaluasi: {name}")

        method_config = config_class()
        model = get_model(method, method_config)
        feature_extractor = get_feature_extractor(method, method_config)

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
            'accuracy':  results['accuracy'],
            'precision': results['precision'],
            'recall':    results['recall'],
            'f1_score':  results['f1_score'],
            'auc':       results['auc']
        }

    if not results_dict:
        print("\nTidak ada model yang ditemukan!")
        return

    # Tabel hasil
    print("\n\n" + "="*70)
    print("HASIL EKSPERIMEN".center(70))
    print("="*70)
    print(f"{'Model':<38} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print("─"*70)

    for method, m in results_dict.items():
        print(f"{m['name']:<38} {m['accuracy']:>8.2f}% "
              f"{m['precision']:>10.4f} {m['recall']:>8.4f} "
              f"{m['f1_score']:>8.4f} {m['auc']:>8.4f}")

    print("="*70)

    # Analisis perbandingan fitur
    spatial_acc = results_dict.get('residual_spatial', {}).get('accuracy')
    dct_acc     = results_dict.get('residual_dct', {}).get('accuracy')
    baseline_acc = results_dict.get('baseline', {}).get('accuracy')

    if spatial_acc and dct_acc:
        print(f"\n{'ANALISIS PERBANDINGAN REPRESENTASI FITUR':^70}")
        print("─"*70)
        if baseline_acc:
            print(f"  Baseline (RGB)          : {baseline_acc:.2f}%  — pembanding")
        print(f"  Residual Spasial        : {spatial_acc:.2f}%  — domain spasial")
        print(f"  Residual DCT            : {dct_acc:.2f}%  — domain frekuensi")
        diff = spatial_acc - dct_acc
        print(f"\n  Selisih Spasial vs DCT  : {diff:+.2f}%")
        better = "Domain Spasial" if diff > 0 else "Domain DCT"
        print(f"  Representasi lebih baik : {better}")
        print("─"*70)

    # Simpan hasil
    comparison_path = os.path.join(config.OUTPUT_DIR, 'thesis_results.json')
    with open(comparison_path, 'w') as f:
        json.dump({
            'judul': 'Deteksi Deepfake Berbasis Residual Noise: Perbandingan Representasi Fitur Domain Spasial dan DCT Menggunakan CNN',
            'results': results_dict,
            'feature_comparison': {
                'baseline_accuracy':        baseline_acc,
                'residual_spatial_accuracy': spatial_acc,
                'residual_dct_accuracy':     dct_acc,
                'spatial_vs_dct_diff':       round(spatial_acc - dct_acc, 4) if (spatial_acc and dct_acc) else None,
            }
        }, f, indent=4)

    print(f"\nHasil disimpan: {comparison_path}")

    from src.visualization import plot_methods_comparison
    plot_methods_comparison(results_dict, config.VISUALIZATIONS_DIR, config)


def feature_comparison_analysis():

    config = Config()

    print("\n" + "="*70)
    print("ANALISIS PERBANDINGAN REPRESENTASI FITUR".center(70))
    print("="*70)

    results_path = os.path.join(config.OUTPUT_DIR, 'thesis_results.json')

    if not os.path.exists(results_path):
        print("\nJalankan 'Evaluasi & Generate Hasil' terlebih dahulu!")
        return

    with open(results_path, 'r') as f:
        data = json.load(f)

    results = data['results']

    baseline_acc  = results.get('baseline', {}).get('accuracy', 0)
    spatial_acc   = results.get('residual_spatial', {}).get('accuracy', 0)
    dct_acc       = results.get('residual_dct', {}).get('accuracy', 0)

    baseline_f1   = results.get('baseline', {}).get('f1_score', 0)
    spatial_f1    = results.get('residual_spatial', {}).get('f1_score', 0)
    dct_f1        = results.get('residual_dct', {}).get('f1_score', 0)

    baseline_auc  = results.get('baseline', {}).get('auc', 0)
    spatial_auc   = results.get('residual_spatial', {}).get('auc', 0)
    dct_auc       = results.get('residual_dct', {}).get('auc', 0)

    print("\n" + "─"*70)
    print(f"  {'Model':<35} {'Accuracy':>9} {'F1':>8} {'AUC':>8}")
    print("─"*70)
    print(f"  {'Baseline (RGB)':<35} {baseline_acc:>8.2f}% {baseline_f1:>8.4f} {baseline_auc:>8.4f}")
    print(f"  {'Residual Noise — Domain Spasial':<35} {spatial_acc:>8.2f}% {spatial_f1:>8.4f} {spatial_auc:>8.4f}")
    print(f"  {'Residual Noise — Domain DCT':<35} {dct_acc:>8.2f}% {dct_f1:>8.4f} {dct_auc:>8.4f}")
    print("─"*70)

    print("\n  TEMUAN UTAMA:")
    print("─"*70)

    # Spasial vs baseline
    diff_s_b = spatial_acc - baseline_acc
    print(f"  • Residual Spasial vs Baseline    : {diff_s_b:+.2f}%")

    # DCT vs baseline
    diff_d_b = dct_acc - baseline_acc
    print(f"  • Residual DCT vs Baseline        : {diff_d_b:+.2f}%")

    # Spasial vs DCT
    diff_s_d = spatial_acc - dct_acc
    print(f"  • Residual Spasial vs Residual DCT: {diff_s_d:+.2f}%")

    print("\n  KESIMPULAN:")
    if spatial_acc > dct_acc:
        print(f"  Representasi fitur domain SPASIAL lebih efektif ({spatial_acc:.2f}%)")
        print(f"  dibanding domain DCT ({dct_acc:.2f}%) untuk deteksi deepfake")
        print(f"  pada dataset ini, dengan selisih {abs(diff_s_d):.2f}%.")
    else:
        print(f"  Representasi fitur domain DCT lebih efektif ({dct_acc:.2f}%)")
        print(f"  dibanding domain spasial ({spatial_acc:.2f}%) untuk deteksi deepfake")
        print(f"  pada dataset ini, dengan selisih {abs(diff_s_d):.2f}%.")

    print("="*70 + "\n")


def main():

    tracker = ProgressTracker()

    methods_map = {
        '1': ('baseline',         'Baseline',        BaselineConfig),
        '2': ('residual_spatial', 'Residual Spatial', ResidualSpatialConfig),
        '3': ('residual_dct',     'Residual DCT',     ResidualDCTConfig),
    }

    while True:
        clear_screen()
        print_banner()
        print_menu(tracker)

        choice = input("Pilih opsi (0-8): ").strip()

        if choice == '0':
            print("\nTerima kasih!\n")
            break

        elif choice in ['1', '2', '3']:
            method_key, method_name, config_class = methods_map[choice]
            train_single_stage(method_key, config_class, tracker, resume=False)
            input("\nTekan Enter untuk melanjutkan...")

        elif choice == '4':
            train_all_sequential(tracker, resume=False)
            input("\nTekan Enter untuk melanjutkan...")

        elif choice == '5':
            next_method = tracker.get_next_method()
            if next_method is None:
                print("\nSemua model sudah selesai!")
                input("\nTekan Enter untuk melanjutkan...")
            else:
                print(f"\nAkan melanjutkan dari: {next_method}")
                confirm = input("Lanjutkan? (y/n): ").strip().lower()
                if confirm == 'y':
                    train_all_sequential(tracker, resume=True)
                input("\nTekan Enter untuk melanjutkan...")

        elif choice == '6':
            evaluate_and_generate_results()
            input("\nTekan Enter untuk melanjutkan...")

        elif choice == '7':
            feature_comparison_analysis()
            input("\nTekan Enter untuk melanjutkan...")

        elif choice == '8':
            print("\nIni akan mereset semua progress training!")
            confirm = input("Yakin? (yes/no): ").strip().lower()
            if confirm == 'yes':
                tracker.reset()
                config = Config()
                if os.path.exists(config.CHECKPOINTS_DIR):
                    for f in os.listdir(config.CHECKPOINTS_DIR):
                        os.remove(os.path.join(config.CHECKPOINTS_DIR, f))
                print("Progress dan checkpoint direset!")
            else:
                print("Dibatalkan.")
            input("\nTekan Enter untuk melanjutkan...")

        else:
            print("\nOpsi tidak valid!")
            input("\nTekan Enter untuk melanjutkan...")


if __name__ == '__main__':
    main()
