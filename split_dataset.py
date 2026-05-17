

import os
import random
import shutil
from tqdm import tqdm
import json
from datetime import datetime

IMSFD_DIR = r"D:\DEEP FAKE\data\IMSFD"
DEEPFAKE_DIR = r"D:\DEEP FAKE\data\IMSFD_DeepFake"

DEST_DIR = r"data"

SPLIT_RATIO = {
    "train": 0.70,
    "val":   0.15,
    "test":  0.15
}


MAX_PER_SUBJECT = None

MAX_TOTAL_REAL = None

RANDOM_SEED = 42

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')



def collect_imsfd_images(imsfd_dir, max_per_subject=None):

    all_images = []
    subject_stats = {}

    if not os.path.exists(imsfd_dir):
        raise FileNotFoundError(f"Folder IMSFD tidak ditemukan: {imsfd_dir}")

    grup_folders = sorted([
        d for d in os.listdir(imsfd_dir)
        if os.path.isdir(os.path.join(imsfd_dir, d))
    ])

    print(f"\nFolder grup ditemukan: {grup_folders}")

    for grup in grup_folders:
        grup_path = os.path.join(imsfd_dir, grup)

        subfolders = os.listdir(grup_path)
        has_split = any(s.lower() in ['training', 'testing'] for s in subfolders)

        if has_split:
            split_dirs = [
                os.path.join(grup_path, s)
                for s in subfolders
                if os.path.isdir(os.path.join(grup_path, s))
            ]
        else:
            split_dirs = [grup_path]

        for split_dir in split_dirs:
            if not os.path.isdir(split_dir):
                continue

            subject_dirs = [
                d for d in os.listdir(split_dir)
                if os.path.isdir(os.path.join(split_dir, d))
            ]

            for subject_id in subject_dirs:
                subject_path = os.path.join(split_dir, subject_id)
                images_in_subject = []

                for root, _, files in os.walk(subject_path):
                    for f in files:
                        if f.lower().endswith(IMG_EXT):
                            images_in_subject.append(
                                (os.path.join(root, f), subject_id)
                            )

                if max_per_subject and len(images_in_subject) > max_per_subject:
                    images_in_subject = random.sample(images_in_subject, max_per_subject)

                all_images.extend(images_in_subject)

                key = f"{grup}/{subject_id}"
                subject_stats[key] = subject_stats.get(key, 0) + len(images_in_subject)

    return all_images, subject_stats


def collect_deepfake_images(deepfake_dir):

    all_images = []

    if not deepfake_dir or not os.path.exists(deepfake_dir):
        print(f"\n[WARNING] Folder deepfake tidak ditemukan: {deepfake_dir}")
        print("Lanjutkan hanya dengan data real (belum bisa training)")
        return []

    for root, _, files in os.walk(deepfake_dir):
        for f in files:
            if f.lower().endswith(IMG_EXT):
                all_images.append(os.path.join(root, f))

    return all_images


def split_and_copy(image_list, label, dest_dir, split_ratio, desc=""):

    random.shuffle(image_list)
    total = len(image_list)

    train_end = int(total * split_ratio['train'])
    val_end   = train_end + int(total * split_ratio['val'])

    splits = {
        'train': image_list[:train_end],
        'val':   image_list[train_end:val_end],
        'test':  image_list[val_end:]
    }

    stats = {}
    for split_name, files in splits.items():
        split_dir = os.path.join(dest_dir, split_name, label)
        os.makedirs(split_dir, exist_ok=True)

        copied = 0
        name_counter = {}

        for item in tqdm(files, desc=f"  {desc} [{split_name}]"):
            src = item[0] if isinstance(item, tuple) else item
            base_name = os.path.basename(src)

            if base_name in name_counter:
                name_counter[base_name] += 1
                name, ext = os.path.splitext(base_name)
                base_name = f"{name}_{name_counter[base_name]}{ext}"
            else:
                name_counter[base_name] = 0

            dst = os.path.join(split_dir, base_name)

            try:
                shutil.copy2(src, dst)
                copied += 1
            except Exception as e:
                print(f"  [ERROR] Gagal copy {src}: {e}")

        stats[split_name] = copied

    return stats


def print_summary(real_stats, fake_stats, subject_stats):
    """Print ringkasan hasil split"""
    print("\n" + "=" * 65)
    print("RINGKASAN SPLIT DATASET".center(65))
    print("=" * 65)

    total_real = sum(real_stats.values())
    total_fake = sum(fake_stats.values())

    print(f"\n{'Split':<10} {'Real':>10} {'Fake':>10} {'Total':>10}")
    print("-" * 45)
    for split in ['train', 'val', 'test']:
        r = real_stats.get(split, 0)
        f = fake_stats.get(split, 0)
        print(f"{split:<10} {r:>10,} {f:>10,} {r+f:>10,}")
    print("-" * 45)
    print(f"{'TOTAL':<10} {total_real:>10,} {total_fake:>10,} {total_real+total_fake:>10,}")

    print(f"\nJumlah subjek unik: {len(subject_stats)}")
    print(f"Rata-rata foto/subjek: {total_real/len(subject_stats):.1f}" if subject_stats else "")

    if total_fake == 0:
        print("\n[!] PERHATIAN: Belum ada data FAKE!")
        print("    Jalankan generate_deepfake.py terlebih dahulu,")
        print("    lalu jalankan split_dataset.py ini kembali.")

    print("=" * 65)


def save_metadata(dest_dir, real_stats, fake_stats, subject_stats, config_used):
    """Simpan metadata split ke JSON untuk referensi"""
    meta = {
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': config_used,
        'real_stats': real_stats,
        'fake_stats': fake_stats,
        'num_subjects': len(subject_stats),
        'subject_distribution': subject_stats
    }
    meta_path = os.path.join(dest_dir, 'dataset_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=4)
    print(f"\nMetadata tersimpan: {meta_path}")


def main():
    random.seed(RANDOM_SEED)

    print("=" * 65)
    print("IMSFD DATASET SPLITTER".center(65))
    print("=" * 65)
    print(f"\nSource IMSFD : {IMSFD_DIR}")
    print(f"Source Fake  : {DEEPFAKE_DIR}")
    print(f"Destination  : {DEST_DIR}")
    print(f"Split Ratio  : train={SPLIT_RATIO['train']:.0%} | "
          f"val={SPLIT_RATIO['val']:.0%} | test={SPLIT_RATIO['test']:.0%}")

    print("\n[1/4] Mengumpulkan gambar REAL dari IMSFD...")
    real_images, subject_stats = collect_imsfd_images(
        IMSFD_DIR,
        max_per_subject=MAX_PER_SUBJECT
    )
    print(f"      Total gambar real ditemukan: {len(real_images):,}")
    print(f"      Total subjek unik          : {len(subject_stats):,}")

    if MAX_TOTAL_REAL and len(real_images) > MAX_TOTAL_REAL:
        real_images = random.sample(real_images, MAX_TOTAL_REAL)
        print(f"      Dibatasi ke: {len(real_images):,} gambar")

    print("\n[2/4] Mengumpulkan gambar FAKE (deepfake)...")
    fake_images = collect_deepfake_images(DEEPFAKE_DIR)
    print(f"      Total gambar fake ditemukan: {len(fake_images):,}")

    if fake_images and len(real_images) != len(fake_images):
        min_count = min(len(real_images), len(fake_images))
        print(f"\n      Balancing dataset: ambil {min_count:,} dari masing-masing kelas")
        real_images = random.sample(real_images, min_count)
        fake_images = random.sample(fake_images, min_count)

    print("\n[3/4] Membuat struktur folder output...")
    for split in SPLIT_RATIO:
        for cls in ['real', 'fake']:
            os.makedirs(os.path.join(DEST_DIR, split, cls), exist_ok=True)
    print(f"      Folder output: {os.path.abspath(DEST_DIR)}")

    print("\n[4/4] Split dan copy gambar...")

    print("\n Memproses data REAL:")
    real_stats = split_and_copy(real_images, 'real', DEST_DIR, SPLIT_RATIO, desc="REAL")

    fake_stats = {'train': 0, 'val': 0, 'test': 0}
    if fake_images:
        print("\n Memproses data FAKE:")
        fake_stats = split_and_copy(fake_images, 'fake', DEST_DIR, SPLIT_RATIO, desc="FAKE")

    print_summary(real_stats, fake_stats, subject_stats)

    config_used = {
        'imsfd_dir': IMSFD_DIR,
        'deepfake_dir': DEEPFAKE_DIR,
        'dest_dir': DEST_DIR,
        'split_ratio': SPLIT_RATIO,
        'max_per_subject': MAX_PER_SUBJECT,
        'max_total_real': MAX_TOTAL_REAL,
        'random_seed': RANDOM_SEED
    }
    save_metadata(DEST_DIR, real_stats, fake_stats, subject_stats, config_used)

    print("\nDataset berhasil di-split!")
    print(f"Selanjutnya jalankan: python main.py\n")


if __name__ == '__main__':
    main()