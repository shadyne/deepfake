
import os
import random
import shutil
from tqdm import tqdm
import json
from datetime import datetime

IMSFD_DIR    = r"D:\Project\deepfake\data\IMSFD"
DEEPFAKE_DIR = r"D:\Project\deepfake\data\IMSFD_DeepFake"
DEST_DIR     = "data"

SPLIT_RATIO = {"train": 0.70, "val": 0.15, "test": 0.15}

MAX_PER_SUBJECT = None   # None = pakai semua foto
RANDOM_SEED     = 42
IMG_EXT         = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def collect_subjects(imsfd_dir, max_per_subject=None):

    subjects = {}

    if not os.path.exists(imsfd_dir):
        print(f"\n[ERROR] Folder IMSFD tidak ditemukan: {imsfd_dir}")
        print("  Pastikan IMSFD_DIR sudah benar!")
        return {}

    grup_list = sorted([
        d for d in os.listdir(imsfd_dir)
        if os.path.isdir(os.path.join(imsfd_dir, d))
    ])

    if not grup_list:
        print(f"\n[ERROR] Tidak ada subfolder di: {imsfd_dir}")
        return {}

    print(f"  Grup ditemukan : {grup_list}")

    for grup in grup_list:
        grup_path = os.path.join(imsfd_dir, grup)
        subfolders = os.listdir(grup_path)

        has_split = any(
            s.lower() in ['training', 'testing']
            for s in subfolders
            if os.path.isdir(os.path.join(grup_path, s))
        )

        candidate_dirs = (
            [os.path.join(grup_path, s)
             for s in subfolders
             if os.path.isdir(os.path.join(grup_path, s))]
            if has_split else [grup_path]
        )

        for cand_dir in candidate_dirs:
            for subject_id in os.listdir(cand_dir):
                subject_path = os.path.join(cand_dir, subject_id)
                if not os.path.isdir(subject_path):
                    continue

                images = []
                for root, _, files in os.walk(subject_path):
                    for f in files:
                        if f.lower().endswith(IMG_EXT):
                            images.append(os.path.join(root, f))

                if not images:
                    continue

                if max_per_subject and len(images) > max_per_subject:
                    images = random.sample(images, max_per_subject)


                key = f"{grup}__{subject_id}"

                if key in subjects:      
                    subjects[key].extend(images)
                else:
                    subjects[key] = images

    return subjects

def split_subjects(subjects, split_ratio, random_seed=42):
    all_keys = sorted(subjects.keys())
    rng = random.Random(random_seed)
    rng.shuffle(all_keys)

    n       = len(all_keys)
    n_train = int(n * split_ratio['train'])
    n_val   = round(n * split_ratio['val'])  
    n_test  = n - n_train - n_val

    if n >= 3:
        n_train = max(1, n_train)
        n_val   = max(1, n_val)
        n_test  = n - n_train - n_val

    split_map = {
        'train': all_keys[:n_train],
        'val':   all_keys[n_train : n_train + n_val],
        'test':  all_keys[n_train + n_val :],
    }

    print(f"\n  Pembagian subjek (total {n}):")
    for name, keys in split_map.items():
        n_imgs = sum(len(subjects[k]) for k in keys)
        print(f"    {name:<8}: {len(keys):>4} subjek  |  {n_imgs:>7,} gambar")

    return split_map

def copy_real_images(subjects, split_map, dest_dir):
    stats = {}

    for split_name, subject_keys in split_map.items():
        out_dir = os.path.join(dest_dir, split_name, 'real')
        os.makedirs(out_dir, exist_ok=True)

        if not subject_keys:
            print(f"  [WARNING] {split_name}: tidak ada subjek yang di-assign!")
            stats[split_name] = 0
            continue

        copied = 0
        failed = 0

        for key in tqdm(subject_keys, desc=f"  REAL [{split_name}]"):
            if key not in subjects:
                print(f"\n  [BUG] Key tidak ada di subjects: {key!r}")
                continue

            safe_key = key.replace(os.sep, '_').replace('/', '_').replace('\\', '_')

            for src_path in subjects[key]:
                if not os.path.isfile(src_path):
                    failed += 1
                    continue

                filename  = f"{safe_key}__{os.path.basename(src_path)}"
                dst_path  = os.path.join(out_dir, filename)

                try:
                    shutil.copy2(src_path, dst_path)
                    copied += 1
                except Exception as e:
                    failed += 1
                    print(f"\n  [ERROR] Gagal copy {src_path}: {e}")

        actual = len([
            f for f in os.listdir(out_dir)
            if f.lower().endswith(IMG_EXT)
        ])
        ok = "[OK]" if actual == copied else "tidak sesuai"
        print(f"  {split_name}/real : {copied:,} disalin  |  {failed} gagal  |"
              f"  di folder: {actual:,}  [{ok}]")

        stats[split_name] = copied

    return stats


def copy_fake_images(deepfake_dir, dest_dir, split_ratio, random_seed=42):

    if not deepfake_dir or not os.path.exists(deepfake_dir):
        print(f"\n  [INFO] Folder fake tidak ada: {deepfake_dir}")
        print("  Jalankan generate_deepfake.py dulu.")
        return {'train': 0, 'val': 0, 'test': 0}

    fakes = []
    for root, _, files in os.walk(deepfake_dir):
        for f in files:
            if f.lower().endswith(IMG_EXT):
                fakes.append(os.path.join(root, f))

    if not fakes:
        print("  [INFO] Tidak ada gambar fake ditemukan.")
        return {'train': 0, 'val': 0, 'test': 0}

    print(f"  Total fake: {len(fakes):,}")

    rng = random.Random(random_seed)
    rng.shuffle(fakes)

    n       = len(fakes)
    n_train = int(n * split_ratio['train'])
    n_val   = round(n * split_ratio['val'])

    buckets = {
        'train': fakes[:n_train],
        'val':   fakes[n_train : n_train + n_val],
        'test':  fakes[n_train + n_val :],
    }

    stats = {}
    for split_name, files in buckets.items():
        out_dir = os.path.join(dest_dir, split_name, 'fake')
        os.makedirs(out_dir, exist_ok=True)
        copied = 0

        for idx, src_path in enumerate(tqdm(files, desc=f"  FAKE [{split_name}]")):
            if not os.path.isfile(src_path):
                continue
            filename = f"fake_{idx:07d}__{os.path.basename(src_path)}"
            dst_path = os.path.join(out_dir, filename)
            try:
                shutil.copy2(src_path, dst_path)
                copied += 1
            except Exception as e:
                print(f"\n  [ERROR] {src_path}: {e}")

        actual = len([f for f in os.listdir(out_dir) if f.lower().endswith(IMG_EXT)])
        print(f"  {split_name}/fake : {copied:,} disalin  |  di folder: {actual:,}")
        stats[split_name] = copied

    return stats


def balance_splits(dest_dir, random_seed=42):
    rng = random.Random(random_seed)

    for split in ['train', 'val', 'test']:
        real_dir = os.path.join(dest_dir, split, 'real')
        fake_dir = os.path.join(dest_dir, split, 'fake')

        if not (os.path.exists(real_dir) and os.path.exists(fake_dir)):
            continue

        real_files = [f for f in os.listdir(real_dir) if f.lower().endswith(IMG_EXT)]
        fake_files = [f for f in os.listdir(fake_dir) if f.lower().endswith(IMG_EXT)]

        n_real, n_fake = len(real_files), len(fake_files)
        if n_real == 0 or n_fake == 0 or n_real == n_fake:
            continue

        if n_real > n_fake:
            excess, excess_dir = rng.sample(real_files, n_real - n_fake), real_dir
        else:
            excess, excess_dir = rng.sample(fake_files, n_fake - n_real), fake_dir

        for f in excess:
            os.remove(os.path.join(excess_dir, f))

        kept = min(n_real, n_fake)
        print(f"  [{split}] balanced → {kept:,} real | {kept:,} fake")


def count_final(dest_dir):
    counts = {}
    for split in ['train', 'val', 'test']:
        counts[split] = {}
        for cls in ['real', 'fake']:
            d = os.path.join(dest_dir, split, cls)
            counts[split][cls] = (
                len([f for f in os.listdir(d) if f.lower().endswith(IMG_EXT)])
                if os.path.exists(d) else 0
            )
    return counts


def print_summary(counts, split_map):
    print("\n" + "=" * 65)
    print("HASIL AKHIR SPLIT DATASET".center(65))
    print("=" * 65)
    print(f"\n  {'Split':<10} {'Real':>8} {'Fake':>8} {'Total':>8} {'Subjek':>8}")
    print("  " + "-" * 45)
    for split in ['train', 'val', 'test']:
        r = counts[split].get('real', 0)
        f = counts[split].get('fake', 0)
        s = len(split_map.get(split, []))
        print(f"  {split:<10} {r:>8,} {f:>8,} {r+f:>8,} {s:>8}")
    print("=" * 65)
    print("\n  ✓ Setiap subjek hanya ada di SATU split (tidak ada identity leakage)")
    no_fake = any(counts[s]['fake'] == 0 for s in ['train','val','test'])
    if no_fake:
        print("Fake belum tersedia. Jalankan generate_deepfake.py lalu ulang script ini.")

def main():
    random.seed(RANDOM_SEED)

    print("=" * 65)
    print("IMSFD DATASET SPLITTER  (FIXED v3)".center(65))
    print("=" * 65)
    print(f"\n  IMSFD_DIR    : {IMSFD_DIR}")
    print(f"  DEEPFAKE_DIR : {DEEPFAKE_DIR}")
    print(f"  DEST_DIR     : {DEST_DIR}")
    print(f"  Strategy     : Subject-level split (no identity leakage)")

    print("\n" + "─" * 65)
    print("[1/5] Scanning IMSFD ...")
    subjects = collect_subjects(IMSFD_DIR, max_per_subject=MAX_PER_SUBJECT)

    if not subjects:
        print("\n[ABORT] Tidak ada subjek. Cek IMSFD_DIR!")
        return

    total_real = sum(len(v) for v in subjects.values())
    print(f"{len(subjects):,} subjek  |  {total_real:,} gambar real")

    print("\n" + "─" * 65)
    print("[2/5] Membagi subjek ...")
    split_map = split_subjects(subjects, SPLIT_RATIO, random_seed=RANDOM_SEED)

    print("\n" + "─" * 65)
    print("[3/5] Membuat folder output ...")
    for split in SPLIT_RATIO:
        for cls in ['real', 'fake']:
            os.makedirs(os.path.join(DEST_DIR, split, cls), exist_ok=True)
    print(f"  Folder: {os.path.abspath(DEST_DIR)}")

    print("\n" + "─" * 65)
    print("[4/5] Menyalin gambar REAL ...")
    copy_real_images(subjects, split_map, DEST_DIR)

    print("\n" + "─" * 65)
    print("[5/5] Menyalin gambar FAKE ...")
    copy_fake_images(DEEPFAKE_DIR, DEST_DIR, SPLIT_RATIO, random_seed=RANDOM_SEED)

    print("\n" + "─" * 65)
    print("Balancing ...")
    balance_splits(DEST_DIR, random_seed=RANDOM_SEED)

    counts = count_final(DEST_DIR)
    print_summary(counts, split_map)

    meta = {
        'created_at':     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'split_strategy': 'subject-level v3',
        'config': {
            'imsfd_dir':    IMSFD_DIR,
            'deepfake_dir': DEEPFAKE_DIR,
            'dest_dir':     DEST_DIR,
            'split_ratio':  SPLIT_RATIO,
            'random_seed':  RANDOM_SEED,
        },
        'subjects': {
            'total': len(subjects),
            'subject_keys': {k: list(v) for k, v in split_map.items()},
        },
        'counts': counts,
    }
    meta_path = os.path.join(DEST_DIR, 'dataset_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=4)
    print(f"\n  Metadata: {meta_path}")
    print("  Selanjutnya: python generate_deepfake.py\n")


if __name__ == '__main__':
    main()