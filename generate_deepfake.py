
import os
import cv2
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

IMSFD_DIR       = r"D:\Project\deepfake\data\IMSFD"
OUTPUT_DIR      = r"D:\Project\deepfake\data\IMSFD_DeepFake"   
DATA_DIR        = "data"                                 
METADATA_PATH   = os.path.join(DATA_DIR, "dataset_metadata.json")

NUM_PAIRS       = 5000
SWAPS_PER_PAIR  = 3
OUTPUT_SIZE     = (224, 224)
RANDOM_SEED     = 42
IMG_EXT         = ('.jpg', '.jpeg', '.png', '.bmp')

FAKE_PER_SPLIT  = {
    'train': None,  
    'val':   None,
    'test':  None,
}


def check_dependencies():
    missing = []
    try:
        import insightface        # noqa
    except ImportError:
        missing.append('insightface')
    try:
        import onnxruntime        # noqa
    except ImportError:
        missing.append('onnxruntime')
    if missing:
        print(f"\n[ERROR] Install dulu: pip install {' '.join(missing)}")
        return False
    return True


def collect_subjects_from_imsfd(imsfd_dir, max_per_subject=None):
    subjects = {}
    for grup in sorted(os.listdir(imsfd_dir)):
        grup_path = os.path.join(imsfd_dir, grup)
        if not os.path.isdir(grup_path):
            continue
        subfolders = os.listdir(grup_path)
        has_split  = any(s.lower() in ['training', 'testing'] for s in subfolders)
        split_dirs = (
            [os.path.join(grup_path, s) for s in subfolders
             if os.path.isdir(os.path.join(grup_path, s))]
            if has_split else [grup_path]
        )
        for split_dir in split_dirs:
            if not os.path.isdir(split_dir):
                continue
            for subject_id in os.listdir(split_dir):
                subject_path = os.path.join(split_dir, subject_id)
                if not os.path.isdir(subject_path):
                    continue
                images = []
                for root, _, files in os.walk(subject_path):
                    for f in files:
                        if f.lower().endswith(IMG_EXT):
                            images.append(os.path.join(root, f))
                if not images:
                    continue
                if max_per_subject:
                    images = images[:max_per_subject]
                # Format key sama dengan split_dataset.py: {grup}__{subject_id}
                key = f"{grup}__{subject_id}"
                if key in subjects:
                    subjects[key].extend(images)
                else:
                    subjects[key] = images
    return subjects


def load_split_subject_keys(metadata_path):

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata tidak ditemukan: {metadata_path}\n"
            "Jalankan split_dataset.py (versi fixed) terlebih dahulu."
        )
    with open(metadata_path) as f:
        meta = json.load(f)

    if 'subjects' in meta and 'subject_keys' in meta['subjects']:
        return meta['subjects']['subject_keys']   # {split: [keys]}
    raise KeyError(
        "Format metadata tidak sesuai. Jalankan split_dataset.py versi fixed."
    )

def load_face_swapper():
    import insightface
    from insightface.app import FaceAnalysis
    print("\nMemuat model InsightFace ...")
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(320, 320))
    swapper = insightface.model_zoo.get_model(
        'inswapper_128.onnx', download=True, download_zip=True
    )
    print("Model berhasil dimuat!")
    return app, swapper


def detect_face(app, image_bgr):
    faces = app.get(image_bgr)
    if not faces:
        return None
    return sorted(faces, key=lambda x: x.bbox[2] * x.bbox[3], reverse=True)[0]


def generate_swap_insight(app, swapper, src_path, tgt_path, output_size):
    src_img = cv2.imread(src_path)
    tgt_img = cv2.imread(tgt_path)
    if src_img is None or tgt_img is None:
        return None
    src_face = detect_face(app, src_img)
    tgt_face = detect_face(app, tgt_img)
    if src_face is None or tgt_face is None:
        return None
    result = swapper.get(tgt_img, tgt_face, src_face, paste_back=True)
    return cv2.resize(result, output_size)


def generate_swap_simple(src_path, tgt_path, output_size):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    def detect(img):
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        return max(faces, key=lambda f: f[2]*f[3]) if len(faces) else None

    src_img = cv2.imread(src_path)
    tgt_img = cv2.imread(tgt_path)
    if src_img is None or tgt_img is None:
        return None

    sf = detect(src_img)
    tf = detect(tgt_img)
    if sf is None or tf is None:
        return None

    sx, sy, sw, sh = sf
    tx, ty, tw, th = tf

    src_crop    = cv2.resize(src_img[sy:sy+sh, sx:sx+sw], (tw, th))
    mask        = np.zeros((th, tw), dtype=np.uint8)
    cv2.ellipse(mask, (tw//2, th//2), (tw//2-5, th//2-5), 0, 0, 360, 255, -1)
    mask        = cv2.GaussianBlur(mask, (21, 21), 11)
    mask_3ch    = cv2.merge([mask]*3).astype(np.float32) / 255.0
    result      = tgt_img.copy()
    roi         = result[ty:ty+th, tx:tx+tw].astype(np.float32)
    result[ty:ty+th, tx:tx+tw] = (
        roi * (1 - mask_3ch) + src_crop.astype(np.float32) * mask_3ch
    ).astype(np.uint8)

    return cv2.resize(result, output_size)


def generate_for_split(split_name, subject_images, output_dir,
                       num_pairs, swaps_per_pair, output_size,
                       use_insightface, app=None, swapper=None):

    os.makedirs(output_dir, exist_ok=True)
    subject_ids = list(subject_images.keys())

    if len(subject_ids) < 2:
        print(f"  [SKIP] {split_name}: butuh ≥2 subjek, hanya ada {len(subject_ids)}")
        return 0

    pairs = [(random.choice(subject_ids),
              random.choice([s for s in subject_ids if s != src]))
             for src in random.choices(subject_ids, k=num_pairs)
             for _ in range(1)]
    pairs = []
    for _ in range(num_pairs):
        src_id = random.choice(subject_ids)
        tgt_id = random.choice([s for s in subject_ids if s != src_id])
        pairs.append((src_id, tgt_id))

    success  = 0
    counter  = len([f for f in os.listdir(output_dir) if f.lower().endswith(IMG_EXT)])
    pbar     = tqdm(pairs, desc=f"  Fake [{split_name}]")

    for src_id, tgt_id in pbar:
        for _ in range(swaps_per_pair):
            src_path = random.choice(subject_images[src_id])
            tgt_path = random.choice(subject_images[tgt_id])

            if use_insightface:
                result = generate_swap_insight(app, swapper, src_path, tgt_path, output_size)
            else:
                result = generate_swap_simple(src_path, tgt_path, output_size)

            if result is not None:
                out_path = os.path.join(output_dir, f"fake_{counter:06d}.jpg")
                cv2.imwrite(out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
                success  += 1
                counter  += 1

        pbar.set_postfix({'success': success})

    return success

def main():
    parser = argparse.ArgumentParser(description='Generate Deepfake dari IMSFD')
    parser.add_argument('--imsfd_dir',     default=IMSFD_DIR)
    parser.add_argument('--output_dir',    default=OUTPUT_DIR)
    parser.add_argument('--data_dir',      default=DATA_DIR)
    parser.add_argument('--metadata_path', default=METADATA_PATH)
    parser.add_argument('--num_pairs',     type=int, default=NUM_PAIRS)
    parser.add_argument('--swaps_per_pair',type=int, default=SWAPS_PER_PAIR)
    parser.add_argument('--mode',          default='auto',
                        choices=['insightface', 'simple', 'auto'])
    parser.add_argument('--max_per_subject', type=int, default=None)
    parser.add_argument(
        '--split_aware', action='store_true',
        help='[DIREKOMENDASIKAN] Generate fake per split menggunakan '
             'metadata dari split_dataset.py. Mencegah subject leakage pada fake.'
    )
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    use_insightface = False
    if args.mode == 'insightface':
        use_insightface = True
    elif args.mode == 'auto':
        use_insightface = check_dependencies()

    app = swapper = None
    if use_insightface:
        app, swapper = load_face_swapper()

    print("=" * 65)
    print("DEEPFAKE GENERATOR  —  FIXED VERSION".center(65))
    print("=" * 65)

    print("\n[1/3] Mengumpulkan subjek dari IMSFD ...")
    all_subjects = collect_subjects_from_imsfd(
        args.imsfd_dir, max_per_subject=args.max_per_subject
    )
    print(f"      Subjek: {len(all_subjects):,}  |  "
          f"Gambar: {sum(len(v) for v in all_subjects.values()):,}")

    if args.split_aware:
        print("\n[2/3] Mode: SPLIT-AWARE (mencegah subject leakage pada fake)")
        split_subject_keys = load_split_subject_keys(args.metadata_path)

        total_target = args.num_pairs * args.swaps_per_pair
        split_targets = {
            'train': int(total_target * SPLIT_RATIO_DEFAULT['train']),
            'val':   int(total_target * SPLIT_RATIO_DEFAULT['val']),
            'test':  int(total_target * SPLIT_RATIO_DEFAULT['test']),
        }

        print("\n[3/3] Generate fake per split ...")
        grand_total = 0
        for split_name, subject_keys in split_subject_keys.items():
            split_images = {
                k: all_subjects[k]
                for k in subject_keys if k in all_subjects
            }
            out_dir = os.path.join(args.data_dir, split_name, 'fake')

            n_pairs = split_targets.get(split_name, 500) // args.swaps_per_pair
            success = generate_for_split(
                split_name, split_images, out_dir,
                num_pairs=n_pairs,
                swaps_per_pair=args.swaps_per_pair,
                output_size=OUTPUT_SIZE,
                use_insightface=use_insightface,
                app=app, swapper=swapper,
            )
            grand_total += success
            print(f"  {split_name}: {success:,} fake dihasilkan → {out_dir}")

        print(f"\n  Total fake: {grand_total:,}")
        print("Setiap split hanya mengandung fake dari subjek split tersebut")

    else:
        print("\n[2/3] Mode: FLAT (semua fake ke satu folder)")
        print("Gunakan --split_aware untuk hasil yang lebih valid")

        print(f"\n[3/3] Generate {args.num_pairs} pasangan ...")
        success = generate_for_split(
            'all', all_subjects, args.output_dir,
            num_pairs=args.num_pairs,
            swaps_per_pair=args.swaps_per_pair,
            output_size=OUTPUT_SIZE,
            use_insightface=use_insightface,
            app=app, swapper=swapper,
        )
        print(f"\n  Total fake: {success:,} → {args.output_dir}")
        print("  Selanjutnya jalankan split_dataset.py")

    log = {
        'generated_at':    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'split_aware':     args.split_aware,
        'mode':            'insightface' if use_insightface else 'simple',
        'num_pairs':       args.num_pairs,
        'swaps_per_pair':  args.swaps_per_pair,
    }
    log_dir  = args.data_dir if args.split_aware else args.output_dir
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'generation_log.json')
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=4)
    print(f"\n  Log: {log_path}")
    print("=" * 65)


SPLIT_RATIO_DEFAULT = {"train": 0.70, "val": 0.15, "test": 0.15}

if __name__ == '__main__':
    main()