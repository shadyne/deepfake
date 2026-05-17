

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


IMSFD_DIR = r"D:\DEEP FAKE\data\IMSFD"

OUTPUT_DIR = r"D:\DEEP FAKE\data\IMSFD_DeepFake"

NUM_PAIRS = 5000

SWAPS_PER_PAIR = 3

OUTPUT_SIZE = (224, 224)

RANDOM_SEED = 42

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')



def check_dependencies():
    """Cek apakah semua library yang dibutuhkan sudah terinstall"""
    missing = []

    try:
        import insightface
    except ImportError:
        missing.append('insightface')

    try:
        import onnxruntime
    except ImportError:
        missing.append('onnxruntime')

    if missing:
        print("\n" + "=" * 60)
        print("DEPENDENCIES BELUM TERINSTALL".center(60))
        print("=" * 60)
        print("\nJalankan perintah berikut untuk install:")
        print(f"\n  pip install {' '.join(missing)}")
        print("\nUntuk GPU (lebih cepat):")
        print("  pip install onnxruntime-gpu")
        print("  (hapus/uninstall onnxruntime biasa dulu)")
        print("\n" + "=" * 60)
        return False

    return True


def collect_subjects(imsfd_dir, max_per_subject=None):
    subjects = {}

    grup_folders = [
        d for d in os.listdir(imsfd_dir)
        if os.path.isdir(os.path.join(imsfd_dir, d))
    ]

    for grup in sorted(grup_folders):
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
                images = []

                for root, _, files in os.walk(subject_path):
                    for f in files:
                        if f.lower().endswith(IMG_EXT):
                            images.append(os.path.join(root, f))

                if max_per_subject:
                    images = images[:max_per_subject]

                key = f"{grup}_{subject_id}"
                if key not in subjects:
                    subjects[key] = []
                subjects[key].extend(images)

    return subjects


def load_face_swapper():
    """Load InsightFace face swapper model"""
    import insightface
    from insightface.app import FaceAnalysis

    print("\nMemuat model InsightFace...")
    print("(Model akan auto-download jika belum ada, ~500MB)")

    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(320, 320))  

    swapper = insightface.model_zoo.get_model(
        'inswapper_128.onnx',
        download=True,
        download_zip=True
    )

    print("Model berhasil dimuat!")
    return app, swapper


def detect_face(app, image_bgr):
    """Detect wajah dalam gambar, return face object atau None"""
    faces = app.get(image_bgr)
    if not faces:
        return None
    faces = sorted(faces, key=lambda x: x.bbox[2] * x.bbox[3], reverse=True)
    return faces[0]


def generate_swap(app, swapper, src_path, tgt_path, output_size=(224, 224)):

    src_img = cv2.imread(src_path)
    tgt_img = cv2.imread(tgt_path)

    if src_img is None or tgt_img is None:
        return None

    src_face = detect_face(app, src_img)
    tgt_face = detect_face(app, tgt_img)

    if src_face is None or tgt_face is None:
        return None

    result = swapper.get(tgt_img, tgt_face, src_face, paste_back=True)

    result = cv2.resize(result, output_size)

    return result


def generate_deepfakes_insightface(subjects, output_dir, num_pairs, swaps_per_pair, output_size):
    """Generate deepfake menggunakan InsightFace"""

    if not check_dependencies():
        return 0

    app, swapper = load_face_swapper()

    os.makedirs(output_dir, exist_ok=True)

    subject_ids = list(subjects.keys())

    if len(subject_ids) < 2:
        print("[ERROR] Butuh minimal 2 subjek untuk face-swap!")
        return 0

    pairs = []
    for _ in range(num_pairs):
        src_id, tgt_id = random.sample(subject_ids, 2)
        pairs.append((src_id, tgt_id))

    print(f"\nMembuat {num_pairs} pasangan face-swap...")
    print(f"Setiap pasangan: {swaps_per_pair} swap")
    print(f"Total target: {num_pairs * swaps_per_pair} gambar fake\n")

    success_count = 0
    fail_count = 0
    counter = 0

    pbar = tqdm(pairs, desc="Generating deepfake")

    for src_id, tgt_id in pbar:
        src_images = subjects.get(src_id, [])
        tgt_images = subjects.get(tgt_id, [])

        if not src_images or not tgt_images:
            continue

        for _ in range(swaps_per_pair):
            src_path = random.choice(src_images)
            tgt_path = random.choice(tgt_images)

            result = generate_swap(app, swapper, src_path, tgt_path, output_size)

            if result is not None:
                out_name = f"fake_{counter:06d}.jpg"
                out_path = os.path.join(output_dir, out_name)
                cv2.imwrite(out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
                success_count += 1
                counter += 1
            else:
                fail_count += 1

        pbar.set_postfix({
            'success': success_count,
            'fail': fail_count
        })

    return success_count


def generate_deepfakes_simple(subjects, output_dir, num_pairs, output_size):

    import cv2

    os.makedirs(output_dir, exist_ok=True)

    subject_ids = list(subjects.keys())
    if len(subject_ids) < 2:
        print("[ERROR] Butuh minimal 2 subjek!")
        return 0

    print("\n[INFO] Menggunakan mode simple blending (fallback)")
    print("       Install insightface untuk hasil yang lebih realistis\n")

    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    def detect_face_opencv(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        if len(faces) == 0:
            return None
        return max(faces, key=lambda f: f[2] * f[3]) 

    def simple_face_blend(src_img, tgt_img):
        """Blend wajah dari src ke tgt menggunakan seamless cloning"""
        src_face = detect_face_opencv(src_img)
        tgt_face = detect_face_opencv(tgt_img)

        if src_face is None or tgt_face is None:
            return None

        sx, sy, sw, sh = src_face
        tx, ty, tw, th = tgt_face

        src_crop = src_img[sy:sy+sh, sx:sx+sw]
        src_resized = cv2.resize(src_crop, (tw, th))

        mask = np.zeros((th, tw), dtype=np.uint8)
        center = (tw // 2, th // 2)
        axes = (tw // 2 - 5, th // 2 - 5)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 11)
        mask_3ch = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0

        result = tgt_img.copy()
        roi = result[ty:ty+th, tx:tx+tw].astype(np.float32)
        blended = roi * (1 - mask_3ch) + src_resized.astype(np.float32) * mask_3ch
        result[ty:ty+th, tx:tx+tw] = blended.astype(np.uint8)

        return result

    success_count = 0
    counter = 0

    pairs = []
    for _ in range(num_pairs):
        src_id, tgt_id = random.sample(subject_ids, 2)
        pairs.append((src_id, tgt_id))

    pbar = tqdm(pairs, desc="Generating deepfake (simple)")

    for src_id, tgt_id in pbar:
        src_images = subjects.get(src_id, [])
        tgt_images = subjects.get(tgt_id, [])

        if not src_images or not tgt_images:
            continue

        src_path = random.choice(src_images)
        tgt_path = random.choice(tgt_images)

        src_img = cv2.imread(src_path)
        tgt_img = cv2.imread(tgt_path)

        if src_img is None or tgt_img is None:
            continue

        result = simple_face_blend(src_img, tgt_img)

        if result is not None:
            result = cv2.resize(result, output_size)
            out_name = f"fake_{counter:06d}.jpg"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
            success_count += 1
            counter += 1

        pbar.set_postfix({'success': success_count})

    return success_count


def save_generation_log(output_dir, stats):
    """Simpan log hasil generasi"""
    log = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        **stats
    }
    log_path = os.path.join(output_dir, 'generation_log.json')
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=4)
    print(f"Log tersimpan: {log_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate Deepfake dari IMSFD')
    parser.add_argument('--imsfd_dir', type=str, default=IMSFD_DIR,
                        help='Path ke folder IMSFD')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='Path output deepfake')
    parser.add_argument('--num_pairs', type=int, default=NUM_PAIRS,
                        help='Jumlah pasangan face-swap')
    parser.add_argument('--swaps_per_pair', type=int, default=SWAPS_PER_PAIR,
                        help='Jumlah swap per pasangan')
    parser.add_argument('--mode', type=str, default='auto',
                        choices=['insightface', 'simple', 'auto'],
                        help='Mode: insightface (best), simple (fallback), auto (detect)')
    parser.add_argument('--max_per_subject', type=int, default=None,
                        help='Maksimum foto per subjek')
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 65)
    print("DEEPFAKE GENERATOR — IMSFD".center(65))
    print("=" * 65)
    print(f"\nSource   : {args.imsfd_dir}")
    print(f"Output   : {args.output_dir}")
    print(f"Pairs    : {args.num_pairs:,}")
    print(f"Per pair : {args.swaps_per_pair}")
    print(f"Target   : {args.num_pairs * args.swaps_per_pair:,} gambar fake")

    print("\n[1/3] Mengumpulkan data subjek dari IMSFD...")
    subjects = collect_subjects(args.imsfd_dir, args.max_per_subject)
    total_imgs = sum(len(v) for v in subjects.values())
    print(f"      Subjek ditemukan : {len(subjects):,}")
    print(f"      Total gambar     : {total_imgs:,}")

    if len(subjects) < 2:
        print("\n[ERROR] Dataset terlalu kecil, butuh minimal 2 subjek!")
        return

    print("\n[2/3] Generate deepfake...")

    use_insightface = False
    if args.mode == 'insightface':
        use_insightface = True
    elif args.mode == 'auto':
        use_insightface = check_dependencies()
        if not use_insightface:
            print("\n[INFO] InsightFace tidak tersedia, pakai mode simple")

    if use_insightface:
        success = generate_deepfakes_insightface(
            subjects, args.output_dir,
            args.num_pairs, args.swaps_per_pair, OUTPUT_SIZE
        )
    else:
        success = generate_deepfakes_simple(
            subjects, args.output_dir,
            args.num_pairs, OUTPUT_SIZE
        )

    print("\n[3/3] Selesai!")
    print("\n" + "=" * 65)
    print("HASIL GENERASI DEEPFAKE".center(65))
    print("=" * 65)
    print(f"Gambar fake berhasil dibuat : {success:,}")
    print(f"Tersimpan di                : {os.path.abspath(args.output_dir)}")

    if success > 0:
        print("\nLangkah selanjutnya:")
        print("  1. Jalankan split_dataset.py untuk split train/val/test")
        print("  2. Jalankan python main.py untuk training")
    else:
        print("\n[WARNING] Tidak ada gambar yang berhasil dibuat!")
        print("  Cek apakah foto di IMSFD dapat dibaca dengan benar.")

    print("=" * 65)

    save_generation_log(args.output_dir, {
        'total_subjects': len(subjects),
        'total_real_images': total_imgs,
        'num_pairs': args.num_pairs,
        'success_count': success,
        'mode': 'insightface' if use_insightface else 'simple'
    })


if __name__ == '__main__':
    main()