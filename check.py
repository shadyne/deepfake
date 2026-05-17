"""
diagnose_dataset.py
====================
Jalankan script ini SEBELUM generate_deepfake.py
untuk mendiagnosa masalah dataset dan face detection.

Cara pakai:
    python diagnose_dataset.py
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path

# ── SESUAIKAN PATH INI ─────────────────────────────────────
IMSFD_DIR = r"D:\DEEP FAKE\data\IMSFD"
SAMPLE_SIZE = 20   # Jumlah foto yang dicek
# ───────────────────────────────────────────────────────────

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')


def scan_folder_structure(imsfd_dir):
    """Scan dan print struktur folder IMSFD secara detail"""
    print("\n" + "="*65)
    print("1. STRUKTUR FOLDER".center(65))
    print("="*65)

    if not os.path.exists(imsfd_dir):
        print(f"[ERROR] Folder tidak ditemukan: {imsfd_dir}")
        return []

    all_image_paths = []
    level1 = sorted(os.listdir(imsfd_dir))  # A1, A2, B1, dst

    print(f"\nRoot: {imsfd_dir}")
    print(f"Folder level-1 (grup): {level1}\n")

    for grup in level1:
        grup_path = os.path.join(imsfd_dir, grup)
        if not os.path.isdir(grup_path):
            continue

        level2 = sorted(os.listdir(grup_path))
        has_split = any(s.lower() in ['training', 'testing'] for s in level2)

        print(f"  [{grup}]  →  {level2[:5]}{'...' if len(level2)>5 else ''}")

        if has_split:
            # Ada subfolder training/testing
            for split in level2:
                split_path = os.path.join(grup_path, split)
                if not os.path.isdir(split_path):
                    continue
                subjects = sorted(os.listdir(split_path))
                print(f"    [{split}]  →  {len(subjects)} subjek")

                # Ambil 2 subjek contoh
                for subj in subjects[:2]:
                    subj_path = os.path.join(split_path, subj)
                    if not os.path.isdir(subj_path):
                        continue
                    imgs = [f for f in os.listdir(subj_path) if f.lower().endswith(IMG_EXT)]
                    print(f"      [{subj}]  →  {len(imgs)} foto")
                    for img in imgs:
                        all_image_paths.append(os.path.join(subj_path, img))
        else:
            # Langsung subfolder ID
            for subj in level2[:2]:
                subj_path = os.path.join(grup_path, subj)
                if not os.path.isdir(subj_path):
                    continue
                imgs = [f for f in os.listdir(subj_path) if f.lower().endswith(IMG_EXT)]
                print(f"    [{subj}]  →  {len(imgs)} foto")
                for img in imgs:
                    all_image_paths.append(os.path.join(subj_path, img))

            # Collect semua gambar dari level ini
            for root, _, files in os.walk(grup_path):
                for f in files:
                    if f.lower().endswith(IMG_EXT):
                        full = os.path.join(root, f)
                        if full not in all_image_paths:
                            all_image_paths.append(full)

    return all_image_paths


def check_image_properties(image_paths, sample_size=20):
    """Cek properti gambar: ukuran, format, bisa dibaca atau tidak"""
    print("\n" + "="*65)
    print("2. PROPERTI GAMBAR (SAMPLE)".center(65))
    print("="*65)

    if not image_paths:
        print("[ERROR] Tidak ada gambar ditemukan!")
        return

    sample = random.sample(image_paths, min(sample_size, len(image_paths)))

    sizes = []
    unreadable = []
    too_small = []

    for path in sample:
        img = cv2.imread(path)
        if img is None:
            unreadable.append(path)
            continue

        h, w = img.shape[:2]
        sizes.append((w, h))

        if w < 60 or h < 60:
            too_small.append((path, w, h))

    print(f"\nSample dicek    : {len(sample)} foto")
    print(f"Tidak terbaca   : {len(unreadable)}")
    print(f"Terlalu kecil   : {len(too_small)} (< 60px)")

    if sizes:
        ws = [s[0] for s in sizes]
        hs = [s[1] for s in sizes]
        print(f"\nUkuran foto:")
        print(f"  Width  — min: {min(ws)}px, max: {max(ws)}px, rata2: {sum(ws)//len(ws)}px")
        print(f"  Height — min: {min(hs)}px, max: {max(hs)}px, rata2: {sum(hs)//len(hs)}px")

    if too_small:
        print(f"\nFoto terlalu kecil (deteksi akan gagal):")
        for p, w, h in too_small[:5]:
            print(f"  {os.path.basename(p)}: {w}x{h}px")

    if unreadable:
        print(f"\nFoto tidak bisa dibaca:")
        for p in unreadable[:5]:
            print(f"  {p}")


def test_face_detection(image_paths, sample_size=20):
    """Test apakah InsightFace bisa mendeteksi wajah"""
    print("\n" + "="*65)
    print("3. TEST FACE DETECTION (InsightFace)".center(65))
    print("="*65)

    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print("[ERROR] InsightFace tidak terinstall!")
        print("        pip install insightface onnxruntime")
        _test_opencv_detection(image_paths, sample_size)
        return

    print("\nMemuat model InsightFace...")

    # Test dengan berbagai det_size
    det_sizes = [(320, 320), (640, 640), (160, 160)]
    best_det_size = None
    best_success = 0

    sample = random.sample(image_paths, min(sample_size, len(image_paths)))

    for det_size in det_sizes:
        app = FaceAnalysis(name='buffalo_l')
        app.prepare(ctx_id=-1, det_size=det_size)  # -1 = CPU

        success = 0
        fail = 0
        fail_examples = []

        for path in sample:
            img = cv2.imread(path)
            if img is None:
                fail += 1
                continue

            faces = app.get(img)
            if faces:
                success += 1
            else:
                fail += 1
                if len(fail_examples) < 3:
                    h, w = img.shape[:2]
                    fail_examples.append((os.path.basename(path), w, h))

        rate = success / len(sample) * 100
        print(f"\n  det_size={det_size}: {success}/{len(sample)} berhasil ({rate:.0f}%)")

        if fail_examples:
            for name, w, h in fail_examples:
                print(f"    Gagal: {name} ({w}x{h}px)")

        if success > best_success:
            best_success = success
            best_det_size = det_size

    print(f"\n→ det_size terbaik: {best_det_size} ({best_success}/{len(sample)} berhasil)")

    if best_success == 0:
        print("\n[!] MASALAH: Wajah tidak terdeteksi sama sekali!")
        print("    Kemungkinan penyebab:")
        print("    1. Foto bukan foto wajah (objek lain)")
        print("    2. Foto wajah terlalu kecil/jauh")
        print("    3. Pencahayaan sangat buruk")
        print("    4. Foto sudah di-crop terlalu rapat (hanya mata/hidung)")
        _suggest_fix(image_paths)
    elif best_success < len(sample) * 0.5:
        print(f"\n[!] PERINGATAN: Deteksi rendah ({best_success/len(sample)*100:.0f}%)")
        print("    Sebagian foto mungkin tidak mengandung wajah yang jelas")

    return best_det_size


def _test_opencv_detection(image_paths, sample_size=10):
    """Fallback: test dengan OpenCV Haar Cascade"""
    print("\n  Mencoba deteksi dengan OpenCV (fallback)...")

    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    sample = random.sample(image_paths, min(sample_size, len(image_paths)))
    success = 0

    for path in sample:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            success += 1

    print(f"  OpenCV deteksi: {success}/{len(sample)} berhasil ({success/len(sample)*100:.0f}%)")


def _suggest_fix(image_paths):
    """Saran perbaikan berdasarkan sample foto"""
    print("\n" + "="*65)
    print("4. ANALISIS & SARAN".center(65))
    print("="*65)

    sample = image_paths[:5]
    print("\nContoh 5 path foto pertama:")
    for p in sample:
        img = cv2.imread(p)
        shape = img.shape[:2] if img is not None else "tidak terbaca"
        print(f"  {p}")
        print(f"    → ukuran: {shape}")

    print("\nSaran berdasarkan hasil diagnosa:")
    print("  A) Jika foto berukuran KECIL (< 100px):")
    print("     → Ubah det_size ke (160, 160) di generate_deepfake.py")
    print("     → Atau resize foto ke minimal 200x200 dulu")
    print()
    print("  B) Jika foto bukan wajah / wajah tidak jelas:")
    print("     → Dataset ini mungkin perlu preprocessing (face crop dulu)")
    print("     → Gunakan insightface untuk crop wajah dari foto aslinya")
    print()
    print("  C) Jika foto wajah sudah di-crop terlalu rapat:")
    print("     → Tambahkan padding di sekitar foto sebelum deteksi")


def count_total_images(imsfd_dir):
    """Hitung total gambar per grup"""
    print("\n" + "="*65)
    print("5. TOTAL GAMBAR PER GRUP".center(65))
    print("="*65)

    grand_total = 0
    for grup in sorted(os.listdir(imsfd_dir)):
        grup_path = os.path.join(imsfd_dir, grup)
        if not os.path.isdir(grup_path):
            continue
        count = 0
        for root, _, files in os.walk(grup_path):
            count += len([f for f in files if f.lower().endswith(IMG_EXT)])
        grand_total += count
        print(f"  {grup:<10} : {count:>6,} gambar")

    print(f"  {'TOTAL':<10} : {grand_total:>6,} gambar")
    print(f"\n  → Untuk dataset seimbang, generate minimal {grand_total:,} foto fake")
    print(f"  → Dengan NUM_PAIRS=5000, SWAPS_PER_PAIR={grand_total//5000 + 1}")


def main():
    random.seed(42)

    print("="*65)
    print("DIAGNOSA DATASET IMSFD".center(65))
    print("="*65)
    print(f"Path: {IMSFD_DIR}")

    # 1. Scan struktur
    all_images = scan_folder_structure(IMSFD_DIR)
    print(f"\nTotal gambar terkumpul dari scan: {len(all_images):,}")

    if not all_images:
        print("\n[ERROR] Tidak ada gambar yang bisa diakses!")
        print("Cek apakah path IMSFD_DIR sudah benar.")
        return

    # 2. Cek properti gambar
    check_image_properties(all_images, SAMPLE_SIZE)

    # 3. Test face detection
    best_size = test_face_detection(all_images, SAMPLE_SIZE)

    # 4. Hitung total
    count_total_images(IMSFD_DIR)

    # 5. Ringkasan
    print("\n" + "="*65)
    print("RINGKASAN & LANGKAH SELANJUTNYA".center(65))
    print("="*65)
    if best_size:
        print(f"\n✓ Gunakan det_size={best_size} di generate_deepfake.py")
        print(f"  Ubah baris ini:")
        print(f"  app.prepare(ctx_id=0, det_size={best_size})")
    print("\nSetelah fix, jalankan kembali: python generate_deepfake.py")
    print("="*65)


if __name__ == '__main__':
    main()