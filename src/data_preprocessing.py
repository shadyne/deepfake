
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import json


IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


class DeepfakeDataset(Dataset):

    def __init__(self, root_dir, transform=None, verbose=True, max_per_class=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        real_dir = os.path.join(root_dir, 'real')
        real_count = self._load_images_from_dir(real_dir, label=0, max_count=max_per_class)

        fake_dir = os.path.join(root_dir, 'fake')
        fake_count = self._load_images_from_dir(fake_dir, label=1, max_count=max_per_class)

        self.num_real = real_count
        self.num_fake = fake_count

        if len(self.images) == 0:
            raise ValueError(
                f"Tidak ada gambar ditemukan di: {root_dir}\n"
                f"Pastikan folder 'real/' dan 'fake/' sudah ada dan berisi gambar.\n"
                f"Jalankan split_dataset.py terlebih dahulu."
            )

        if verbose:
            self._print_stats()

    def _load_images_from_dir(self, dir_path, label, max_count=None):
        """Load gambar dari satu direktori (flat, tidak rekursif)"""
        if not os.path.exists(dir_path):
            return 0

        files = sorted([
            f for f in os.listdir(dir_path)
            if f.lower().endswith(IMG_EXT)
        ])

        if max_count and len(files) > max_count:
            import random
            files = random.sample(files, max_count)

        for img_name in files:
            self.images.append(os.path.join(dir_path, img_name))
            self.labels.append(label)

        return len(files)

    def _print_stats(self):
        total = len(self.images)
        split_name = os.path.basename(self.root_dir).upper()

        print(f"\n{'─'*50}")
        print(f"  {split_name} Dataset")
        print(f"{'─'*50}")
        print(f"  Total gambar : {total:,}")
        print(f"  Real         : {self.num_real:,} ({self.num_real/total*100:.1f}%)")
        print(f"  Fake         : {self.num_fake:,} ({self.num_fake/total*100:.1f}%)")

        if self.num_real == 0:
            print("  [!] PERINGATAN: Tidak ada gambar REAL!")
        if self.num_fake == 0:
            print("  [!] PERINGATAN: Tidak ada gambar FAKE!")
            print("      → Jalankan generate_deepfake.py dulu")
        if total > 0 and abs(self.num_real - self.num_fake) / total > 0.1:
            print(f"  [!] Dataset tidak seimbang (imbalanced)")
        print(f"{'─'*50}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"  [ERROR] Gagal load gambar: {img_path} — {e}")
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self):
        """Return distribusi kelas sebagai dict"""
        total = len(self.images)
        return {
            'real': self.num_real,
            'fake': self.num_fake,
            'total': total,
            'real_ratio': self.num_real / total if total > 0 else 0,
            'fake_ratio': self.num_fake / total if total > 0 else 0,
            'is_balanced': abs(self.num_real - self.num_fake) / max(total, 1) < 0.1
        }

    def get_weighted_sampler(self):

        class_counts = [self.num_real, self.num_fake]
        weights_per_class = [1.0 / c if c > 0 else 0 for c in class_counts]
        sample_weights = [weights_per_class[label] for label in self.labels]

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )


def get_transforms(config, mode='train'):

    img_size = getattr(config, 'IMAGE_SIZE', 224)
    mean = getattr(config, 'NORMALIZE_MEAN', [0.5, 0.5, 0.5])
    std  = getattr(config, 'NORMALIZE_STD',  [0.5, 0.5, 0.5])
    use_aug = getattr(config, 'USE_AUGMENTATION', True)
    flip_p  = getattr(config, 'HORIZONTAL_FLIP_PROB', 0.5)
    rot_deg = getattr(config, 'ROTATION_DEGREES', 10)

    if mode == 'train' and use_aug:
        transform = transforms.Compose([
            transforms.Resize((img_size + 20, img_size + 20)),  
            transforms.RandomCrop((img_size, img_size)),         
            transforms.RandomHorizontalFlip(p=flip_p),
            transforms.RandomRotation(degrees=rot_deg),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.05
            ),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)) 
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    return transform


def verify_dataset_structure(config):

    print("\n" + "=" * 65)
    print("VERIFIKASI STRUKTUR DATASET".center(65))
    print("=" * 65)

    required = {
        'Train': config.TRAIN_DIR,
        'Val':   config.VAL_DIR,
        'Test':  config.TEST_DIR,
    }

    all_ok = True

    for split_name, split_dir in required.items():
        real_dir = os.path.join(split_dir, 'real')
        fake_dir = os.path.join(split_dir, 'fake')

        real_count = len([
            f for f in os.listdir(real_dir)
            if f.lower().endswith(IMG_EXT)
        ]) if os.path.exists(real_dir) else 0

        fake_count = len([
            f for f in os.listdir(fake_dir)
            if f.lower().endswith(IMG_EXT)
        ]) if os.path.exists(fake_dir) else 0

        status_real = "✓" if real_count > 0 else "✗"
        status_fake = "✓" if fake_count > 0 else "✗"

        print(f"\n  {split_name} ({split_dir})")
        print(f"    [{status_real}] real/ : {real_count:,} gambar")
        print(f"    [{status_fake}] fake/ : {fake_count:,} gambar")

        if real_count == 0 or fake_count == 0:
            all_ok = False

    if not all_ok:
        print("\n  [!] Dataset belum lengkap!")
        print("  Langkah yang perlu dilakukan:")
        print("  1. Pastikan IMSFD_DIR di split_dataset.py sudah benar")
        print("  2. Jalankan: python generate_deepfake.py")
        print("  3. Jalankan: python split_dataset.py")

    print("=" * 65 + "\n")
    return all_ok


def get_dataloaders(config, use_weighted_sampler=False):

    print("\n" + "=" * 65)
    print("LOADING DATASETS".center(65))
    print("=" * 65)

    _check_dir_exists(config.TRAIN_DIR, "Train")
    _check_dir_exists(config.VAL_DIR, "Val")
    _check_dir_exists(config.TEST_DIR, "Test")

    train_transform = get_transforms(config, mode='train')
    val_transform   = get_transforms(config, mode='val')
    test_transform  = get_transforms(config, mode='test')

    train_dataset = DeepfakeDataset(
        root_dir=config.TRAIN_DIR,
        transform=train_transform,
        verbose=True
    )

    val_dataset = DeepfakeDataset(
        root_dir=config.VAL_DIR,
        transform=val_transform,
        verbose=True
    )

    test_dataset = DeepfakeDataset(
        root_dir=config.TEST_DIR,
        transform=test_transform,
        verbose=True
    )

    pin_mem = (config.DEVICE.type == 'cuda')
    num_workers = getattr(config, 'NUM_WORKERS', 0)

    if use_weighted_sampler:
        sampler = train_dataset.get_weighted_sampler()
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_mem
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=True
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem
    )

    print(f"\n{'─'*65}")
    print(f"  DataLoader Summary")
    print(f"{'─'*65}")
    print(f"  Train : {len(train_loader):,} batch  ({len(train_dataset):,} gambar)")
    print(f"  Val   : {len(val_loader):,} batch  ({len(val_dataset):,} gambar)")
    print(f"  Test  : {len(test_loader):,} batch  ({len(test_dataset):,} gambar)")
    print(f"  Batch size   : {config.BATCH_SIZE}")
    print(f"  Num workers  : {num_workers}")
    print(f"  Device       : {config.DEVICE}")
    print(f"{'─'*65}\n")

    return train_loader, val_loader, test_loader


def _check_dir_exists(dir_path, name):
    """Helper: cek keberadaan folder dan isinya"""
    if not os.path.exists(dir_path):
        print(f"  [!] Folder {name} tidak ditemukan: {dir_path}")
        return False
    return True


def denormalize(tensor, mean, std):
    """Denormalize tensor untuk visualisasi"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)


def visualize_batch_samples(dataloader, config, num_samples=8, save_path=None):
    """
    Visualisasi sampel dari dataloader.
    Berguna untuk verifikasi data loading sudah benar.
    """
    import matplotlib.pyplot as plt

    images, labels = next(iter(dataloader))
    images_vis = denormalize(images.clone(), config.NORMALIZE_MEAN, config.NORMALIZE_STD)

    num_samples = min(num_samples, len(images))
    cols = 4
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).ravel()

    for i in range(num_samples):
        img = images_vis[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        label_text = 'REAL' if labels[i] == 0 else 'FAKE'
        color = 'green' if labels[i] == 0 else 'red'
        axes[i].set_title(label_text, fontsize=11, fontweight='bold', color=color)
        axes[i].axis('off')

    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Sample Dataset (Hijau=Real, Merah=Fake)', fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Sample visualisasi tersimpan: {save_path}")

    plt.close()
    return fig