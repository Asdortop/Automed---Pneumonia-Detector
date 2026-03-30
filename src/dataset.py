"""
dataset.py — Phase 1
Shared PyTorch Dataset & DataLoader factory used by both
the classifier (Phase 2) and any future phases.

Labels: NORMAL = 0, PNEUMONIA = 1
"""

from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
PROC_DIR   = BASE_DIR / "data" / "processed"
CLASSES    = ["NORMAL", "PNEUMONIA"]          # index 0, 1
# ─────────────────────────────────────────────────────────────────────────────


# ── Transforms ────────────────────────────────────────────────────────────────
def get_transforms(split: str) -> transforms.Compose:
    """
    Training split: augmentation (flip, rotate, color jitter).
    Val / test splits: only normalize — no augmentation.

    ImageNet mean/std used because ResNet-50 was pretrained on ImageNet.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),                                    # [0,1] float
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    else:  # val / test — deterministic
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
# ─────────────────────────────────────────────────────────────────────────────


class XRayDataset(Dataset):
    """
    Loads preprocessed chest X-ray PNGs from data/processed/<split>/.

    Args:
        split   : "train", "val", or "test"
        transform: torchvision transform pipeline (auto-selected if None)
    """

    def __init__(self, split: str, transform: Optional[transforms.Compose] = None):
        self.split     = split
        self.transform = transform or get_transforms(split)
        self.samples: list[Tuple[Path, int]] = []

        split_dir = PROC_DIR / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Processed data not found at '{split_dir}'.\n"
                f"Run `python src/preprocess.py` first."
            )

        for label_idx, cls in enumerate(CLASSES):
            cls_dir = split_dir / cls
            for img_path in sorted(cls_dir.glob("*.png")):
                self.samples.append((img_path, label_idx))

        if not self.samples:
            raise RuntimeError(f"No images found in '{split_dir}'.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label

    def class_counts(self) -> dict:
        """Returns {class_name: count} — useful for computing class weights."""
        counts = {cls: 0 for cls in CLASSES}
        for _, label in self.samples:
            counts[CLASSES[label]] += 1
        return counts


# ── DataLoader Factory ────────────────────────────────────────────────────────
def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 2,
    balance: bool = True,
) -> dict[str, DataLoader]:
    """
    Returns a dict of DataLoaders for train / val / test splits.

    Args:
        batch_size  : images per batch
        num_workers : parallel data loading workers
        balance     : if True, uses WeightedRandomSampler on train split
                      to compensate for PNEUMONIA:NORMAL imbalance (~3:1)
    """
    loaders = {}

    for split in ["train", "val", "test"]:
        dataset = XRayDataset(split)

        if split == "train" and balance:
            # Compute per-sample weights so each class is equally likely
            counts  = dataset.class_counts()
            total   = sum(counts.values())
            weights = {cls: total / cnt for cls, cnt in counts.items()}
            sample_weights = [
                weights[CLASSES[label]] for _, label in dataset.samples
            ]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                pin_memory=True,
            )

        loaders[split] = loader
        print(f"  [{split:5s}] {len(dataset):5d} images | "
              f"classes: {dataset.class_counts()}")

    return loaders


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading dataloaders...")
    loaders = get_dataloaders(batch_size=32)

    for split, loader in loaders.items():
        imgs, labels = next(iter(loader))
        print(f"  {split}: batch shape {imgs.shape}, "
              f"labels {labels.unique().tolist()}, "
              f"pixel mean {imgs.mean():.4f}")

    print("\n✅  Dataset verified successfully.")
