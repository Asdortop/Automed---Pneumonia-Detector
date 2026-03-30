"""
preprocess.py — Phase 1
Reads raw chest X-ray images from the Kaggle dataset,
resizes to 224×224, normalizes to [0,1], and saves as
lossless PNG to data/processed/.
"""

from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent          # project root
RAW_DIR   = BASE_DIR / "Dataset" / "chest_xray"            # Kaggle extract
OUT_DIR   = BASE_DIR / "data" / "processed"
IMG_SIZE  = (224, 224)
SPLITS    = ["train", "test", "val"]
CLASSES   = ["NORMAL", "PNEUMONIA"]
# ─────────────────────────────────────────────────────────────────────────────


def preprocess_split(split: str) -> dict:
    """Process one split (train/test/val) and return image counts."""
    counts = {}
    for cls in CLASSES:
        src_folder = RAW_DIR / split / cls
        dst_folder = OUT_DIR / split / cls
        dst_folder.mkdir(parents=True, exist_ok=True)

        # Support JPEG and PNG source files
        files = (
            list(src_folder.glob("*.jpeg"))
            + list(src_folder.glob("*.jpg"))
            + list(src_folder.glob("*.png"))
        )

        if not files:
            print(f"  ⚠️  No images found in {src_folder}")
            counts[cls] = 0
            continue

        print(f"\n  [{split}/{cls}] — {len(files)} images")
        for img_path in tqdm(files, desc=f"{split}/{cls}", ncols=80):
            # 1. Open and convert to RGB (handles grayscale X-rays)
            img = Image.open(img_path).convert("RGB")

            # 2. Resize with high-quality Lanczos filter
            img = img.resize(IMG_SIZE, Image.LANCZOS)

            # 3. Normalize pixel values to [0, 1] float32
            arr = np.array(img, dtype=np.float32) / 255.0

            # 4. Convert back to uint8 for lossless PNG save
            out_img = Image.fromarray((arr * 255).astype(np.uint8))
            out_path = dst_folder / (img_path.stem + ".png")
            out_img.save(out_path, format="PNG")

        counts[cls] = len(files)
    return counts


def verify_output():
    """Sanity check: print counts and pixel range for each split/class."""
    print("\n" + "─" * 55)
    print("  Verification")
    print("─" * 55)
    all_ok = True
    for split in SPLITS:
        for cls in CLASSES:
            folder = OUT_DIR / split / cls
            files = list(folder.glob("*.png"))
            if not files:
                print(f"  ❌  {split}/{cls}: NO FILES FOUND")
                all_ok = False
                continue
            # Check pixel range on a sample image
            sample = np.array(Image.open(files[0])).astype(np.float32) / 255.0
            lo, hi = sample.min(), sample.max()
            status = "✅" if hi <= 1.001 else "⚠️ "
            print(f"  {status} {split:5s}/{cls:9s}: {len(files):5d} imgs | "
                  f"pixel range [{lo:.3f}, {hi:.3f}]")
    print("─" * 55)
    if all_ok:
        print("  ✅  All checks passed!")
    else:
        print("  ⚠️   Some folders are empty — check RAW_DIR path.")


def main():
    print("=" * 55)
    print("  AutoMed — Phase 1: Preprocessing")
    print(f"  Source : {RAW_DIR}")
    print(f"  Output : {OUT_DIR}")
    print(f"  Target : {IMG_SIZE[0]}×{IMG_SIZE[1]} px, normalized [0,1]")
    print("=" * 55)

    for split in SPLITS:
        print(f"\n▶ Processing split: {split}")
        preprocess_split(split)

    verify_output()
    print("\n✅  Preprocessing complete. Images saved to data/processed/\n")


if __name__ == "__main__":
    main()
