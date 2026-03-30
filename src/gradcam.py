"""
gradcam.py — Phase 3
Generates Grad-CAM heatmap overlays on chest X-rays using
the trained ResNet-50 classifier.

Usage (CLI):
    python src/gradcam.py --image path/to/xray.png
    python src/gradcam.py --split test --num_samples 5
"""

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).parent))
from classifier import XRayClassifier, build_model

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
MODELS_DIR  = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
# ─────────────────────────────────────────────────────────────────────────────


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for ResNet-50.
    Hooks into the final convolutional layer (model.backbone[-2] = layer4)
    and extracts activation maps + gradients.
    """

    def __init__(self, model: XRayClassifier, device: torch.device):
        self.model  = model.eval()
        self.device = device

        # layer4 is the 7th child of the backbone Sequential
        # backbone = [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool]
        self.target_layer = list(model.backbone.children())[-2]   # layer4

        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        # Register forward & backward hooks
        self._fwd_hook = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def generate(self, image_tensor: torch.Tensor) -> tuple[np.ndarray, float, int]:
        """
        Args:
            image_tensor: (1, 3, 224, 224) normalized tensor on self.device

        Returns:
            cam      : (224, 224) float32 heatmap in [0, 1]
            prob     : probability of PNEUMONIA (float)
            pred     : 0 = NORMAL, 1 = PNEUMONIA
        """
        self.model.zero_grad()

        image_tensor = image_tensor.to(self.device).requires_grad_(False)
        image_tensor.requires_grad = True

        logit = self.model(image_tensor)           # (1,) raw logit
        prob  = torch.sigmoid(logit).item()
        pred  = int(prob >= 0.5)

        # Backprop w.r.t. the logit score
        logit.backward()

        # Global average pool gradients over spatial dims → channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * self._activations).sum(dim=1)          # (1, H, W)
        cam     = torch.relu(cam).squeeze().cpu().numpy()           # (H, W)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upsample to 224×224
        cam = cv2.resize(cam, (224, 224))
        return cam, prob, pred


def get_location_from_cam(cam: np.ndarray) -> str:
    """
    Divides the 224×224 CAM into 4 quadrants and finds which region
    has the highest mean activation.  Returns a human-readable string
    like "lower right lobe" — used by the RAG pipeline.
    """
    h, w = cam.shape
    mid_h, mid_w = h // 2, w // 2

    quadrants = {
        "upper left lobe":  cam[:mid_h, :mid_w].mean(),
        "upper right lobe": cam[:mid_h, mid_w:].mean(),
        "lower left lobe":  cam[mid_h:, :mid_w].mean(),
        "lower right lobe": cam[mid_h:, mid_w:].mean(),
    }
    return max(quadrants, key=quadrants.get)


def overlay_heatmap(original_np: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Blends the Grad-CAM heatmap onto the original X-ray image.

    Args:
        original_np : (224, 224, 3) uint8 original image
        cam         : (224, 224) float32 heatmap in [0, 1]
        alpha       : heatmap opacity

    Returns:
        overlay: (224, 224, 3) uint8 blended image
    """
    heatmap = cv2.applyColorMap(
        (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
    )  # (224, 224, 3) BGR
    heatmap  = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    original = original_np.astype(np.float32)
    blended  = (1 - alpha) * original + alpha * heatmap.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def analyze_image(
    image_path: str,
    model: XRayClassifier,
    gradcam: GradCAM,
    device: torch.device,
    save_path: Optional[Path] = None,
) -> dict:
    """
    Full inference pipeline for one image.

    Returns a dict with:
        label      : "NORMAL" or "PNEUMONIA"
        probability: float (0–100 %)
        severity   : "high" | "moderate" | "low" confidence
        location   : e.g. "lower right lobe"
        query      : assembled FAISS query string
        overlay    : (224, 224, 3) uint8 numpy overlay image
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    img_pil   = Image.open(image_path).convert("RGB").resize((224, 224), Image.LANCZOS)
    img_np    = np.array(img_pil)
    img_t     = transform(img_pil).unsqueeze(0).to(device)

    cam, prob, pred = gradcam.generate(img_t)

    label    = "PNEUMONIA" if pred == 1 else "NORMAL"
    pct      = prob * 100 if pred == 1 else (1 - prob) * 100
    location = get_location_from_cam(cam) if pred == 1 else "N/A"

    if pct >= 85:
        severity = "high confidence, severe presentation"
    elif pct >= 65:
        severity = "moderate confidence, possible pneumonia"
    else:
        severity = "low confidence, borderline findings"

    query   = (f"{label} detected, {location}, {severity}"
               if pred == 1
               else "Normal chest X-ray, no opacity detected")
    overlay = overlay_heatmap(img_np, cam)

    result = {
        "label":       label,
        "probability": round(pct, 2),
        "severity":    severity,
        "location":    location,
        "query":       query,
        "overlay":     overlay,
        "cam":         cam,
    }

    if save_path:
        Image.fromarray(overlay).save(save_path)
        print(f"  🎨  Overlay saved → {save_path}")

    return result


def load_model(device: torch.device) -> XRayClassifier:
    ckpt_path = MODELS_DIR / "classifier.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {ckpt_path}.\n"
            "Run `python src/train_classifier.py` first."
        )
    model = build_model(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model   = load_model(device)
    gradcam = GradCAM(model, device)
    print("✅  Model + Grad-CAM ready")

    if args.image:
        save_path = OUTPUTS_DIR / f"gradcam_{Path(args.image).stem}.png"
        result    = analyze_image(args.image, model, gradcam, device, save_path)
        print(f"\n  Label      : {result['label']}")
        print(f"  Confidence : {result['probability']:.1f}%")
        print(f"  Location   : {result['location']}")
        print(f"  RAG Query  : {result['query']}")

    elif args.split:
        from dataset import XRayDataset
        ds        = XRayDataset(args.split, transform=None)
        count     = min(args.num_samples, len(ds.samples))
        samples   = ds.samples[:count]

        for i, (img_path, label) in enumerate(samples):
            save_path = OUTPUTS_DIR / f"gradcam_{args.split}_{i}.png"
            result    = analyze_image(str(img_path), model, gradcam, device, save_path)
            true_cls  = "NORMAL" if label == 0 else "PNEUMONIA"
            correct   = "✅" if result["label"] == true_cls else "❌"
            print(f"  [{i+1}/{count}] {correct} True:{true_cls:9s} "
                  f"Pred:{result['label']:9s} ({result['probability']:.1f}%)")

    gradcam.remove_hooks()
    print("\n✅  Grad-CAM analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoMed Grad-CAM")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, help="Path to single X-ray image")
    group.add_argument("--split",  type=str, choices=["train","val","test"],
                       help="Run on N samples from a dataset split")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples when using --split")
    args = parser.parse_args()
    main(args)
