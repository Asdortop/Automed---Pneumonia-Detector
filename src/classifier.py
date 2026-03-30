"""
classifier.py — Phase 2
ResNet-50 with a custom MLP classification head for binary
chest X-ray diagnosis: NORMAL (0) vs PNEUMONIA (1).
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class XRayClassifier(nn.Module):
    """
    ResNet-50 backbone (pretrained on ImageNet) with a custom
    binary classification head.

    Frozen layers:  conv1, bn1, layer1, layer2, layer3
    Trainable:      layer4 + custom FC head

    Custom head: Linear(2048→512) → BN → ReLU → Dropout → Linear(512→1)
    Output is a raw logit — use BCEWithLogitsLoss during training,
    and sigmoid() during inference to get probability.
    """

    def __init__(self, dropout: float = 0.4, freeze_until: str = "layer3"):
        super().__init__()

        # ── Load pretrained backbone ──────────────────────────────────────
        base = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # ── Freeze early layers ───────────────────────────────────────────
        # We fine-tune layer4 and the custom head; freeze everything before.
        layers_to_freeze = ["conv1", "bn1", "layer1", "layer2", "layer3"]
        if freeze_until == "layer4":
            layers_to_freeze.append("layer4")

        for name, param in base.named_parameters():
            top = name.split(".")[0]
            if top in layers_to_freeze:
                param.requires_grad = False

        # ── Strip the original FC head ────────────────────────────────────
        self.backbone = nn.Sequential(*list(base.children())[:-1])   # output: (B, 2048, 1, 1)

        # ── Custom classification head ────────────────────────────────────
        # layer4 output features = 2048
        self.head = nn.Sequential(
            nn.Flatten(),                           # (B, 2048)
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),                      # raw logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)                 # (B, 2048, 1, 1)
        logit    = self.head(features)              # (B, 1)
        return logit.squeeze(1)                     # (B,)

    def get_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the 512-dim embedding from the penultimate layer.
        Used downstream by Grad-CAM and the RAG pipeline.
        """
        features = self.backbone(x)                 # (B, 2048, 1, 1)
        flat     = features.flatten(1)              # (B, 2048)
        # Pass through first two linear layers only
        feat_vec = self.head[:4](flat)              # up to ReLU → (B, 512)
        return feat_vec


def build_model(device: torch.device = None) -> XRayClassifier:
    """Convenience factory — builds model and moves to device."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XRayClassifier(dropout=0.4)
    return model.to(device)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(device)

    # Count trainable vs frozen params
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    print(f"Device      : {device}")
    print(f"Total params: {total:,}")
    print(f"Trainable   : {trainable:,} ({100*trainable/total:.1f}%)")
    print(f"Frozen      : {frozen:,} ({100*frozen/total:.1f}%)")

    # Forward pass sanity check
    dummy = torch.randn(4, 3, 224, 224).to(device)
    out   = model(dummy)
    feat  = model.get_feature_vector(dummy)
    print(f"Output shape: {out.shape}   (expected: torch.Size([4]))")
    print(f"Feature vec : {feat.shape}  (expected: torch.Size([4, 512]))")
    print("✅  Model architecture OK")
