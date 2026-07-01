"""Model module for multilabel image classification."""

from typing import Optional

import torch
import torch.nn as nn


class TinyConvBackbone(nn.Module):
    """Tiny local backbone for CPU smoke tests."""

    def __init__(self):
        super().__init__()
        self.num_features = 64
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.num_features, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


def _torchvision_models():
    try:
        import torchvision.models as models
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchvision is required for torchvision backbones. "
            "Use model.name='tiny_cnn' for the dependency-light smoke demo."
        ) from exc
    return models


def _get_backbone_features(backbone: nn.Module, model_name: str) -> int:
    """Return backbone feature dimension and strip classifier head."""
    name = model_name.lower()
    if hasattr(backbone, "num_features"):
        return backbone.num_features
    if "resnet" in name or "resnext" in name or "vit_b_16" in name:
        n = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return n
    if "efficientnet" in name:
        n = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        return n
    if "eva02" in name or "csatv2" in name:
        return backbone.num_features
    return 512


def _make_backbone(model_name: str, pretrained: bool) -> nn.Module:
    """Instantiate a backbone by name."""
    name = model_name.lower()
    if "tiny_cnn" in name:
        return TinyConvBackbone()
    models = _torchvision_models()
    if "resnet50" in name:
        return models.resnet50(pretrained=pretrained)
    if "resnet18" in name:
        return models.resnet18(pretrained=pretrained)
    if "resnet34" in name:
        return models.resnet34(pretrained=pretrained)
    if "resnet101" in name:
        return models.resnet101(pretrained=pretrained)
    if "resnext101_64x4d" in name:
        return models.resnext101_64x4d(pretrained=pretrained)
    if "vit_l_16" in name:
        return models.vit_l_16(pretrained=pretrained)
    if "csatv2" in name:
        import timm
        return timm.create_model(
            "csatv2_21m.sw_r640_in1k",
            pretrained=pretrained,
            num_classes=0,
        )
    if "eva02_large" in name: 
        if pretrained:
            import timm
            return timm.create_model(
                "timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
                pretrained=pretrained,
                num_classes=0,
            )
        else:
            from .eva import eva02_large_patch14_448
            return eva02_large_patch14_448(
                num_classes=0,
            )

    if "eva02" in name:
        if pretrained:
            import timm
            return timm.create_model(
                "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
                pretrained=pretrained,
                num_classes=0,
            )
        else:
            from .eva import eva02_base_patch14_448
            return eva02_base_patch14_448(
                num_classes=0,
            )
    if "efficientnet" in name:
        if "b0" in name:
            return models.efficientnet_b0(pretrained=pretrained)
        if "b1" in name:
            return models.efficientnet_b1(pretrained=pretrained)
        return models.efficientnet_v2_m(pretrained=pretrained)
    return models.resnet50(pretrained=pretrained)


class MultilabelClassifier(nn.Module):
    """Multilabel image classifier with FC head."""

    def __init__(
        self,
        num_classes: int,
        fc_activation: str,
        model_name: str = "resnet50",
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name

        self.backbone = _make_backbone(model_name, pretrained)
        num_features = _get_backbone_features(self.backbone, model_name)
        if fc_activation == "silu":
            fc_activation = nn.SiLU()
        elif fc_activation == "relu":
            fc_activation = nn.ReLU()
        elif fc_activation == "gelu":
            fc_activation = nn.GELU()
        else:
            raise ValueError(f"Invalid activation function: {fc_activation}")
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(num_features, 256),
            fc_activation,
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def load_checkpoint(self, checkpoint_path: str, device: torch.device):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                self.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                self.load_state_dict(checkpoint["state_dict"])
            else:
                self.load_state_dict(checkpoint)
        else:
            self.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")


def create_model(
    num_classes: int,
    model_name: str = "resnet50",
    pretrained: bool = True,
    fc_activation: str = "silu",
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    model = MultilabelClassifier(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
        fc_activation=fc_activation,
    )
    if checkpoint_path:
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        model.load_checkpoint(checkpoint_path, device)
    return model.to(dtype=torch.float32)
