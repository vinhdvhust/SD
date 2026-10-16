"""Inception-ResNet-V2 transfer learning architecture for Smoker Detection.

This module implements the complete deep neural network architecture proposed in:
    Khan, A., Khan, S., Hassan, B., & Zheng, Z. (2022).
    CNN-Based Smoker Classification and Detection in Smart City Application.
    Sensors, 22(3), 892. https://doi.org/10.3390/s22030892

Architectural Specifications (Section 4.1, Section 4.2 & Figure 3):
    1. Pretrained Inception-ResNet-V2 convolutional backbone (164 layers deep).
    2. Convolutional base weights frozen to preserve generic visual representations.
    3. Output feature map from Inception-ResNet-C / Reduction-B: (5x5x1536).
    4. Global Average Pooling (or Flatten) -> 1536-dimensional feature vector.
    5. Custom Fully-Connected (FC) Classification Head:
       - Dense Layer 1: 1536 neurons, ReLU activation, Dropout
       - Dense Layer 2: 1536 neurons, ReLU activation, Dropout
       - Dense Layer 3: 512 neurons, ReLU activation, Dropout
       - Output Layer: 1 neuron, binary classification output (Logits / Sigmoid)
"""

from typing import Dict, Optional, Tuple, Union
import timm
import torch
import torch.nn as nn

try:
    from scrip.config import ProjectConfig, config as default_config
except ImportError:
    from config import ProjectConfig, config as default_config


class SmokerClassificationHead(nn.Module):
    """Custom classification head as specified in Section 4.1 & Figure 3.

    Structure:
        Input (in_features = 1536)
        -> Linear(1536, 1536) -> ReLU -> Dropout(p)
        -> Linear(1536, 1536) -> ReLU -> Dropout(p)
        -> Linear(1536, 512)  -> ReLU -> Dropout(p)
        -> Linear(512, 1)     -> Logits (Sigmoid applied in loss/predict)
    """

    def __init__(
        self,
        in_features: int = 1536,
        dense_layers: Tuple[int, ...] = (1536, 1536, 512),
        dropout_rate: float = 0.5,
        num_classes: int = 1,
    ) -> None:
        """Initialize classification head layers.

        Args:
            in_features: Dimensionality of backbone output feature vector (1536).
            dense_layers: Sequence of hidden dense layer neuron counts.
            dropout_rate: Dropout regularization probability.
            num_classes: Number of target outputs (1 for binary classification).
        """
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = in_features

        for hidden_dim in dense_layers:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout_rate),
                ]
            )
            prev_dim = hidden_dim

        # Final projection layer to produce binary logits
        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize linear layer weights using He/Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classification head.

        Args:
            x: Input feature tensor of shape (batch_size, 1536).

        Returns:
            torch.Tensor: Raw logits tensor of shape (batch_size, 1).
        """
        return self.classifier(x)


class SmokerInceptionResNetV2(nn.Module):
    """Full transfer learning model for smoker detection.

    Combines pretrained Inception-ResNet-V2 convolutional backbone with
    custom fully connected dense layers.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.5,
        dense_layers: Tuple[int, ...] = (1536, 1536, 512),
        num_classes: int = 1,
    ) -> None:
        """Initialize Inception-ResNet-V2 smoker classifier.

        Args:
            pretrained: Whether to load ImageNet pretrained weights.
            freeze_backbone: Whether to freeze convolutional backbone weights.
            dropout_rate: Dropout probability for classification head.
            dense_layers: Hidden layer dimensions for classification head.
            num_classes: Number of outputs (1 for binary classification).
        """
        super().__init__()

        # Load Inception-ResNet-V2 with pooled feature representation (1536 dimensions)
        self.backbone = timm.create_model(
            "inception_resnet_v2",
            pretrained=pretrained,
            num_classes=0,  # Strips default 1000-class ImageNet head, applies global average pooling
        )

        in_features = self.backbone.num_features  # 1536

        # Custom classification head (Section 4.1 & Figure 3)
        self.head = SmokerClassificationHead(
            in_features=in_features,
            dense_layers=dense_layers,
            dropout_rate=dropout_rate,
            num_classes=num_classes,
        )

        # Freeze backbone weights if requested
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze all parameters in the convolutional backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, num_layers: Optional[int] = None) -> None:
        """Unfreeze backbone parameters for fine-tuning.

        Args:
            num_layers: If None, unfreezes entire backbone.
                        If specified, unfreezes only the last `num_layers` parameters.
        """
        params = list(self.backbone.parameters())
        if num_layers is None:
            for p in params:
                p.requires_grad = True
        else:
            for p in params[-num_layers:]:
                p.requires_grad = True

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature maps from the backbone before global pooling.

        Args:
            x: Input image tensor of shape (batch_size, 3, 224, 224).

        Returns:
            torch.Tensor: Feature map tensor of shape (batch_size, 1536, 5, 5).
        """
        return self.backbone.forward_features(x)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Perform forward pass.

        Args:
            x: Input image tensor of shape (batch_size, 3, 224, 224).
            return_features: If True, also returns pooled 1536-dimensional feature vector.

        Returns:
            torch.Tensor: Raw logits tensor of shape (batch_size, 1).
            (Optional) torch.Tensor: Pooled feature vector of shape (batch_size, 1536).
        """
        # Feature extraction through Inception-ResNet-V2 convolutional base
        features = self.backbone(x)  # Shape: (batch_size, 1536)

        # Classification through custom dense head
        logits = self.head(features)  # Shape: (batch_size, 1)

        if return_features:
            return logits, features
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute calibrated probability of Smoking (class 1) via Sigmoid activation.

        Formula (Section 4.2, Equation 2):
            Sigmoid(x) = 1 / (1 + e^(-x))

        Args:
            x: Input image tensor of shape (batch_size, 3, 224, 224).

        Returns:
            torch.Tensor: Probabilities in range [0, 1] of shape (batch_size, 1).
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def get_num_parameters(self) -> Dict[str, int]:
        """Compute breakdown of trainable, frozen, and total parameters.

        Returns:
            Dict[str, int]: Counts of total, trainable, and frozen parameters.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
        }

    def print_summary(
        self,
        input_size: Tuple[int, int, int, int] = (2, 3, 224, 224),
    ) -> None:
        """Display comprehensive model summary and layer breakdown.

        Args:
            input_size: Sample batch shape for forward pass verification.
        """
        params = self.get_num_parameters()
        device = next(self.parameters()).device

        print("=" * 75)
        print("  INCEPTION-RESNET-V2 SMOKER CLASSIFIER - ARCHITECTURE SUMMARY")
        print("=" * 75)
        print(f"[Backbone]             : Inception-ResNet-V2 (164 layers deep, pretrained)")
        print(f"[Device]               : {str(device).upper()}")
        print(f"[Input Resolution]     : {input_size[1:]} (Batch Size = {input_size[0]})")
        print(f"[Feature Map Out]      : (1536, 5, 5) -> Global Avg Pool -> (1536,)")
        print("-" * 75)
        print("Custom Classification Head (Section 4.1 & Figure 3):")
        print("  1. Linear(1536 -> 1536)  + ReLU + Dropout(p=0.5)")
        print("  2. Linear(1536 -> 1536)  + ReLU + Dropout(p=0.5)")
        print("  3. Linear(1536 -> 512)   + ReLU + Dropout(p=0.5)")
        print("  4. Linear(512  -> 1)     + Sigmoid (Binary Output: NotSmoking=0, Smoking=1)")
        print("-" * 75)
        print(f"[Total Parameters]     : {params['total']:,}")
        print(f"[Frozen Parameters]    : {params['frozen']:,} (Conv base feature extractor)")
        print(f"[Trainable Parameters] : {params['trainable']:,} (Custom FC dense head)")
        print("-" * 75)

        # Verification forward pass
        self.eval()
        dummy_input = torch.randn(*input_size, device=device)
        with torch.no_grad():
            dummy_logits = self.forward(dummy_input)
            dummy_probs = self.predict_proba(dummy_input)

        print(f"[Forward Test Output]  : Logits shape = {tuple(dummy_logits.shape)}")
        print(f"[Sigmoid Probs Output] : Range = [{dummy_probs.min().item():.4f}, {dummy_probs.max().item():.4f}]")
        print("=" * 75 + "\n")


def build_smoker_model(
    cfg: Optional[ProjectConfig] = None,
) -> SmokerInceptionResNetV2:
    """Factory function to build and configure the SmokerInceptionResNetV2 model.

    Args:
        cfg: Optional project configuration instance.

    Returns:
        SmokerInceptionResNetV2: Configured model ready for training or evaluation.
    """
    if cfg is None:
        cfg = default_config

    m_cfg = cfg.model
    model = SmokerInceptionResNetV2(
        pretrained=m_cfg.pretrained,
        freeze_backbone=m_cfg.freeze_backbone,
        dropout_rate=m_cfg.dropout_rate,
        dense_layers=m_cfg.dense_layers,
        num_classes=m_cfg.num_classes,
    )

    # Move to designated target device (e.g. cuda)
    device = torch.device(cfg.training.device)
    model.to(device)

    return model


if __name__ == "__main__":
    # 1. Instantiate model via factory
    smoker_model = build_smoker_model(default_config)

    # 2. Print architecture summary and test forward pass
    smoker_model.print_summary()

    # 3. Assert parameter integrity
    param_counts = smoker_model.get_num_parameters()
    assert param_counts["trainable"] == 5_509_121, (
        f"Unexpected trainable parameter count: {param_counts['trainable']}"
    )
    assert param_counts["frozen"] > 50_000_000, (
        f"Backbone was not properly frozen! Frozen params: {param_counts['frozen']}"
    )
