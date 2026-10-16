"""Configuration module for Smoker Detection and Classification project.

Reference:
    Khan, A., Khan, S., Hassan, B., & Zheng, Z. (2022).
    CNN-Based Smoker Classification and Detection in Smart City Application.
    Sensors, 22(3), 892. https://doi.org/10.3390/s22030892
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import torch


@dataclass
class DatasetConfig:
    """Dataset and data augmentation hyperparameters according to Section 5."""

    # Project directories
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(init=False)
    train_dir: Path = field(init=False)
    val_dir: Path = field(init=False)
    test_dir: Path = field(init=False)

    # Class mappings
    classes: List[str] = field(default_factory=lambda: ["NotSmoking", "Smoking"])
    class_to_idx: Dict[str, int] = field(
        default_factory=lambda: {"notsmoking": 0, "smoking": 1}
    )
    idx_to_class: Dict[int, str] = field(
        default_factory=lambda: {0: "NotSmoking", 1: "Smoking"}
    )

    # Image properties
    raw_size: Tuple[int, int] = (250, 250)      # Preprocessed size in dataset (Section 5.1)
    image_size: Tuple[int, int] = (224, 224)    # Inception-ResNet-V2 input size (Section 5.2)
    channels: int = 3

    # Data Augmentation parameters (Section 5.2)
    # - Random rotation up to 50 degrees
    # - Random zoom/scale factor up to 0.2 (0.8 - 1.2)
    # - Horizontal and vertical translation factor up to 0.2
    # - Shear transformation up to a factor of 0.2 (~11.31 degrees)
    # - Random horizontal flip
    rotation_degrees: float = 50.0
    scale_range: Tuple[float, float] = (0.8, 1.2)
    translation_factor: Tuple[float, float] = (0.2, 0.2)
    shear_degrees: float = 11.31  # arctan(0.2) in degrees
    horizontal_flip_prob: float = 0.5
    fill_mode: str = "nearest"  # Edge replication matching Keras ImageDataGenerator (Section 5.2 & Figure 5)

    # Image normalization statistics (ImageNet standard)
    norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        """Initialize resolved paths."""
        self.data_dir = self.project_root / "data"
        self.train_dir = self.data_dir / "Training"
        self.val_dir = self.data_dir / "Validation"
        self.test_dir = self.data_dir / "Testing"


@dataclass
class TrainingConfig:
    """Training hyperparameters according to Table 2 (Section 6.1)."""

    # Optimizer and Loss
    optimizer_name: str = "SGD"
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    loss_function: str = "BinaryCrossEntropy"

    # Batching and Epochs
    batch_size: int = 64
    epochs: int = 50
    steps_per_epoch: int = 100
    validation_steps: int = 100

    # System parameters
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42

    # Hardware device
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


@dataclass
class ModelConfig:
    """Model architecture parameters according to Section 4.1 & Figure 3."""

    backbone_name: str = "inception_resnet_v2"
    pretrained: bool = True
    freeze_backbone: bool = True

    # Custom classification head dense layers
    # Figure 3: Feature vector (1536) -> Dense(1536) -> Dense(1536) -> Dense(512) -> Dense(1)
    dense_layers: Tuple[int, ...] = (1536, 1536, 512)
    dropout_rate: float = 0.5
    num_classes: int = 1  # Binary output (Sigmoid)


@dataclass
class ProjectConfig:
    """Master configuration uniting all sub-configs."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    # Artifact directories
    model_save_dir: Path = field(init=False)
    output_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        """Ensure necessary output folders exist."""
        self.model_save_dir = self.dataset.project_root / "model"
        self.output_dir = self.dataset.project_root / "outputs"

        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def print_summary(self) -> None:
        """Display formatted configuration summary."""
        print("=" * 65)
        print("  SMOKER DETECTION SYSTEM - PROJECT CONFIGURATION")
        print("=" * 65)
        print(f"[Device]           : {self.training.device.upper()}")
        print(f"[Project Root]     : {self.dataset.project_root}")
        print(f"[Train Data Dir]   : {self.dataset.train_dir}")
        print(f"[Val Data Dir]     : {self.dataset.val_dir}")
        print(f"[Test Data Dir]    : {self.dataset.test_dir}")
        print(f"[Output Dir]       : {self.output_dir}")
        print("-" * 65)
        print(f"[Input Resolution] : {self.dataset.image_size}")
        print(f"[Batch Size]       : {self.training.batch_size}")
        print(f"[Epochs]           : {self.training.epochs}")
        print(f"[Optimizer]        : {self.training.optimizer_name} (lr={self.training.learning_rate})")
        print(f"[Backbone]         : {self.model.backbone_name} (frozen={self.model.freeze_backbone})")
        print(f"[Dense Head]       : {self.model.dense_layers} -> Output(1)")
        print("=" * 65)


# Global default configuration instance
config = ProjectConfig()


if __name__ == "__main__":
    config.print_summary()
