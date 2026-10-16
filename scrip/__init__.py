"""Smoker Detection Package.

Reference:
    Khan, A., Khan, S., Hassan, B., & Zheng, Z. (2022).
    CNN-Based Smoker Classification and Detection in Smart City Application.
    Sensors, 22(3), 892. https://doi.org/10.3390/s22030892
"""

from scrip.config import ProjectConfig, config
from scrip.dataset import (
    KerasStyleAffineAugmentation,
    SmokerDataset,
    build_dataloaders,
    build_transforms,
    extract_label_from_filename,
    verify_dataset_integrity,
)
from scrip.model import (
    SmokerClassificationHead,
    SmokerInceptionResNetV2,
    build_smoker_model,
)
from scrip.train import (
    plot_training_history,
    train_one_epoch,
    validate_one_epoch,
)
from scrip.evaluate import (
    compute_all_metrics,
    run_test_inference,
)

__all__ = [
    "ProjectConfig",
    "config",
    "KerasStyleAffineAugmentation",
    "SmokerDataset",
    "build_dataloaders",
    "build_transforms",
    "extract_label_from_filename",
    "verify_dataset_integrity",
    "SmokerClassificationHead",
    "SmokerInceptionResNetV2",
    "build_smoker_model",
    "plot_training_history",
    "train_one_epoch",
    "validate_one_epoch",
    "compute_all_metrics",
    "run_test_inference",
]
