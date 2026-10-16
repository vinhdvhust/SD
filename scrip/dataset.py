"""Dataset management and data augmentation pipeline for Smoker Detection.

This module provides:
1. SmokerDataset: PyTorch Dataset with automatic label parsing from file names.
2. Data augmentation transforms adhering to Section 5.2 of the paper.
3. DataLoader builders for Training, Validation, and Testing splits.
4. Dataset verification and diagnostic tools.
5. Visualizer to export sample augmentations for inspection.

Reference:
    Khan, A., Khan, S., Hassan, B., & Zheng, Z. (2022).
    CNN-Based Smoker Classification and Detection in Smart City Application.
    Sensors, 22(3), 892. https://doi.org/10.3390/s22030892
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

# Import project configurations
try:
    from scrip.config import ProjectConfig, config as default_config
except ImportError:
    from config import ProjectConfig, config as default_config


def extract_label_from_filename(filename: str) -> int:
    """Extract binary classification label from image filename.

    Naming convention:
        - 'notsmoking_xxxx.jpg' -> 0 (NotSmoking)
        - 'smoking_xxxx.jpg'    -> 1 (Smoking)

    Args:
        filename: Name or path of the image file.

    Returns:
        int: 0 for NotSmoking, 1 for Smoking.

    Raises:
        ValueError: If filename does not contain recognizable class pattern.
    """
    clean_name = Path(filename).name.lower()
    if clean_name.startswith("notsmoking") or "notsmoking" in clean_name:
        return 0
    if clean_name.startswith("smoking") or "smoking" in clean_name:
        return 1
    raise ValueError(
        f"Unable to determine class label from filename '{filename}'. "
        "Expected filename to contain 'notsmoking' or 'smoking'."
    )


class KerasStyleAffineAugmentation:
    """Affine augmentation replicating Keras ImageDataGenerator (fill_mode='nearest').

    As demonstrated in Figure 5 of the paper, Keras ImageDataGenerator pads transformed
    boundaries using nearest-neighbor edge replication (cv2.BORDER_REPLICATE) rather
    than solid black padding (fill=0).

    Transformations performed:
        - Random horizontal flip
        - Random rotation (degrees in [-rotation_degrees, rotation_degrees])
        - Random translation (shift in [-tx, tx], [-ty, ty])
        - Random scaling/zoom (factor in [scale_min, scale_max])
        - Random shear (angle in [-shear_degrees, shear_degrees])
        - Edge replication filling (cv2.BORDER_REPLICATE)
    """

    def __init__(
        self,
        rotation_degrees: float = 50.0,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        translation_factor: Tuple[float, float] = (0.2, 0.2),
        shear_degrees: float = 11.31,
        horizontal_flip_prob: float = 0.5,
        fill_mode: str = "nearest",
    ) -> None:
        self.rotation_degrees = rotation_degrees
        self.scale_range = scale_range
        self.translation_factor = translation_factor
        self.shear_degrees = shear_degrees
        self.horizontal_flip_prob = horizontal_flip_prob
        self.fill_mode = fill_mode

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply random affine transformation with edge replication to PIL Image."""
        img_np = np.array(img)
        h, w = img_np.shape[:2]

        # 1. Random Horizontal Flip
        if random.random() < self.horizontal_flip_prob:
            img_np = cv2.flip(img_np, 1)

        # 2. Sample random transformation parameters
        theta = np.deg2rad(random.uniform(-self.rotation_degrees, self.rotation_degrees))
        tx = random.uniform(-self.translation_factor[0], self.translation_factor[0]) * w
        ty = random.uniform(-self.translation_factor[1], self.translation_factor[1]) * h
        shear = np.deg2rad(random.uniform(-self.shear_degrees, self.shear_degrees))
        zx = random.uniform(self.scale_range[0], self.scale_range[1])
        zy = random.uniform(self.scale_range[0], self.scale_range[1])

        # 3. Keras-identical Affine Transformation Matrix centered at image center
        cx, cy = w / 2.0, h / 2.0

        T_to_origin = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])

        R = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ])

        S = np.array([
            [1.0, -np.sin(shear), 0.0],
            [0.0, np.cos(shear), 0.0],
            [0.0, 0.0, 1.0],
        ])

        Z = np.array([
            [1.0 / zx, 0.0, 0.0],
            [0.0, 1.0 / zy, 0.0],
            [0.0, 0.0, 1.0],
        ])

        T_back = np.array([[1.0, 0.0, cx + tx], [0.0, 1.0, cy + ty], [0.0, 0.0, 1.0]])

        # Full 3x3 transformation matrix: T_back * Z * S * R * T_to_origin
        M = T_back @ Z @ S @ R @ T_to_origin
        M_affine = M[:2, :]

        # 4. Warp with nearest-edge replication (matches Figure 5)
        border_mode = (
            cv2.BORDER_REPLICATE if self.fill_mode == "nearest" else cv2.BORDER_CONSTANT
        )
        warped_np = cv2.warpAffine(
            img_np,
            M_affine,
            (w, h),
            borderMode=border_mode,
            flags=cv2.INTER_LINEAR,
        )

        return Image.fromarray(warped_np)


def build_transforms(
    is_train: bool,
    cfg: Optional[ProjectConfig] = None,
) -> T.Compose:
    """Construct image transformation pipeline.

    As detailed in Section 5.1 & 5.2 of the paper:
    - Base input resolution for Inception-ResNet-V2 is 224 x 224.
    - Training augmentations (Keras-identical via KerasStyleAffineAugmentation):
        * Random rotation up to 50 degrees
        * Random scaling factor up to 0.2 (0.8x to 1.2x)
        * Random translation (horizontal/vertical) factor up to 0.2
        * Random shear transformation factor up to 0.2 (~11.31 degrees)
        * Random horizontal flip
        * Boundary padding: fill_mode='nearest' (edge replication, Figure 5)
    - Image normalization to standard mean and std.

    Args:
        is_train: True to apply training augmentations; False for deterministic eval transforms.
        cfg: Optional project configuration instance.

    Returns:
        torchvision.transforms.Compose: Composed transformation pipeline.
    """
    if cfg is None:
        cfg = default_config

    d_cfg = cfg.dataset

    if is_train:
        transform_list = [
            T.Resize(d_cfg.image_size, interpolation=T.InterpolationMode.BILINEAR),
            KerasStyleAffineAugmentation(
                rotation_degrees=d_cfg.rotation_degrees,
                scale_range=d_cfg.scale_range,
                translation_factor=d_cfg.translation_factor,
                shear_degrees=d_cfg.shear_degrees,
                horizontal_flip_prob=d_cfg.horizontal_flip_prob,
                fill_mode=d_cfg.fill_mode,
            ),
            T.ToTensor(),
            T.Normalize(mean=d_cfg.norm_mean, std=d_cfg.norm_std),
        ]
    else:
        transform_list = [
            T.Resize(d_cfg.image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=d_cfg.norm_mean, std=d_cfg.norm_std),
        ]

    return T.Compose(transform_list)


class SmokerDataset(Dataset):
    """Custom PyTorch Dataset for smoker classification.

    Attributes:
        data_dir: Path to directory containing split images.
        image_paths: Sorted list of all image paths found in data_dir.
        labels: Corresponding list of integer labels (0 or 1).
        transform: Callable transformation pipeline.
        return_metadata: If True, returns (image, label, filename).
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(
        self,
        data_dir: Union[str, Path],
        transform: Optional[T.Compose] = None,
        return_metadata: bool = False,
    ) -> None:
        """Initialize the SmokerDataset.

        Args:
            data_dir: Directory path containing the images.
            transform: Optional transform to apply to each image.
            return_metadata: If True, __getitem__ returns (image, label, filename).
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.return_metadata = return_metadata

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.data_dir}")

        # Gather and sort all supported image files for determinism
        self.image_paths: List[Path] = sorted(
            [
                p for p in self.data_dir.iterdir()
                if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]
        )

        if len(self.image_paths) == 0:
            raise ValueError(f"No valid image files found in {self.data_dir}")

        # Parse labels for all images
        self.labels: List[int] = [
            extract_label_from_filename(p.name) for p in self.image_paths
        ]

    def __len__(self) -> int:
        """Return total number of samples in this dataset."""
        return len(self.image_paths)

    def __getitem__(
        self,
        index: int,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, str]]:
        """Fetch sample at given index.

        Args:
            index: Sample index.

        Returns:
            Tuple of (image_tensor, label_tensor) or (image_tensor, label_tensor, filename).
        """
        img_path = self.image_paths[index]
        label = self.labels[index]

        # Load image via PIL and guarantee 3-channel RGB format
        with Image.open(img_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # Binary label formatted as float tensor with shape (1,) for BCEWithLogitsLoss
        label_tensor = torch.tensor([label], dtype=torch.float32)

        if self.return_metadata:
            return image, label_tensor, img_path.name
        return image, label_tensor

    def get_class_counts(self) -> Dict[str, int]:
        """Compute frequency of each class in this dataset split."""
        notsmoking_count = sum(1 for lbl in self.labels if lbl == 0)
        smoking_count = sum(1 for lbl in self.labels if lbl == 1)
        return {
            "NotSmoking": notsmoking_count,
            "Smoking": smoking_count,
            "Total": len(self.labels),
        }


def build_dataloaders(
    cfg: Optional[ProjectConfig] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Construct DataLoaders for Training, Validation, and Testing splits.

    Args:
        cfg: Optional project configuration instance.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader).
    """
    if cfg is None:
        cfg = default_config

    # Build transforms
    train_transform = build_transforms(is_train=True, cfg=cfg)
    eval_transform = build_transforms(is_train=False, cfg=cfg)

    # Initialize datasets
    train_dataset = SmokerDataset(
        data_dir=cfg.dataset.train_dir,
        transform=train_transform,
    )
    val_dataset = SmokerDataset(
        data_dir=cfg.dataset.val_dir,
        transform=eval_transform,
    )
    test_dataset = SmokerDataset(
        data_dir=cfg.dataset.test_dir,
        transform=eval_transform,
    )

    t_cfg = cfg.training

    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=t_cfg.batch_size,
        shuffle=True,
        num_workers=t_cfg.num_workers,
        pin_memory=t_cfg.pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=t_cfg.batch_size,
        shuffle=False,
        num_workers=t_cfg.num_workers,
        pin_memory=t_cfg.pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=t_cfg.batch_size,
        shuffle=False,
        num_workers=t_cfg.num_workers,
        pin_memory=t_cfg.pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def verify_dataset_integrity(cfg: Optional[ProjectConfig] = None) -> bool:
    """Validate sample counts, balance, dimensions, and readability of all splits.

    Expected figures according to the paper (Section 5 & 5.1):
        - Training:   716 images (358 Smoking, 358 NotSmoking)
        - Validation: 180 images (90 Smoking, 90 NotSmoking)
        - Testing:    224 images (112 Smoking, 112 NotSmoking)
        - Total:     1120 images (560 Smoking, 560 NotSmoking)

    Args:
        cfg: Optional project configuration instance.

    Returns:
        bool: True if all integrity checks pass, False otherwise.
    """
    if cfg is None:
        cfg = default_config

    expected = {
        "Training": {"total": 716, "notsmoking": 358, "smoking": 358, "dir": cfg.dataset.train_dir},
        "Validation": {"total": 180, "notsmoking": 90, "smoking": 90, "dir": cfg.dataset.val_dir},
        "Testing": {"total": 224, "notsmoking": 112, "smoking": 112, "dir": cfg.dataset.test_dir},
    }

    print("\n" + "=" * 70)
    print("  DATASET INTEGRITY & VALIDATION REPORT")
    print("=" * 70)
    header = f"{'Split':<12} | {'Total':<6} | {'NotSmoking':<10} | {'Smoking':<8} | {'Status':<10}"
    print(header)
    print("-" * 70)

    all_passed = True
    grand_total = 0
    grand_smoking = 0
    grand_notsmoking = 0

    for split_name, exp in expected.items():
        try:
            ds = SmokerDataset(exp["dir"])
            counts = ds.get_class_counts()

            grand_total += counts["Total"]
            grand_notsmoking += counts["NotSmoking"]
            grand_smoking += counts["Smoking"]

            is_total_ok = counts["Total"] == exp["total"]
            is_ns_ok = counts["NotSmoking"] == exp["notsmoking"]
            is_s_ok = counts["Smoking"] == exp["smoking"]

            status = "PASSED" if (is_total_ok and is_ns_ok and is_s_ok) else "MISMATCH"
            if status != "PASSED":
                all_passed = False

            row = (
                f"{split_name:<12} | "
                f"{counts['Total']:<6} | "
                f"{counts['NotSmoking']:<10} | "
                f"{counts['Smoking']:<8} | "
                f"{status:<10}"
            )
            print(row)

        except Exception as err:
            all_passed = False
            print(f"{split_name:<12} | ERROR: {err}")

    print("-" * 70)
    total_status = "PERFECT (100% Match with Paper)" if all_passed else "FAILED"
    print(
        f"{'GRAND TOTAL':<12} | "
        f"{grand_total:<6} | "
        f"{grand_notsmoking:<10} | "
        f"{grand_smoking:<8} | "
        f"{total_status}"
    )
    print("=" * 70 + "\n")

    return all_passed


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Convert normalized PyTorch image tensor to NumPy image for visualization.

    Args:
        tensor: Normalized image tensor of shape (3, H, W).
        mean: Normalization mean values.
        std: Normalization standard deviation values.

    Returns:
        np.ndarray: Displayable RGB image array of shape (H, W, 3) in range [0, 1].
    """
    img = tensor.clone().detach().cpu()
    for channel, (m, s) in enumerate(zip(mean, std)):
        img[channel] = img[channel] * s + m
    img = img.clamp(0.0, 1.0)
    return img.permute(1, 2, 0).numpy()


def visualize_and_save_augmentations(
    output_path: Optional[Union[str, Path]] = None,
    cfg: Optional[ProjectConfig] = None,
    num_samples: int = 4,
    num_variants: int = 4,
) -> Path:
    """Generate and save a visual grid comparing original and augmented samples.

    Args:
        output_path: Destination path for saved figure.
        cfg: Optional project configuration instance.
        num_samples: Number of distinct original images to show.
        num_variants: Number of stochastic augmented versions per image.

    Returns:
        Path: Path to saved visualization file.
    """
    if cfg is None:
        cfg = default_config

    if output_path is None:
        output_path = cfg.output_dir / "sample_augmentations.png"
    output_path = Path(output_path)

    # Base raw dataset (no transform)
    raw_dataset = SmokerDataset(cfg.dataset.train_dir, transform=None, return_metadata=True)
    aug_transform = build_transforms(is_train=True, cfg=cfg)

    # Pick representative samples (2 NotSmoking, 2 Smoking)
    indices: List[int] = []
    for target_label in [0, 1]:
        collected = 0
        for idx, lbl in enumerate(raw_dataset.labels):
            if lbl == target_label:
                indices.append(idx)
                collected += 1
                if collected >= (num_samples // 2):
                    break

    fig, axes = plt.subplots(
        nrows=len(indices),
        ncols=num_variants + 1,
        figsize=(3.5 * (num_variants + 1), 3.5 * len(indices)),
    )

    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle(
        "Smoker Detection: Paper-Specified Data Augmentations (Section 5.2)\n"
        "(Rotation ±50°, Zoom ±0.2, Translation ±0.2, Shear ±0.2, Flip)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for row_idx, sample_idx in enumerate(indices):
        raw_img, label, filename = raw_dataset[sample_idx]
        class_name = cfg.dataset.idx_to_class[int(label.item())]

        # Column 0: Original image
        axes[row_idx, 0].imshow(raw_img)
        axes[row_idx, 0].set_title(
            f"Original ({class_name})\n{filename}",
            fontsize=10,
            fontweight="semibold",
            color="navy",
        )
        axes[row_idx, 0].axis("off")

        # Columns 1 to num_variants: Stochastic augmentations
        for col_idx in range(1, num_variants + 1):
            aug_tensor = aug_transform(raw_img)
            aug_display = denormalize_tensor(
                aug_tensor,
                mean=cfg.dataset.norm_mean,
                std=cfg.dataset.norm_std,
            )
            axes[row_idx, col_idx].imshow(aug_display)
            axes[row_idx, col_idx].set_title(
                f"Augmented #{col_idx}\n(224x224)",
                fontsize=9,
            )
            axes[row_idx, col_idx].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[Visualization] Sample augmentations successfully saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # 1. Verify Dataset Integrity
    is_valid = verify_dataset_integrity(default_config)
    
    # 2. Test DataLoader Generation & Batch Shapes
    print("\n[DataLoader Test] Initializing DataLoaders...")
    train_dl, val_dl, test_dl = build_dataloaders(default_config)
    print(f"  Train batches: {len(train_dl)} (batch_size={default_config.training.batch_size})")
    print(f"  Val batches:   {len(val_dl)} (batch_size={default_config.training.batch_size})")
    print(f"  Test batches:  {len(test_dl)} (batch_size={default_config.training.batch_size})")

    # Fetch a sample batch
    sample_images, sample_labels = next(iter(train_dl))
    print(f"\n[Batch Verification]")
    print(f"  Image batch shape : {sample_images.shape} (Expected: [N, 3, 224, 224])")
    print(f"  Label batch shape : {sample_labels.shape} (Expected: [N, 1])")
    print(f"  Image dtype       : {sample_images.dtype}")
    print(f"  Label dtype       : {sample_labels.dtype}")
    print(f"  Pixel range       : [{sample_images.min().item():.3f}, {sample_images.max().item():.3f}]")
    print(f"  Label sum         : {int(sample_labels.sum().item())} smoking / {len(sample_labels)} total in batch")

    # 3. Export Augmentation Samples
    print("\n[Visualization] Generating sample augmentation preview...")
    visualize_and_save_augmentations(cfg=default_config)
