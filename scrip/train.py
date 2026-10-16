"""Training pipeline for Inception-ResNet-V2 Smoker Detection.

This script implements Phase 3 of the project plan adhering to Section 6.1 (Table 2) of:
    Khan, A., Khan, S., Hassan, B., & Zheng, Z. (2022).
    CNN-Based Smoker Classification and Detection in Smart City Application.
    Sensors, 22(3), 892. https://doi.org/10.3390/s22030892

Training Hyperparameters:
    - Input size: 224 x 224 x 3
    - Optimizer: Stochastic Gradient Descent (SGD) with momentum 0.9
    - Loss Function: Binary Cross-Entropy (BCEWithLogitsLoss)
    - Learning rate: 0.01
    - Batch Size: 64
    - Epochs: 50
    - Model Checkpoint: Saves weights yielding highest validation accuracy / lowest loss
"""

import argparse
import json
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI environments
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from scrip.config import ProjectConfig, config as default_config
    from scrip.dataset import build_dataloaders
    from scrip.model import SmokerInceptionResNetV2, build_smoker_model
except ImportError:
    from config import ProjectConfig, config as default_config
    from dataset import build_dataloaders
    from model import SmokerInceptionResNetV2, build_smoker_model


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    """Parse command line training arguments."""
    parser = argparse.ArgumentParser(
        description="Train Inception-ResNet-V2 Smoker Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=default_config.training.epochs,
        help="Number of training epochs (paper specifies 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=default_config.training.batch_size,
        help="Batch size for training and validation (paper specifies 64)",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Optional number of batch gradient steps per epoch (e.g. 100). Default is None (iterates through full dataset once per epoch).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=default_config.training.learning_rate,
        help="Initial learning rate for SGD optimizer (paper specifies 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=default_config.training.momentum,
        help="Momentum factor for SGD optimizer",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=default_config.training.weight_decay,
        help="Weight decay (L2 penalty) factor",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=default_config.model.dropout_rate,
        help="Dropout probability for classification head",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_config.training.device,
        help="Compute device ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=default_config.training.num_workers,
        help="Number of DataLoader worker processes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=default_config.training.seed,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(default_config.model_save_dir),
        help="Directory to save checkpoint weights",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_config.output_dir),
        help="Directory to save logs and training curves",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pth file to resume training from",
    )
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    steps_per_epoch: Optional[int] = None,
    data_iter: Optional[Any] = None,
) -> Tuple[float, float, Any]:
    """Execute one training epoch.

    Args:
        model: SmokerInceptionResNetV2 neural network.
        dataloader: Training DataLoader with on-the-fly augmentation.
        criterion: Binary cross-entropy loss module (BCEWithLogitsLoss).
        optimizer: SGD optimizer updating classification head weights.
        device: Target hardware device (cuda or cpu).
        epoch: Current epoch index (1-indexed).
        total_epochs: Total number of epochs.
        steps_per_epoch: Optional number of steps per epoch (e.g. 100 as in paper Table 2).
                         If None or <= 0, iterates once through full dataset.
        data_iter: Persistent DataLoader iterator across epochs when steps_per_epoch is used.

    Returns:
        Tuple[float, float, Any]: (average_loss, accuracy_percentage, data_iter).
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    use_fixed_steps = steps_per_epoch is not None and steps_per_epoch > 0

    if use_fixed_steps:
        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"Epoch [{epoch:02d}/{total_epochs:02d}] Training",
            leave=False,
            unit="step",
        )
        for _ in pbar:
            if data_iter is None:
                data_iter = iter(dataloader)
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                images, labels = next(data_iter)

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass: obtain raw logits
            logits = model(images)
            loss = criterion(logits, labels)

            # Backward pass & weight update
            loss.backward()
            optimizer.step()

            # Compute running statistics
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            # Binary prediction threshold = 0.5 (logits >= 0.0 corresponds to sigmoid >= 0.5)
            predictions = (logits >= 0.0).float()
            correct_predictions += (predictions == labels).sum().item()
            total_samples += batch_size

            batch_acc = (correct_predictions / total_samples) * 100.0
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{batch_acc:.2f}%"})
    else:
        pbar = tqdm(
            dataloader,
            desc=f"Epoch [{epoch:02d}/{total_epochs:02d}] Training",
            leave=False,
            unit="batch",
        )
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            predictions = (logits >= 0.0).float()
            correct_predictions += (predictions == labels).sum().item()
            total_samples += batch_size

            batch_acc = (correct_predictions / total_samples) * 100.0
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{batch_acc:.2f}%"})

    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_predictions / total_samples) * 100.0
    return epoch_loss, epoch_acc, data_iter


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    """Execute one validation epoch.

    Args:
        model: SmokerInceptionResNetV2 neural network.
        dataloader: Validation DataLoader (deterministic eval transform).
        criterion: Binary cross-entropy loss module (BCEWithLogitsLoss).
        device: Target hardware device (cuda or cpu).
        epoch: Current epoch index (1-indexed).
        total_epochs: Total number of epochs.

    Returns:
        Tuple[float, float]: (average_validation_loss, validation_accuracy_percentage).
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    pbar = tqdm(
        dataloader,
        desc=f"Epoch [{epoch:02d}/{total_epochs:02d}] Validation",
        leave=False,
        unit="batch",
    )

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        predictions = (logits >= 0.0).float()
        correct_predictions += (predictions == labels).sum().item()
        total_samples += batch_size

        batch_acc = (correct_predictions / total_samples) * 100.0
        pbar.set_postfix({"Val Loss": f"{loss.item():.4f}", "Val Acc": f"{batch_acc:.2f}%"})

    val_loss = running_loss / total_samples
    val_acc = (correct_predictions / total_samples) * 100.0
    return val_loss, val_acc


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Path,
) -> None:
    """Plot and save Loss and Accuracy learning curves.

    Args:
        history: Dictionary containing lists for train_loss, val_loss, train_acc, val_acc.
        save_path: Path where the generated figure will be saved.
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 1. Loss Curve
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2.0)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#d62728", linewidth=2.0, linestyle="--")
    min_val_loss_epoch = int(np.argmin(history["val_loss"])) + 1
    min_val_loss = float(np.min(history["val_loss"]))
    ax1.scatter([min_val_loss_epoch], [min_val_loss], color="#d62728", s=80, zorder=5)
    ax1.annotate(
        f"Min Val Loss: {min_val_loss:.4f}\n(Epoch {min_val_loss_epoch})",
        xy=(min_val_loss_epoch, min_val_loss),
        xytext=(min_val_loss_epoch + 2, min_val_loss + 0.05),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
        fontsize=9,
        fontweight="bold",
    )
    ax1.set_title("Binary Cross-Entropy Loss vs. Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # 2. Accuracy Curve
    ax2.plot(epochs, history["train_acc"], label="Train Accuracy", color="#2ca02c", linewidth=2.0)
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="#ff7f0e", linewidth=2.0, linestyle="--")
    max_val_acc_epoch = int(np.argmax(history["val_acc"])) + 1
    max_val_acc = float(np.max(history["val_acc"]))
    ax2.scatter([max_val_acc_epoch], [max_val_acc], color="#ff7f0e", s=80, zorder=5)
    ax2.annotate(
        f"Max Val Acc: {max_val_acc:.2f}%\n(Epoch {max_val_acc_epoch})",
        xy=(max_val_acc_epoch, max_val_acc),
        xytext=(max_val_acc_epoch - 12, max_val_acc - 5.0),
        arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.2),
        fontsize=9,
        fontweight="bold",
    )
    ax2.set_title("Classification Accuracy vs. Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_ylim(bottom=50.0, top=102.0)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True)

    plt.suptitle(
        "Inception-ResNet-V2 Smoker Detection: Training Dynamics (Section 6.1)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Curves] Learning curves saved to: {save_path}")


def main() -> None:
    """Master training entry point."""
    args = parse_args()

    # 1. Initialize environment & seeds
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    save_dir = Path(args.save_dir)
    output_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = save_dir / "best_model.pth"
    latest_model_path = save_dir / "latest_checkpoint.pth"
    history_json_path = output_dir / "training_history.json"
    curves_png_path = output_dir / "training_curves.png"

    print("=" * 75)
    print("  SMOKER DETECTION SYSTEM - TRAINING PIPELINE")
    print("=" * 75)
    use_fixed = args.steps_per_epoch is not None and args.steps_per_epoch > 0
    print(f"[Device]           : {str(device).upper()}")
    print(f"[Epochs]           : {args.epochs}")
    print(f"[Steps per Epoch]  : {args.steps_per_epoch if use_fixed else 'Full Dataset'}")
    print(f"[Batch Size]       : {args.batch_size}")
    print(f"[Learning Rate]    : {args.lr}")
    print(f"[Optimizer]        : SGD (momentum={args.momentum}, weight_decay={args.weight_decay})")
    print(f"[Model Save Dir]   : {save_dir}")
    print(f"[Outputs Dir]      : {output_dir}")
    print("=" * 75)

    # 2. Build DataLoaders
    # Apply user-overridden batch size and worker counts
    cfg = ProjectConfig()
    cfg.training.batch_size = args.batch_size
    cfg.training.steps_per_epoch = args.steps_per_epoch
    cfg.training.num_workers = args.num_workers
    cfg.training.learning_rate = args.lr
    cfg.training.momentum = args.momentum
    cfg.training.weight_decay = args.weight_decay
    cfg.training.epochs = args.epochs
    cfg.model.dropout_rate = args.dropout

    print("\n[1/4] Loading Datasets...")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    print(f"  Training samples   : {len(train_loader.dataset)} ({len(train_loader)} batches)")
    print(f"  Validation samples : {len(val_loader.dataset)} ({len(val_loader)} batches)")
    print(f"  Testing samples    : {len(test_loader.dataset)} ({len(test_loader)} batches)")

    # 3. Build Model
    print("\n[2/4] Initializing Pretrained Inception-ResNet-V2...")
    model = build_smoker_model(cfg)
    params = model.get_num_parameters()
    print(f"  Backbone           : Inception-ResNet-V2 (Frozen: {params['frozen']:,} params)")
    print(f"  Trainable Head     : 3-Layer Dense Head ({params['trainable']:,} params)")

    # 4. Criterion, Optimizer & Scheduler
    criterion = nn.BCEWithLogitsLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(
        trainable_params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler (reduces LR by 0.5 if val_loss plateaus for 5 epochs)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-5,
    )

    # Tracking metrics
    start_epoch = 1
    best_val_acc = 0.0
    best_val_loss = float("inf")
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    # Optional checkpoint resuming
    if args.resume and Path(args.resume).exists():
        print(f"\n[Resuming] Loading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        history = checkpoint.get("history", history)
        print(f"  Resumed at epoch {start_epoch} (Best Val Acc: {best_val_acc:.2f}%)")

    # 5. Training Loop
    print("\n[3/4] Starting Training Execution...")
    total_start_time = time.time()
    train_iter = None

    header = (
        f"{'Epoch':^7} | {'Train Loss':^11} | {'Train Acc':^11} | "
        f"{'Val Loss':^11} | {'Val Acc':^11} | {'LR':^9} | {'Time':^7} | {'Status'}"
    )
    print("-" * 85)
    print(header)
    print("-" * 85)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()

        # Training phase
        train_loss, train_acc, train_iter = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            data_iter=train_iter,
        )

        # Validation phase
        val_loss, val_acc = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        # Step learning rate scheduler based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        epoch_duration = time.time() - epoch_start_time

        # Check for model improvement
        # Primary criterion: higher validation accuracy; Secondary: lower validation loss
        is_best = (val_acc > best_val_acc) or (
            abs(val_acc - best_val_acc) < 1e-5 and val_loss < best_val_loss
        )
        status_tag = ""

        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            status_tag = "BEST"

            # Save best checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "config": cfg.__dict__,
                },
                best_model_path,
            )

        # Save latest checkpoint for resume capability
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
                "history": history,
            },
            latest_model_path,
        )

        # Display progress row
        row = (
            f"[{epoch:02d}/{args.epochs:02d}] | "
            f"{train_loss:^11.4f} | "
            f"{train_acc:^10.2f}% | "
            f"{val_loss:^11.4f} | "
            f"{val_acc:^10.2f}% | "
            f"{current_lr:^9.5f} | "
            f"{epoch_duration:^6.1f}s | "
            f"{status_tag}"
        )
        print(row)

    total_training_time = time.time() - total_start_time
    print("-" * 85)
    print(f"\n[4/4] Training Complete in {total_training_time / 60.0:.2f} minutes.")
    print(f"  Best Validation Accuracy : {best_val_acc:.2f}% (Loss: {best_val_loss:.4f})")
    print(f"  Best Model Checkpoint    : {best_model_path}")
    print(f"  Latest Checkpoint        : {latest_model_path}")

    # 6. Save history & generate learning curves
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
    print(f"  Training History JSON    : {history_json_path}")

    plot_training_history(history, curves_png_path)


if __name__ == "__main__":
    main()
