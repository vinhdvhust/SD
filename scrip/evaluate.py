"""Evaluation and Metric Benchmarking Pipeline for Smoker Detection.

This script implements Phase 4 of the project plan adhering to Section 6.2 (Equations 4-13)
and Section 6.4 (Table 3, Figures 6, 7, 8) of:
    Khan, A., Khan, S., Hassan, B., & Zheng, Z. (2022).
    CNN-Based Smoker Classification and Detection in Smart City Application.
    Sensors, 22(3), 892. https://doi.org/10.3390/s22030892

Evaluates:
    1. Prediction Accuracy (Equation 4)
    2. Precision / Positive Predictive Value (Equation 5)
    3. Recall / Sensitivity (Equation 6)
    4. Specificity / True Negative Rate (Equation 7)
    5. Negative Predictive Value - PVn (Equation 8)
    6. False Positive Rate - FPR (Equation 9)
    7. False Negative Rate - FNR (Equation 10)
    8. False Discovery Rate - FDR (Equation 11)
    9. F1-Score (Equation 12)
    10. Error Rate - Er (Equation 13)
    11. Area Under ROC Curve (AUC) & Average Precision (AP)

Exports:
    - Confusion Matrix heatmap (Figure 6 replica)
    - ROC Curve with zoomed sub-panel (Figure 7 replica)
    - Precision-Recall Curve with scaled sub-panel (Figure 8 replica)
    - Metrics comparison report against paper benchmark (Markdown & JSON)
    - Detailed list of misclassified images
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from scrip.config import ProjectConfig, config as default_config
    from scrip.dataset import SmokerDataset, build_transforms
    from scrip.model import SmokerInceptionResNetV2, build_smoker_model
except ImportError:
    from config import ProjectConfig, config as default_config
    from dataset import SmokerDataset, build_transforms
    from model import SmokerInceptionResNetV2, build_smoker_model


def parse_args() -> argparse.Namespace:
    """Parse evaluation command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Inception-ResNet-V2 Smoker Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Default model path checks best_model_50epochs.pth first, then best_model.pth
    default_ckpt = default_config.model_save_dir / "best_model_50epochs.pth"
    if not default_ckpt.exists():
        default_ckpt = default_config.model_save_dir / "best_model.pth"

    parser.add_argument(
        "--model-path",
        type=str,
        default=str(default_ckpt),
        help="Path to trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_config.dataset.project_root / "eval" / "50epochs"),
        help="Directory where evaluation charts and reports will be saved",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=default_config.training.batch_size,
        help="Batch size for test evaluation DataLoader",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_config.training.device,
        help="Compute device ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision probability threshold for classifying as Smoking (1)",
    )
    return parser.parse_args()


@torch.no_grad()
def run_test_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Execute forward inference across the testing dataset.

    Args:
        model: Evaluated PyTorch neural network.
        dataloader: Testing DataLoader returning (images, labels, filenames).
        device: Target compute device.
        threshold: Probability cutoff for binary decision.

    Returns:
        Tuple:
            - y_true: Ground truth binary labels (0 or 1).
            - y_prob: Predicted sigmoid probabilities in [0, 1].
            - y_pred: Binary predicted labels (0 or 1).
            - filenames: List of evaluated image file names.
    """
    model.eval()
    all_true: List[int] = []
    all_prob: List[float] = []
    all_pred: List[int] = []
    all_files: List[str] = []

    for images, labels, names in tqdm(dataloader, desc="Evaluating Test Set", unit="batch"):
        images = images.to(device, non_blocking=True)
        probs = model.predict_proba(images).squeeze(-1).cpu().numpy()
        targets = labels.squeeze(-1).numpy().astype(int)

        for prob, target, name in zip(probs, targets, names):
            p_val = float(prob)
            all_true.append(int(target))
            all_prob.append(p_val)
            all_pred.append(1 if p_val >= threshold else 0)
            all_files.append(name)

    return (
        np.array(all_true, dtype=int),
        np.array(all_prob, dtype=float),
        np.array(all_pred, dtype=int),
        all_files,
    )


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """Compute all 11 scientific metrics according to Section 6.2 (Equations 4-13).

    Definitions:
        - Tp: True Positives (Smoking correctly classified as Smoking)
        - Tn: True Negatives (NotSmoking correctly classified as NotSmoking)
        - Fp: False Positives (NotSmoking incorrectly classified as Smoking)
        - Fn: False Negatives (Smoking incorrectly classified as NotSmoking)

    Equations:
        - Accuracy    = (Tp + Tn) / (Tp + Tn + Fp + Fn)         (Eq. 4)
        - Precision   = Tp / (Tp + Fp)                          (Eq. 5)
        - Recall      = Tp / (Tp + Fn)                          (Eq. 6)
        - Specificity = Tn / (Tn + Fp)                          (Eq. 7)
        - PVn         = Tn / (Tn + Fn)                          (Eq. 8)
        - FPR         = 1 - Specificity = Fp / (Tn + Fp)        (Eq. 9)
        - FNR         = 1 - Sensitivity = Fn / (Tp + Fn)        (Eq. 10)
        - FDR         = 1 - Precision   = Fp / (Tp + Fp)        (Eq. 11)
        - F1 score    = 2 * (Precision * Recall) / (P + R)      (Eq. 12)
        - Er          = (Fp + Fn) / (Tp + Tn + Fp + Fn)         (Eq. 13)
        - AUC         = Area under ROC curve
        - AP          = Average Precision (area under PR curve)
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm layout:
    # row 0 (actual NotSmoking): Tn = cm[0, 0], Fp = cm[0, 1]
    # row 1 (actual Smoking):    Fn = cm[1, 0], Tp = cm[1, 1]
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    pvn = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    fpr = 1.0 - specificity
    fnr = 1.0 - recall
    fdr = 1.0 - precision
    error_rate = (fp + fn) / total if total > 0 else 0.0

    f1 = (
        2.0 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    auc = float(roc_auc_score(y_true, y_prob))
    ap = float(average_precision_score(y_true, y_prob))

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "pvn": pvn,
        "fpr": fpr,
        "fnr": fnr,
        "fdr": fdr,
        "f1_score": f1,
        "error_rate": error_rate,
        "auc": auc,
        "ap": ap,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def plot_confusion_matrix_figure(
    cm: np.ndarray,
    save_path: Path,
) -> None:
    """Generate and save publication-quality Confusion Matrix matching Figure 6.

    Figure 6 details:
        - X-axis: Predicted (NotSmoking, Smoking)
        - Y-axis: Actual (NotSmoking, Smoking)
        - Blue-toned colormap with count labels in each cell
        - Border and colorbar on the right
    """
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])

    matrix_data = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix_data, interpolation="nearest", cmap=plt.cm.Blues)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    classes = ["NotSmoking", "Smoking"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=11)

    # Axis labels matching Figure 6
    ax.set_xlabel("Predicted", fontsize=12, labelpad=8)
    ax.set_ylabel("Actual", fontsize=12, labelpad=8)
    ax.set_title("Figure 6: Confusion Matrix", fontsize=13, fontweight="bold", pad=12)

    # Cell numbers annotations
    thresh = matrix_data.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = matrix_data[i, j]
            color = "white" if val > thresh else "black"
            ax.text(
                j,
                i,
                f"{val}",
                horizontalalignment="center",
                verticalalignment="center",
                color=color,
                fontsize=16,
                fontweight="bold",
            )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"[Chart] Confusion Matrix saved to: {save_path}")


def plot_roc_curve_figure(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    auc_score: float,
    save_path: Path,
) -> None:
    """Generate and save publication-quality ROC Curve matching Figure 7.

    Figure 7 details:
        - Subplot (a): Full ROC curve (FPR vs TPR) with AUC in legend
        - Subplot (b): Zoomed ROC curve at top left (FPR in [0.0, 0.2], TPR in [0.8, 1.0])
        - Golden/Orange curve line styling
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Subplot (a): Full ROC Curve ---
    ax1.plot(
        fpr,
        tpr,
        color="#f39c12",
        linewidth=2.5,
        label=f"ROC curve (AUC = {auc_score:.4f})",
    )
    ax1.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1.0, alpha=0.5)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.05])
    ax1.set_xlabel("False positive rate", fontsize=11)
    ax1.set_ylabel("True positive rate", fontsize=11)
    ax1.set_title("(a) Receiver Operating Characteristic curve", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=10, frameon=True)

    # --- Subplot (b): Zoomed ROC Curve at top left ---
    ax2.plot(
        fpr,
        tpr,
        color="#f39c12",
        linewidth=2.5,
        label=f"ROC curve (AUC = {auc_score:.4f})",
    )
    ax2.set_xlim([0.0, 0.20])
    ax2.set_ylim([0.80, 1.005])
    ax2.set_xlabel("False positive rate", fontsize=11)
    ax2.set_ylabel("True positive rate", fontsize=11)
    ax2.set_title("(b) ROC curve (zoomed in at top left)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=10, frameon=True)

    fig.suptitle(
        f"Figure 7: Receiver Operating Characteristic (ROC) Analysis (AUC = {auc_score:.4f})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"[Chart] ROC Curve saved to: {save_path}")


def plot_pr_curve_figure(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ap_score: float,
    save_path: Path,
) -> None:
    """Generate and save publication-quality Precision-Recall Curve matching Figure 8.

    Figure 8 details:
        - Subplot (a): Full Precision-Recall curve with AP in legend
        - Subplot (b): Scaled presentation (Recall in [0.8, 1.0], Precision in [0.4, 1.0])
        - Golden/Orange curve line styling
    """
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Subplot (a): Standard PR Curve ---
    ax1.plot(
        recalls,
        precisions,
        color="#f39c12",
        linewidth=2.5,
        label=f"Precision-Recall curve (AP = {ap_score:.4f})",
    )
    ax1.set_xlim([0.48, 1.02])
    ax1.set_ylim([-0.02, 1.05])
    ax1.set_xlabel("Recall", fontsize=11)
    ax1.set_ylabel("Precision", fontsize=11)
    ax1.set_title("(a) Precision–Recall curve", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=10, frameon=True)

    # --- Subplot (b): Scaled PR Curve ---
    ax2.plot(
        recalls,
        precisions,
        color="#f39c12",
        linewidth=2.5,
        label=f"Precision-Recall curve (AP = {ap_score:.4f})",
    )
    ax2.set_xlim([0.80, 1.005])
    ax2.set_ylim([0.40, 1.01])
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title("(b) PR curve (scaled presentation)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left", fontsize=10, frameon=True)

    fig.suptitle(
        f"Figure 8: Precision–Recall (PR) Analysis (AP = {ap_score:.4f})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"[Chart] Precision-Recall Curve saved to: {save_path}")


def save_reports(
    metrics: Dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    filenames: List[str],
    output_dir: Path,
    model_name: str,
) -> None:
    """Save quantitative comparison Markdown report and JSON archives.

    Compares against published results from:
        Khan et al. (2022) Table 3: Performance Analysis
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Benchmark metrics published in Table 3 of the paper
    paper_benchmark = {
        "tp": 109,
        "tn": 108,
        "fp": 4,
        "fn": 3,
        "precision": 0.9732,
        "recall": 0.9646,
        "pvn": 0.9729,
        "specificity": 0.9643,
        "fpr": 0.0357,
        "fnr": 0.0354,
        "fdr": 0.0268,
        "f1_score": 0.9688,
        "accuracy": 0.9687,
        "error_rate": 0.0312,
        "auc": 0.9859,
        "ap": 0.9848,
    }

    # Extract misclassified instances
    false_positives: List[Dict[str, Any]] = []
    false_negatives: List[Dict[str, Any]] = []

    for name, true_lbl, pred_lbl, prob in zip(filenames, y_true, y_pred, y_prob):
        if true_lbl == 0 and pred_lbl == 1:
            false_positives.append({"filename": name, "prob_smoking": round(float(prob), 4)})
        elif true_lbl == 1 and pred_lbl == 0:
            false_negatives.append({"filename": name, "prob_smoking": round(float(prob), 4)})

    # Sort by prediction confidence
    false_positives.sort(key=lambda x: -x["prob_smoking"])
    false_negatives.sort(key=lambda x: x["prob_smoking"])

    # 1. JSON Report
    json_path = output_dir / "evaluation_metrics.json"
    full_data = {
        "model_checkpoint": model_name,
        "metrics_50epochs": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
        "paper_benchmark": paper_benchmark,
        "misclassifications": {
            "false_positives_count": len(false_positives),
            "false_negatives_count": len(false_negatives),
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_data, f, indent=4)
    print(f"[Report] JSON metrics archive saved to: {json_path}")

    # 2. Markdown Report
    md_path = output_dir / "metrics_comparison.md"
    diff_acc = (metrics["accuracy"] - paper_benchmark["accuracy"]) * 100.0

    lines = [
        f"# BÁO CÁO ĐÁNH GIÁ CHI TIẾT 11 CHỈ SỐ: MÔ HÌNH 50 EPOCHS",
        f"",
        f"> **Tệp trọng số**: `{model_name}`  ",
        f"> **Tập kiểm thử**: 224 ảnh (`SD/data/Testing`: 112 Smoking, 112 NotSmoking)  ",
        f"> **Ngưỡng quyết định**: $\\tau = 0.5$  ",
        f"",
        f"---",
        f"",
        f"## 1. Bảng đối chiếu 11 chỉ số khoa học so với bài báo gốc (Table 3)",
        f"",
        f"| Chỉ số (Metric) | Công thức (Section 6.2) | Kết quả 50 Epochs | Bài báo công bố (Inception-ResNet-V2) | Chênh lệch |",
        f"| :--- | :---: | :---: | :---: | :---: |",
        f"| **True Positives ($T_p$)** | $T_p$ | **{metrics['tp']}** | 109 | {metrics['tp'] - 109:+d} |",
        f"| **True Negatives ($T_n$)** | $T_n$ | **{metrics['tn']}** | 108 | {metrics['tn'] - 108:+d} |",
        f"| **False Positives ($F_p$)** | $F_p$ | **{metrics['fp']}** | 4 | {metrics['fp'] - 4:+d} |",
        f"| **False Negatives ($F_n$)** | $F_n$ | **{metrics['fn']}** | 3 | {metrics['fn'] - 3:+d} |",
        f"| **Độ chính xác (Accuracy)** | $\\frac{{T_p + T_n}}{{\\text{{Total}}}}$ (Eq. 4) | **{metrics['accuracy']*100:.2f}%** | 96.87% | {diff_acc:+.2f}% |",
        f"| **Độ chuẩn xác (Precision - $PV_p$)** | $\\frac{{T_p}}{{T_p + F_p}}$ (Eq. 5) | **{metrics['precision']*100:.2f}%** | 97.32% | {(metrics['precision']-paper_benchmark['precision'])*100:+.2f}% |",
        f"| **Độ nhạy (Recall / Sensitivity)** | $\\frac{{T_p}}{{T_p + F_n}}$ (Eq. 6) | **{metrics['recall']*100:.2f}%** | 96.46% | {(metrics['recall']-paper_benchmark['recall'])*100:+.2f}% |",
        f"| **Độ đặc hiệu (Specificity)** | $\\frac{{T_n}}{{T_n + F_p}}$ (Eq. 7) | **{metrics['specificity']*100:.2f}%** | 96.43% | {(metrics['specificity']-paper_benchmark['specificity'])*100:+.2f}% |",
        f"| **Negative Pred Value ($PV_n$)** | $\\frac{{T_n}}{{T_n + F_n}}$ (Eq. 8) | **{metrics['pvn']*100:.2f}%** | 97.29% | {(metrics['pvn']-paper_benchmark['pvn'])*100:+.2f}% |",
        f"| **Tỷ lệ báo động giả ($FPR$)** | $1 - \\text{{Specificity}}$ (Eq. 9) | **{metrics['fpr']*100:.2f}%** | 3.57% | {(metrics['fpr']-paper_benchmark['fpr'])*100:+.2f}% |",
        f"| **Tỷ lệ bỏ sót ($FNR$)** | $1 - \\text{{Sensitivity}}$ (Eq. 10) | **{metrics['fnr']*100:.2f}%** | 3.54% | {(metrics['fnr']-paper_benchmark['fnr'])*100:+.2f}% |",
        f"| **Tỷ lệ phát hiện sai ($FDR$)** | $1 - PV_p$ (Eq. 11) | **{metrics['fdr']*100:.2f}%** | 2.68% | {(metrics['fdr']-paper_benchmark['fdr'])*100:+.2f}% |",
        f"| **Điểm $F_1$ ($F_1\\text{{-Score}}$)** | $2 \\times \\frac{{P \\times R}}{{P + R}}$ (Eq. 12) | **{metrics['f1_score']:.4f}** | 0.9688 | {metrics['f1_score']-paper_benchmark['f1_score']:+.4f} |",
        f"| **Tỷ lệ lỗi ($E_r$)** | $\\frac{{F_p + F_n}}{{\\text{{Total}}}}$ (Eq. 13) | **{metrics['error_rate']*100:.2f}%** | 3.12% | {(metrics['error_rate']-paper_benchmark['error_rate'])*100:+.2f}% |",
        f"| **Diện tích dưới ROC ($AUC$)** | Area under ROC | **{metrics['auc']:.4f}** | 0.9859 | {metrics['auc']-paper_benchmark['auc']:+.4f} |",
        f"| **Độ chính xác TB ($AP$)** | Area under PR | **{metrics['ap']:.4f}** | 0.9848 | {metrics['ap']-paper_benchmark['ap']:+.4f} |",
        f"",
        f"---",
        f"",
        f"## 2. Các biểu đồ trực quan hóa khoa học",
        f"",
        f"- **Confusion Matrix**: `confusion_matrix.png`  ",
        f"- **Đường cong ROC (Full & Zoomed)**: `roc_curve.png` (AUC = {metrics['auc']:.4f})  ",
        f"- **Đường cong Precision-Recall (Full & Scaled)**: `precision_recall_curve.png` (AP = {metrics['ap']:.4f})  ",
        f"",
        f"---",
        f"",
        f"## 3. Danh sách các mẫu phân loại sai (Error Analysis)",
        f"",
        f"### A. False Positives ({len(false_positives)} ảnh - Thực tế: NotSmoking, Đoán nhầm: Smoking)",
        f"| STT | Tên tệp ảnh | Xác suất Smoking |",
        f"| :---: | :--- | :---: |",
    ]

    for idx, item in enumerate(false_positives, 1):
        lines.append(f"| {idx} | `{item['filename']}` | {item['prob_smoking']:.4f} |")

    lines.extend([
        f"",
        f"### B. False Negatives ({len(false_negatives)} ảnh - Thực tế: Smoking, Bỏ sót thành: NotSmoking)",
        f"| STT | Tên tệp ảnh | Xác suất Smoking |",
        f"| :---: | :--- | :---: |",
    ])

    for idx, item in enumerate(false_negatives, 1):
        lines.append(f"| {idx} | `{item['filename']}` | {item['prob_smoking']:.4f} |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Report] Markdown comparison report saved to: {md_path}")


def main() -> None:
    """Master evaluation entry point."""
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {model_path}")

    print("=" * 75)
    print("  SMOKER DETECTION SYSTEM - EVALUATION PIPELINE")
    print("=" * 75)
    print(f"[Device]           : {str(device).upper()}")
    print(f"[Checkpoint]       : {model_path.name}")
    print(f"[Output Directory] : {output_dir}")
    print(f"[Threshold]        : {args.threshold}")
    print("=" * 75)

    # 1. Load Dataset
    print("\n[1/3] Loading Testing Dataset (224 images)...")
    eval_transform = build_transforms(is_train=False)
    test_dataset = SmokerDataset(
        data_dir=default_config.dataset.test_dir,
        transform=eval_transform,
        return_metadata=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=default_config.training.num_workers,
        pin_memory=default_config.training.pin_memory,
    )
    print(f"  Test samples loaded: {len(test_dataset)} ({len(test_loader)} batches)")

    # 2. Load Model
    print("\n[2/3] Initializing Model & Loading Weights...")
    model = build_smoker_model(default_config)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Run Inference
    print("\n[3/3] Running Inference & Calculating 11 Metrics...")
    y_true, y_prob, y_pred, filenames = run_test_inference(
        model=model,
        dataloader=test_loader,
        device=device,
        threshold=args.threshold,
    )

    # 4. Compute Metrics
    metrics = compute_all_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob)

    # Display console summary
    print("\n" + "=" * 75)
    print("  EVALUATION RESULTS - 11 QUANTITATIVE METRICS (SECTION 6.2)")
    print("=" * 75)
    print(f"  Accuracy (Acc)              : {metrics['accuracy'] * 100:.2f}% ({metrics['tp'] + metrics['tn']}/{metrics['total']} correct)")
    print(f"  Precision (PVp)             : {metrics['precision'] * 100:.2f}%")
    print(f"  Recall / Sensitivity        : {metrics['recall'] * 100:.2f}%")
    print(f"  Specificity                 : {metrics['specificity'] * 100:.2f}%")
    print(f"  Negative Pred Value (PVn)   : {metrics['pvn'] * 100:.2f}%")
    print(f"  False Positive Rate (FPR)   : {metrics['fpr'] * 100:.2f}% (Fp = {metrics['fp']})")
    print(f"  False Negative Rate (FNR)   : {metrics['fnr'] * 100:.2f}% (Fn = {metrics['fn']})")
    print(f"  False Discovery Rate (FDR)  : {metrics['fdr'] * 100:.2f}%")
    print(f"  F1-Score                    : {metrics['f1_score']:.4f}")
    print(f"  Error Rate (Er)             : {metrics['error_rate'] * 100:.2f}%")
    print(f"  ROC Area Under Curve (AUC)  : {metrics['auc']:.4f}")
    print(f"  Average Precision (AP)      : {metrics['ap']:.4f}")
    print("=" * 75 + "\n")

    # 5. Generate and Export Scientific Visualizations
    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix_figure(cm, output_dir / "confusion_matrix.png")
    plot_roc_curve_figure(y_true, y_prob, metrics["auc"], output_dir / "roc_curve.png")
    plot_pr_curve_figure(y_true, y_prob, metrics["ap"], output_dir / "precision_recall_curve.png")

    # 6. Save Reports
    save_reports(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        filenames=filenames,
        output_dir=output_dir,
        model_name=model_path.name,
    )


if __name__ == "__main__":
    main()
