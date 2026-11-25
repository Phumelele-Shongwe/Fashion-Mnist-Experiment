
"""
Confusion matrix visualisation for Fashion-MNIST models.

Generates:
1. Raw confusion matrix (prediction counts).
2. Normalised confusion matrix (row-wise percentages).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import confusion_matrix


def plot_confusion_matrices(
    model_name,
    metrics_path,
    output_dir,
    class_names,
):
    """
    Generate raw and normalised confusion matrix plots.

    Args:
        model_name: Name of the evaluated model.
        metrics_path: Path to the saved test metrics JSON.
        output_dir: Directory for saving visualisations.
        class_names: Fashion-MNIST class labels.
    """

    import json

    # Load saved evaluation results
    with open(metrics_path, "r", encoding="utf-8") as file:
        results = json.load(file)

    matrix = np.array(results["confusion_matrix"])

    # Row-wise normalisation
    row_totals = matrix.sum(axis=1, keepdims=True)

    normalised_matrix = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    normalised_matrix *= 100

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. Raw confusion matrix
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 9))

    image = ax.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
    )

    fig.colorbar(image, ax=ax, label="Number of predictions")

    ax.set(
        title=f"{model_name} - Confusion Matrix",
        xlabel="Predicted class",
        ylabel="Actual class",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
    )

    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = matrix.max() / 2

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):

            ax.text(
                j,
                i,
                f"{matrix[i, j]}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()

    raw_path = output_dir / "confusion_matrix_raw.png"

    fig.savefig(
        raw_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------
    # 2. Normalised confusion matrix
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 9))

    image = ax.imshow(
        normalised_matrix,
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=100,
    )

    fig.colorbar(image, ax=ax, label="Percentage (%)")

    ax.set(
        title=f"{model_name} - Normalised Confusion Matrix",
        xlabel="Predicted class",
        ylabel="Actual class",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
    )

    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(normalised_matrix.shape[0]):
        for j in range(normalised_matrix.shape[1]):

            value = normalised_matrix[i, j]

            ax.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                color="white" if value > 50 else "black",
                fontsize=8,
            )

    fig.tight_layout()

    normalised_path = output_dir / "confusion_matrix_normalised.png"

    fig.savefig(
        normalised_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\n{model_name} CONFUSION MATRICES GENERATED")
    print("-" * 50)
    print(raw_path)
    print(normalised_path)


def main():

    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    # LeNet-5
    plot_confusion_matrices(
        model_name="LeNet-5",
        metrics_path="results/lenet5/metrics/test_metrics.json",
        output_dir="results/lenet5/plots",
        class_names=class_names,
    )

    # VGG
    plot_confusion_matrices(
        model_name="VGG",
        metrics_path="results/vgg/metrics/test_metrics.json",
        output_dir="results/vgg/plots",
        class_names=class_names,
    )


if __name__ == "__main__":
    main()