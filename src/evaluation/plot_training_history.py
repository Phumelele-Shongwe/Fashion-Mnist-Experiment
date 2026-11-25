
"""
Training history visualisation for Fashion-MNIST models.

Generates:
1. Training and validation loss curves.
2. Training and validation accuracy curves.

The same plotting functions are used for LeNet-5 and VGG.
"""

import json
import matplotlib.pyplot as plt

from pathlib import Path


def load_training_history(metrics_path):
    """
    Load training history from a JSON file.

    Supports:
        - History stored directly in the JSON.
        - History stored inside a top-level "history" key.
    """

    with open(metrics_path, "r", encoding="utf-8") as file:
        results = json.load(file)

    history = results.get("history", results)

    required_keys = [
        "train_loss",
        "validation_loss",
        "train_accuracy",
        "validation_accuracy",
    ]

    for key in required_keys:
        if key not in history:
            raise KeyError(
                f"Missing '{key}' in training history: {metrics_path}"
            )

    return history


def plot_loss_curve(model_name, history, output_dir):
    """
    Plot training loss against validation loss.
    """

    epochs = range(1, len(history["train_loss"]) + 1)

    # Find epoch with lowest validation loss
    best_epoch = min(
        range(len(history["validation_loss"])),
        key=lambda i: history["validation_loss"][i]
    ) + 1

    best_loss = min(history["validation_loss"])

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        epochs,
        history["train_loss"],
        marker="o",
        label="Training loss",
    )

    ax.plot(
        epochs,
        history["validation_loss"],
        marker="o",
        label="Validation loss",
    )

    ax.axvline(
        best_epoch,
        linestyle="--",
        alpha=0.7,
        label=f"Lowest validation loss (epoch {best_epoch})",
    )

    ax.set_title(f"{model_name} - Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")

    ax.set_xticks(list(epochs))
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output_path = output_dir / "loss_curve.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Loss curve saved: {output_path}")
    print(
        f"Lowest validation loss: {best_loss:.4f} "
        f"(epoch {best_epoch})"
    )


def plot_accuracy_curve(model_name, history, output_dir):
    """
    Plot training accuracy against validation accuracy.
    """

    epochs = range(1, len(history["train_accuracy"]) + 1)

    # Find epoch with highest validation accuracy
    best_epoch = max(
        range(len(history["validation_accuracy"])),
        key=lambda i: history["validation_accuracy"][i]
    ) + 1

    best_accuracy = max(history["validation_accuracy"])

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        epochs,
        history["train_accuracy"],
        marker="o",
        label="Training accuracy",
    )

    ax.plot(
        epochs,
        history["validation_accuracy"],
        marker="o",
        label="Validation accuracy",
    )

    ax.axvline(
        best_epoch,
        linestyle="--",
        alpha=0.7,
        label=f"Best validation accuracy (epoch {best_epoch})",
    )

    ax.set_title(f"{model_name} - Training and Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")

    ax.set_xticks(list(epochs))
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output_path = output_dir / "accuracy_curve.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Accuracy curve saved: {output_path}")
    print(
        f"Best validation accuracy: {best_accuracy:.4f} "
        f"(epoch {best_epoch})"
    )


def plot_model_history(model_name, metrics_path, output_dir):
    """
    Generate loss and accuracy plots for one model.
    """

    history = load_training_history(metrics_path)

    # Ensure all metrics contain the same number of epochs
    lengths = [
        len(history["train_loss"]),
        len(history["validation_loss"]),
        len(history["train_accuracy"]),
        len(history["validation_accuracy"]),
    ]

    if len(set(lengths)) != 1:
        raise ValueError(
            f"Inconsistent training history lengths for {model_name}: "
            f"{lengths}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{model_name.upper()} TRAINING HISTORY")
    print("-" * 50)

    plot_loss_curve(
        model_name,
        history,
        output_dir,
    )

    plot_accuracy_curve(
        model_name,
        history,
        output_dir,
    )


def main():

    # LeNet-5
    plot_model_history(
        model_name="LeNet-5",
        metrics_path="results/lenet5/metrics/training_history.json",
        output_dir="results/lenet5/plots",
    )

    # VGG
    plot_model_history(
        model_name="VGG",
        metrics_path="results/vgg/metrics/training_history.json",
        output_dir="results/vgg/plots",
    )

    print("\nTRAINING HISTORY VISUALISATION COMPLETED")


if __name__ == "__main__":
    main()