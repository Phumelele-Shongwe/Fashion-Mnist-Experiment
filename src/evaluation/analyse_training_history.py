
"""
Numerical analysis of Fashion-MNIST training histories.

Reports:
- Best validation accuracy and its epoch.
- Lowest validation loss and its epoch.
- Final training-validation accuracy gap.
- Final training-validation loss gap.
"""

from src.evaluation.plot_training_history import load_training_history


def analyse_model(model_name, metrics_path):

    history = load_training_history(metrics_path)

    best_accuracy_epoch = (
        max(
            range(len(history["validation_accuracy"])),
            key=lambda i: history["validation_accuracy"][i]
        )
    )

    best_loss_epoch = (
        min(
            range(len(history["validation_loss"])),
            key=lambda i: history["validation_loss"][i]
        )
    )

    final_epoch = len(history["train_loss"]) - 1

    train_accuracy = history["train_accuracy"][final_epoch]
    val_accuracy = history["validation_accuracy"][final_epoch]

    train_loss = history["train_loss"][final_epoch]
    val_loss = history["validation_loss"][final_epoch]

    print(f"\n{model_name.upper()} TRAINING ANALYSIS")
    print("=" * 55)

    print(
        f"Best validation accuracy: "
        f"{history['validation_accuracy'][best_accuracy_epoch]:.4f}"
    )

    print(f"Best accuracy epoch: {best_accuracy_epoch + 1}")

    print(
        f"Lowest validation loss: "
        f"{history['validation_loss'][best_loss_epoch]:.4f}"
    )

    print(f"Lowest loss epoch: {best_loss_epoch + 1}")

    print(f"\nFinal epoch: {final_epoch + 1}")

    print(f"Training accuracy:   {train_accuracy:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")

    print(
        f"Accuracy gap:        "
        f"{train_accuracy - val_accuracy:.4f}"
    )

    print(f"\nTraining loss:       {train_loss:.4f}")
    print(f"Validation loss:     {val_loss:.4f}")

    print(
        f"Loss gap:            "
        f"{val_loss - train_loss:.4f}"
    )


def main():

    analyse_model(
        "LeNet-5",
        "results/lenet5/metrics/training_history.json",
    )

    analyse_model(
        "VGG",
        "results/vgg/metrics/training_history.json",
    )


if __name__ == "__main__":
    main()