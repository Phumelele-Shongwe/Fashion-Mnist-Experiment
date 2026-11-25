
"""
Consolidated experiment summary for Fashion-MNIST.

Collects:
- Model architecture information.
- Training configuration and learning history.
- Best validation performance.
- Test performance.
- Per-class classification metrics.

Generates:
1. experiment_summary.json
2. model_comparison.csv
3. experiment_report.md
"""

import csv
import json

from pathlib import Path

from configs.experiment_config import (
    SEED,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    OPTIMIZER_NAME,
    LOSS_FUNCTION_NAME,
    VALIDATION_FRACTION,
)

from src.architectures.lenet5 import LeNet5
from src.architectures.vgg import VGG11FashionMNIST

from src.evaluation.plot_training_history import load_training_history


# --------------------------------------------------
# Experiment configuration
# --------------------------------------------------

MODELS = {
    "LeNet-5": {
        "model_class": LeNet5,
        "training_history": "results/lenet5/metrics/training_history.json",
        "test_metrics": "results/lenet5/metrics/test_metrics.json",
    },

    "VGG": {
        "model_class": VGG11FashionMNIST,
        "training_history": "results/vgg/metrics/training_history.json",
        "test_metrics": "results/vgg/metrics/test_metrics.json",
    },
}


# --------------------------------------------------
# Data loading and analysis
# --------------------------------------------------

def count_parameters(model):
    """Calculate the number of trainable parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def load_test_metrics(metrics_path):
    """Load saved test evaluation metrics."""

    with open(metrics_path, "r", encoding="utf-8") as file:
        return json.load(file)


def analyse_model(model_name, configuration):
    """Collect training and test results for one model."""

    # Load training history
    history = load_training_history(
        configuration["training_history"]
    )

    # Load test metrics
    test_results = load_test_metrics(
        configuration["test_metrics"]
    )

    # Instantiate architecture to count parameters
    model = configuration["model_class"]()

    parameters = count_parameters(model)

    # Best validation accuracy
    best_accuracy_index = max(
        range(len(history["validation_accuracy"])),
        key=lambda i: history["validation_accuracy"][i]
    )

    # Lowest validation loss
    best_loss_index = min(
        range(len(history["validation_loss"])),
        key=lambda i: history["validation_loss"][i]
    )

    final_index = len(history["train_loss"]) - 1

    # Training duration, if recorded
    epoch_durations = history.get("epoch_duration", [])

    total_training_time = (
        sum(epoch_durations)
        if epoch_durations
        else None
    )

    return {
        "model": model_name,

        "architecture": {
            "trainable_parameters": parameters,
        },

        "training": {
            "epochs_completed": len(history["train_loss"]),
            "best_validation_accuracy": history["validation_accuracy"][best_accuracy_index],
            "best_accuracy_epoch": best_accuracy_index + 1,
            "lowest_validation_loss": history["validation_loss"][best_loss_index],
            "lowest_loss_epoch": best_loss_index + 1,
            "final_training_accuracy": history["train_accuracy"][final_index],
            "final_validation_accuracy": history["validation_accuracy"][final_index],
            "final_training_loss": history["train_loss"][final_index],
            "final_validation_loss": history["validation_loss"][final_index],
            "final_accuracy_gap": (
                history["train_accuracy"][final_index]
                - history["validation_accuracy"][final_index]
            ),
            "final_loss_gap": (
                history["validation_loss"][final_index]
                - history["train_loss"][final_index]
            ),
            "total_training_time_seconds": total_training_time,
        },

        "test": test_results,
    }


# --------------------------------------------------
# Output generation
# --------------------------------------------------

def save_json(summary, output_dir):

    output_path = output_dir / "experiment_summary.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    return output_path


def save_comparison_csv(summary, output_dir):

    output_path = output_dir / "model_comparison.csv"

    rows = []

    for model_name, results in summary["models"].items():

        training = results["training"]
        test = results["test"]
        architecture = results["architecture"]

        rows.append({
            "Model": model_name,
            "Trainable Parameters": architecture["trainable_parameters"],
            "Best Validation Accuracy": training["best_validation_accuracy"],
            "Best Validation Epoch": training["best_accuracy_epoch"],
            "Test Accuracy": test["accuracy"],
            "Test Loss": test["loss"],
            "Macro Precision": test["macro_precision"],
            "Macro Recall": test["macro_recall"],
            "Macro F1": test["macro_f1"],
            "Weighted F1": test["weighted_f1"],
            "Final Accuracy Gap": training["final_accuracy_gap"],
            "Final Loss Gap": training["final_loss_gap"],
        })

    with open(output_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    return output_path


def save_markdown_report(summary, output_dir):

    output_path = output_dir / "experiment_report.md"

    models = summary["models"]

    lines = [
        "# Fashion-MNIST CNN Experiment",
        "",
        "## 1. Experiment configuration",
        "",
        f"- Random seed: {summary['configuration']['seed']}",
        f"- Epochs: {summary['configuration']['epochs']}",
        f"- Batch size: {summary['configuration']['batch_size']}",
        f"- Learning rate: {summary['configuration']['learning_rate']}",
        f"- Optimizer: {summary['configuration']['optimizer']}",
        f"- Loss function: {summary['configuration']['loss_function']}",
        f"- Validation fraction: {summary['configuration']['validation_fraction']:.0%}",
        "",
        "## 2. Model comparison",
        "",
        "| Metric | LeNet-5 | VGG |",
        "|---|---:|---:|",
    ]

    comparison_metrics = [
        ("Trainable parameters", "architecture", "trainable_parameters"),
        ("Best validation accuracy", "training", "best_validation_accuracy"),
        ("Best validation epoch", "training", "best_accuracy_epoch"),
        ("Test accuracy", "test", "accuracy"),
        ("Test loss", "test", "loss"),
        ("Macro precision", "test", "macro_precision"),
        ("Macro recall", "test", "macro_recall"),
        ("Macro F1", "test", "macro_f1"),
        ("Final accuracy gap", "training", "final_accuracy_gap"),
        ("Final loss gap", "training", "final_loss_gap"),
    ]

    for label, section, key in comparison_metrics:

        lenet_value = models["LeNet-5"][section][key]
        vgg_value = models["VGG"][section][key]

        if "accuracy" in key or "precision" in key or "recall" in key or "f1" in key:
            lenet_value = f"{lenet_value:.4f}"
            vgg_value = f"{vgg_value:.4f}"

        elif "loss" in key:
            lenet_value = f"{lenet_value:.4f}"
            vgg_value = f"{vgg_value:.4f}"

        elif "parameters" in key:
            lenet_value = f"{lenet_value:,}"
            vgg_value = f"{vgg_value:,}"

        lines.append(
            f"| {label} | {lenet_value} | {vgg_value} |"
        )

    # Per-class F1 comparison
    lines.extend([
        "",
        "## 3. Per-class performance",
        "",
        "| Class | LeNet-5 F1 | VGG F1 |",
        "|---|---:|---:|",
    ])

    lenet_report = models["LeNet-5"]["test"]["classification_report"]
    vgg_report = models["VGG"]["test"]["classification_report"]

    class_names = [
        name for name in lenet_report
        if name in vgg_report
        and isinstance(lenet_report[name], dict)
        and "f1-score" in lenet_report[name]
    ]

    for class_name in class_names:

        lenet_f1 = lenet_report[class_name]["f1-score"]
        vgg_f1 = vgg_report[class_name]["f1-score"]

        lines.append(
            f"| {class_name} | {lenet_f1:.4f} | {vgg_f1:.4f} |"
        )

    lines.extend([
        "",
        "## 4. Experimental observations",
        "",
        "- Both models were trained using the same dataset preparation and training configuration.",
        "- VGG achieved higher overall test accuracy in this experiment.",
        "- LeNet-5 has substantially fewer trainable parameters.",
        "- VGG showed a larger final training-validation accuracy gap.",
        "- The architectures differ in several design choices; this experiment does not isolate network depth as the sole cause of the performance difference.",
        "",
        "## 5. Methodological note",
        "",
        "The official Fashion-MNIST test set had been explored in an earlier experiment. "
        "Consequently, it should not be described as a completely untouched final holdout. "
        "The reported test metrics are presented as an evaluation and were not used to tune this experiment.",
        "",
    ])

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    return output_path


# --------------------------------------------------
# Main execution
# --------------------------------------------------

def main():

    print("=" * 60)
    print("FASHION-MNIST EXPERIMENT SUMMARY")
    print("=" * 60)

    # Shared configuration
    experiment_configuration = {
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": OPTIMIZER_NAME,
        "loss_function": LOSS_FUNCTION_NAME,
        "validation_fraction": VALIDATION_FRACTION,
    }

    # Collect model results
    model_results = {}

    for model_name, configuration in MODELS.items():

        print(f"\nCollecting results: {model_name}")

        model_results[model_name] = analyse_model(
            model_name,
            configuration,
        )

    summary = {
        "experiment": "Fashion-MNIST CNN Comparison",
        "configuration": experiment_configuration,
        "models": model_results,
    }

    # Output directory
    output_dir = Path("results/experiment_summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output files
    json_path = save_json(summary, output_dir)
    csv_path = save_comparison_csv(summary, output_dir)
    report_path = save_markdown_report(summary, output_dir)

    # Display compact comparison
    print("\nMODEL PERFORMANCE")
    print("-" * 60)

    for model_name, results in model_results.items():

        print(f"\n{model_name}")
        print(
            f"Parameters:       "
            f"{results['architecture']['trainable_parameters']:,}"
        )
        print(
            f"Validation acc:   "
            f"{results['training']['best_validation_accuracy']:.4f}"
        )
        print(
            f"Test accuracy:    "
            f"{results['test']['accuracy']:.4f}"
        )
        print(
            f"Test macro F1:    "
            f"{results['test']['macro_f1']:.4f}"
        )

    print("\nGENERATED FILES")
    print("-" * 60)
    print(json_path)
    print(csv_path)
    print(report_path)

    print("\nEXPERIMENT SUMMARY COMPLETED")


if __name__ == "__main__":
    main()