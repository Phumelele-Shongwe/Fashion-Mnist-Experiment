
"""
Evaluate the best LeNet-5 checkpoint on Fashion-MNIST.

Loads the best validation checkpoint, evaluates it on the
test dataset, and saves the resulting performance metrics.
"""

import json
import torch
import torch.nn as nn
import pandas as pd

from pathlib import Path

from configs.experiment_config import SEED
from src.architectures.lenet5 import LeNet5
from src.preprocessing.common import get_dataloaders
from src.evaluation.evaluator import evaluate_model


def main():

    # Reproducibility
    torch.manual_seed(SEED)

    # Device selection
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("LENET-5 MODEL EVALUATION")
    print("=" * 60)

    print(f"Device: {device}")

    # Load test data
    data = get_dataloaders()

    test_loader = data["test_loader"]
    class_names = data["test_dataset"].classes

    print(f"Test samples: {len(data['test_dataset'])}")

    # Initialize model
    model = LeNet5().to(device)

    # Load best checkpoint
    checkpoint_path = Path(
        "models/checkpoints/lenet5_best.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Checkpoint epoch: {checkpoint['epoch']}")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate model
    results = evaluate_model(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
    )

    # Prepare output directory
    output_dir = Path("results/lenet5/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save evaluation metrics
    metrics_path = output_dir / "test_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "model": "LeNet-5",
                "checkpoint_epoch": checkpoint["epoch"],
                "test_samples": len(data["test_dataset"]),
                **results,
            },
            file,
            indent=4,
        )

    # Save confusion matrix
    confusion_matrix = pd.DataFrame(
        results["confusion_matrix"],
        index=class_names,
        columns=class_names,
    )

    confusion_matrix.index.name = "Actual"
    confusion_matrix.columns.name = "Predicted"

    confusion_matrix_path = (
        output_dir / "confusion_matrix.csv"
    )

    confusion_matrix.to_csv(confusion_matrix_path)

    # Display results
    print("\nTEST PERFORMANCE")
    print("-" * 60)

    print(f"Test loss:          {results['loss']:.4f}")
    print(f"Accuracy:           {results['accuracy']:.4f}")
    print(f"Macro precision:    {results['macro_precision']:.4f}")
    print(f"Macro recall:       {results['macro_recall']:.4f}")
    print(f"Macro F1-score:     {results['macro_f1']:.4f}")

    print(f"\nWeighted precision: {results['weighted_precision']:.4f}")
    print(f"Weighted recall:    {results['weighted_recall']:.4f}")
    print(f"Weighted F1-score:  {results['weighted_f1']:.4f}")

    print("\nCLASSIFICATION REPORT")
    print("-" * 60)

    for class_name, metrics in results["classification_report"].items():

        if class_name in class_names:

            print(
                f"{class_name:20s} "
                f"Precision: {metrics['precision']:.4f}  "
                f"Recall: {metrics['recall']:.4f}  "
                f"F1: {metrics['f1-score']:.4f}"
            )

    print("\nGenerated files:")
    print(metrics_path)
    print(confusion_matrix_path)

    print("\nEVALUATION COMPLETED")


if __name__ == "__main__":
    main()