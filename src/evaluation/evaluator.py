
"""
Shared evaluation engine for Fashion-MNIST models.

Evaluates trained models and calculates classification
metrics using the official test dataset.
"""

import torch

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def evaluate_model(
    model,
    data_loader,
    criterion,
    device,
    class_names,
):
    """
    Evaluate a trained model on a dataset.

    Returns:
        Dictionary containing:
            - loss
            - accuracy
            - macro precision, recall and F1
            - weighted precision, recall and F1
            - confusion matrix
            - per-class classification report
    """

    model.eval()

    running_loss = 0.0
    total_samples = 0

    all_predictions = []
    all_labels = []

    with torch.inference_mode():

        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Predictions
            predictions = outputs.argmax(dim=1)

            # Accumulate loss
            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size
            total_samples += batch_size

            # Store predictions and true labels
            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    # Average test loss
    test_loss = running_loss / total_samples

    # Overall accuracy
    test_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    # Macro-averaged metrics
    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        )
    )

    # Weighted-averaged metrics
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0,
        )
    )

    # Confusion matrix
    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
    )

    # Per-class classification report
    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "loss": test_loss,
        "accuracy": test_accuracy,

        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,

        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,

        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }