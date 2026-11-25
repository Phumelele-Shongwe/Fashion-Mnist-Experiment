"""
Shared training engine for Fashion-MNIST models.

This module provides reusable training, evaluation,
metric tracking, and checkpoint saving.
"""

import time
from pathlib import Path
import torch


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    """
    Train the model for one complete epoch.

    Returns:
        Dictionary containing average loss and accuracy.
    """

    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in data_loader:

        images = images.to(device)
        labels = labels.to(device)

        # 1. Clear gradients from the previous iteration
        optimizer.zero_grad(set_to_none=True)

        # 2. Forward pass
        outputs = model(images)

        # 3. Calculate loss
        loss = criterion(outputs, labels)

        # 4. Backpropagation
        loss.backward()

        # 5. Update model parameters
        optimizer.step()

        # 6. Accumulate metrics
        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        correct_predictions += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    return {
        "loss": epoch_loss,
        "accuracy": epoch_accuracy,
    }


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model without updating its parameters.

    Returns:
        Dictionary containing average loss and accuracy.
    """

    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Accumulate metrics
            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    return {
        "loss": epoch_loss,
        "accuracy": epoch_accuracy,
    }


def fit(
    model,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    epochs,
    device,
    checkpoint_path,
):
    """
    Execute the complete training process.

    Tracks training and validation metrics and saves
    the best checkpoint based on validation accuracy.

    Returns:
        Dictionary containing training history and
        best validation performance.
    """

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "epoch_duration": [],
    }

    best_validation_accuracy = -1.0
    best_validation_loss = float("inf")
    best_epoch = 0

    model.to(device)

    print(f"Training device: {device}")
    print(f"Number of epochs: {epochs}")
    print("-" * 60)

    for epoch in range(epochs):

        epoch_start = time.perf_counter()

        # Training phase
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        # Validation phase
        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        epoch_duration = time.perf_counter() - epoch_start

        # Record metrics
        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])

        history["validation_loss"].append(validation_metrics["loss"])
        history["validation_accuracy"].append(
            validation_metrics["accuracy"]
        )

        history["epoch_duration"].append(epoch_duration)

        # Determine whether this is the best checkpoint
        current_accuracy = validation_metrics["accuracy"]
        current_loss = validation_metrics["loss"]

        is_best = (
            current_accuracy > best_validation_accuracy
            or (
                current_accuracy == best_validation_accuracy
                and current_loss < best_validation_loss
            )
        )

        if is_best:

            best_validation_accuracy = current_accuracy
            best_validation_loss = current_loss
            best_epoch = epoch + 1

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_accuracy": best_validation_accuracy,
                    "validation_loss": best_validation_loss,
                },
                checkpoint_path,
            )

        # Display epoch results
        print(
            f"Epoch [{epoch + 1:02d}/{epochs:02d}] | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {validation_metrics['loss']:.4f} | "
            f"Val Acc: {validation_metrics['accuracy']:.4f} | "
            f"Time: {epoch_duration:.2f}s"
        )

    print("-" * 60)
    print("TRAINING COMPLETED")
    print(f"Best epoch: {best_epoch}")
    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy:.4f}"
    )
    print(
        f"Best validation loss: "
        f"{best_validation_loss:.4f}"
    )
    print(f"Checkpoint saved: {checkpoint_path}")

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "best_validation_loss": best_validation_loss,
        "checkpoint_path": str(checkpoint_path),
    }