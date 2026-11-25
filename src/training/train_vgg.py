
"""
VGG Fashion-MNIST training script.

Configures the VGG architecture and delegates
the actual training process to the shared engine.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from configs.experiment_config import (
    SEED,
    EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    OPTIMIZER_NAME,
    LOSS_FUNCTION_NAME,
)

from src.architectures.vgg import VGG11FashionMNIST
from src.preprocessing.common import get_dataloaders
from src.training.trainer import fit


def main():

    # 1. Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # 2. Select computation device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 3. Load datasets
    data = get_dataloaders(batch_size=BATCH_SIZE)

    train_loader = data["train_loader"]
    validation_loader = data["validation_loader"]

    # 4. Initialize VGG
    model = VGG11FashionMNIST()

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    # 5. Configure loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # 6. Define output paths
    checkpoint_path = (
        Path("models")
        / "checkpoints"
        / "vgg_best.pth"
    )

    history_path = (
        Path("results")
        / "vgg"
        / "metrics"
        / "training_history.json"
    )

    history_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # 7. Display experiment configuration
    print("=" * 60)
    print("VGG FASHION-MNIST EXPERIMENT")
    print("=" * 60)

    print(f"Device:             {device}")
    print(f"Random seed:        {SEED}")
    print(f"Training samples:   {len(data['train_dataset'])}")
    print(f"Validation samples: {len(data['validation_dataset'])}")
    print(f"Batch size:         {BATCH_SIZE}")
    print(f"Epochs:             {EPOCHS}")
    print(f"Optimizer:          {OPTIMIZER_NAME}")
    print(f"Learning rate:      {LEARNING_RATE}")
    print(f"Loss function:      {LOSS_FUNCTION_NAME}")
    print(f"Trainable params:   {total_parameters:,}")

    print("=" * 60)

    # 8. Execute training
    result = fit(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=EPOCHS,
        device=device,
        checkpoint_path=checkpoint_path,
    )

    # 9. Save training history and experiment metadata
    experiment_results = {
        "model": "VGG11-FashionMNIST",
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": OPTIMIZER_NAME,
        "loss_function": LOSS_FUNCTION_NAME,
        "trainable_parameters": total_parameters,
        "best_epoch": result["best_epoch"],
        "best_validation_accuracy": (
            result["best_validation_accuracy"]
        ),
        "best_validation_loss": (
            result["best_validation_loss"]
        ),
        "history": result["history"],
    }

    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(
            experiment_results,
            file,
            indent=4
        )

    print(f"\nTraining history saved: {history_path}")


if __name__ == "__main__":
    main()