"""
Shared Fashion-MNIST preprocessing and data loading.

Both LeNet-5 and VGG use this module to ensure
identical dataset preparation.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from configs.experiment_config import (
    SEED,
    DATA_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    VALIDATION_FRACTION,
    IMAGE_MEAN,
    IMAGE_STD,
)


# 1. Image transformations
def get_transforms():

    transform = transforms.Compose([

        # Pad 28x28 images to 32x32.
        transforms.Pad(
            padding=2,
            fill=0
        ),

        # Convert PIL image into a tensor.
        transforms.ToTensor(),

        # Normalize pixel values to approximately [-1, 1].
        transforms.Normalize(
            mean=IMAGE_MEAN,
            std=IMAGE_STD
        ),

    ])

    return transform


# 2. Dataset loading and splitting
def load_datasets():

    transform = get_transforms()

    # Download/load the original training dataset.
    full_train_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    # Download/load the official test dataset.
    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    # Determine training and validation sizes.
    train_size = int(
        (1 - VALIDATION_FRACTION)
        * len(full_train_dataset)
    )

    validation_size = (
        len(full_train_dataset) - train_size
    )

    # Use a dedicated generator for reproducible splitting.
    split_generator = torch.Generator().manual_seed(SEED)

    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        [train_size, validation_size],
        generator=split_generator
    )

    return train_dataset, validation_dataset, test_dataset


# 3. DataLoader creation
def get_dataloaders(batch_size=BATCH_SIZE):

    train_dataset, validation_dataset, test_dataset = (
        load_datasets()
    )

    # Reproducible training data shuffling.
    train_generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=train_generator
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return {
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset,
        "test_dataset": test_dataset,

        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "test_loader": test_loader,
    }