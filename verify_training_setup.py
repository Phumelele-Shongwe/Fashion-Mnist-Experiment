
from src.architectures.lenet5 import LeNet5
from src.architectures.vgg import VGG11FashionMNIST

from configs.experiment_config import (
    SEED,
    EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    OPTIMIZER_NAME,
    LOSS_FUNCTION_NAME,
)


models = {
    "LeNet-5": LeNet5(),
    "VGG": VGG11FashionMNIST(),
}

print("=" * 60)
print("TRAINING CONFIGURATION VERIFICATION")
print("=" * 60)

print("\nSHARED EXPERIMENT CONFIGURATION")
print("-" * 40)

print(f"Random seed:        {SEED}")
print(f"Epochs:             {EPOCHS}")
print(f"Batch size:         {BATCH_SIZE}")
print(f"Learning rate:      {LEARNING_RATE}")
print(f"Optimizer:          {OPTIMIZER_NAME}")
print(f"Loss function:      {LOSS_FUNCTION_NAME}")

print("\nMODEL CONFIGURATION")
print("-" * 40)

for name, model in models.items():

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"{name}:")
    print(f"  Parameters: {parameters:,}")
    print(f"  Output classes: 10")

print("\n" + "=" * 60)
print("TRAINING CONFIGURATION VERIFICATION COMPLETED")
print("=" * 60)