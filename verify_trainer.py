
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import fit


# Reproducibility
torch.manual_seed(76)

# Synthetic dataset
images = torch.randn(256, 1, 32, 32)
labels = torch.randint(0, 10, (256,))

dataset = TensorDataset(images, labels)

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
)

validation_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
)

# Small test model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(32 * 32, 10),
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)

device = torch.device("cpu")

# Execute training engine
result = fit(
    model=model,
    train_loader=train_loader,
    validation_loader=validation_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=2,
    device=device,
    checkpoint_path="models/checkpoints/trainer_test.pth",
)

# Verification
checkpoint = torch.load(
    result["checkpoint_path"],
    map_location=device,
    weights_only=True,
)

print("\nTRAINER VERIFICATION")
print("=" * 50)

print("History keys:", list(result["history"].keys()))
print("Best epoch:", result["best_epoch"])
print("Checkpoint epoch:", checkpoint["epoch"])
print(
    "Checkpoint model parameters:",
    len(checkpoint["model_state_dict"])
)

assert len(result["history"]["train_loss"]) == 2
assert len(result["history"]["validation_loss"]) == 2
assert checkpoint["epoch"] == result["best_epoch"]

print("\nALL TRAINER CHECKS PASSED")