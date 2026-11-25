
"""
Shared experimental configuration.

These settings are common to the LeNet-5 and VGG experiments.
"""

# 1. Reproducibility
SEED = 76

# 2. Dataset configuration
DATA_DIR = "data"
NUM_CLASSES = 10
VALIDATION_FRACTION = 0.10
BATCH_SIZE = 64
NUM_WORKERS = 0

# 3. Image normalization
IMAGE_MEAN = (0.5,)
IMAGE_STD = (0.5,)

# 4. Training configuration
EPOCHS = 10
LEARNING_RATE = 0.001
OPTIMIZER_NAME = "Adam"
LOSS_FUNCTION_NAME = "CrossEntropyLoss"