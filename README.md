# Fashion-MNIST: LeNet-5 vs. VGG

A reproducible deep learning experiment comparing a classical LeNet-5 convolutional neural network with a compact VGG-11-inspired architecture on the Fashion-MNIST image classification dataset.

## 1. Project Overview

This project investigates how CNN architectural complexity affects image classification performance.

Two architectures are implemented and evaluated under a shared experimental pipeline:

- **LeNet-5:** A classical CNN featuring tanh activations, 5×5 convolution filters, and trainable subsampling.
- **Compact VGG-11-inspired:** A deeper CNN featuring 3×3 convolution filters, ReLU activations, max pooling, and global average pooling.

The objective is to compare their predictive performance, model complexity, training behaviour, and class-specific classification errors.

## 2. Experimental Results

Both models were trained for 10 epochs using the same dataset split, preprocessing pipeline, batch size, optimizer, and learning rate.

| Metric | LeNet-5 | Compact VGG |
|---|---:|---:|
| Trainable parameters | 60,850 | 2,308,234 |
| Best validation accuracy | 89.70% | 91.77% |
| Best epoch | 9 | 6 |
| Test accuracy | 89.09% | 91.47% |
| Test macro F1-score | 89.07% | 91.44% |
| Test loss | 0.3083 | 0.2476 |

The compact VGG architecture achieved a 2.38 percentage-point improvement in test accuracy, at the cost of approximately 37.94 times more trainable parameters.

These results illustrate the trade-off between model complexity and predictive performance.

## 3. Dataset and Preprocessing

The experiment uses the Fashion-MNIST dataset, containing 70,000 grayscale images across 10 clothing categories.

The original training set is divided into:

- Training: 54,000 images
- Validation: 6,000 images
- Official test set: 10,000 images

Both architectures use the same preprocessing pipeline:

1. Pad the original 28×28 images to 32×32.
2. Convert images to PyTorch tensors.
3. Normalize pixel values using mean 0.5 and standard deviation 0.5.

The resulting input shape is `(N, 1, 32, 32)`.

A fixed random seed of 76 is used to support reproducibility.

## 4. Model Architectures

### LeNet-5

The implementation includes:

- Three convolutional stages.
- Two trainable subsampling stages.
- Tanh activation functions.
- Partial connectivity in the C3 layer.
- A fully connected classifier producing 10 output logits.

### Compact VGG-11-inspired

The implementation includes:

- Eight convolutional layers.
- Five max-pooling stages.
- 3×3 convolution filters.
- ReLU activation functions.
- Adaptive global average pooling.
- A fully connected classifier producing 10 output logits.

**Note:** The VGG implementation is adapted for Fashion-MNIST and is not an exact reproduction of the canonical VGG-11 architecture.

## 5. Experimental Configuration

| Setting | Value |
|---|---|
| Framework | PyTorch |
| Dataset | Fashion-MNIST |
| Random seed | 76 |
| Batch size | 64 |
| Epochs | 10 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | CrossEntropyLoss |
| Image dimensions | 32 × 32 |
| Hardware | CPU |

Both models use the same training engine and evaluation procedures.

The best model checkpoint is selected using validation accuracy, with validation loss used as a tie-breaker.

## 6. Project Structure

```text
Fashion_Mnist_Experiment/
â”‚
â”œâ”€â”€ configs/
â”‚   â””â”€â”€ experiment_config.py
â”‚
â”œâ”€â”€ data/
â”‚
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ checkpoints/
â”‚   â””â”€â”€ model_summaries/
â”‚
â”œâ”€â”€ notebooks/
â”‚
â”œâ”€â”€ results/
â”‚   â”œâ”€â”€ lenet5/
â”‚   â”‚   â”œâ”€â”€ metrics/
â”‚   â”‚   â”œâ”€â”€ plots/
â”‚   â”‚   â””â”€â”€ predictions/
â”‚   â”‚
â”‚   â”œâ”€â”€ vgg/
â”‚   â”‚   â”œâ”€â”€ metrics/
â”‚   â”‚   â”œâ”€â”€ plots/
â”‚   â”‚   â””â”€â”€ predictions/
â”‚   â”‚
â”‚   â””â”€â”€ experiment_summary/
â”‚
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ architectures/
â”‚   â”œâ”€â”€ preprocessing/
â”‚   â”œâ”€â”€ training/
â”‚   â”œâ”€â”€ evaluation/
â”‚   â””â”€â”€ utils/
â”‚
â”œâ”€â”€ tests/
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ README.md
â””â”€â”€ .gitignore
```

## 7. Installation

Clone the repository:

```bash
git clone <repository-url>
cd Fashion_Mnist_Experiment
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The Fashion-MNIST dataset is downloaded automatically when the preprocessing pipeline is executed for the first time.

## 8. Running the Experiments

Run the training scripts from the project root:

```bash
python -m src.training.train_lenet5
python -m src.training.train_vgg
```

Evaluate the trained models:

```bash
python -m src.evaluation.evaluate_lenet5
python -m src.evaluation.evaluate_vgg
```

Generate training visualizations:

```bash
python -m src.evaluation.plot_training_history
```

Generate confusion matrices:

```bash
python -m src.evaluation.plot_confusion_matrix
```

Analyse training behaviour:

```bash
python -m src.evaluation.analyse_training_history
```

Generate the consolidated experiment report:

```bash
python -m src.evaluation.summarise_experiment
```

## 9. Results and Analysis

The `results/` directory contains the generated experimental evidence, including:

- Training and validation loss curves.
- Training and validation accuracy curves.
- Raw confusion matrices.
- Normalized confusion matrices.
- Per-class evaluation metrics.
- Prediction outputs.
- Consolidated model comparison results.

The experiment indicates that the deeper VGG-inspired architecture performs better overall, particularly for visually similar clothing categories such as pullovers and coats.

However, both architectures exhibit signs of overfitting towards the later training epochs.

## 10. Limitations and Future Work

This experiment provides a controlled comparison of two CNN architectures, but several limitations remain:

- Only two architectures were evaluated.
- Both models were trained for a fixed number of epochs.
- No systematic hyperparameter search was conducted.
- Regularization and data augmentation strategies were not systematically investigated.
- The official test set had been explored in an earlier, separate experiment. Consequently, the test results should not be interpreted as a completely untouched final evaluation.

Potential future investigations include:

- Dropout and weight decay.
- Data augmentation.
- Learning-rate scheduling and early stopping.
- Model pruning and quantization.
- Computational efficiency and inference latency.
- Deployment on resource-constrained edge devices.

## 11. Reproducibility

The project uses a shared configuration, fixed random seeds, consistent data splits, and common training and evaluation routines.

These design choices help ensure that differences in model performance can be investigated under a consistent experimental setup.

---

**Project focus:** Convolutional Neural Networks, image classification, architectural comparison, reproducible experimentation, and model evaluation.
