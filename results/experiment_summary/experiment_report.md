# Fashion-MNIST CNN Experiment

## 1. Experiment configuration

- Random seed: 76
- Epochs: 10
- Batch size: 64
- Learning rate: 0.001
- Optimizer: Adam
- Loss function: CrossEntropyLoss
- Validation fraction: 10%

## 2. Model comparison

| Metric | LeNet-5 | VGG |
|---|---:|---:|
| Trainable parameters | 60,850 | 2,308,234 |
| Best validation accuracy | 0.8970 | 0.9177 |
| Best validation epoch | 9.0000 | 6.0000 |
| Test accuracy | 0.8909 | 0.9147 |
| Test loss | 0.3083 | 0.2476 |
| Macro precision | 0.8910 | 0.9145 |
| Macro recall | 0.8909 | 0.9147 |
| Macro F1 | 0.8907 | 0.9144 |
| Final accuracy gap | 0.0284 | 0.0354 |
| Final loss gap | 0.0791 | 0.1138 |

## 3. Per-class performance

| Class | LeNet-5 F1 | VGG F1 |
|---|---:|---:|
| T-shirt/top | 0.8468 | 0.8573 |
| Trouser | 0.9798 | 0.9850 |
| Pullover | 0.8035 | 0.8829 |
| Dress | 0.8904 | 0.9155 |
| Coat | 0.8135 | 0.8609 |
| Sandal | 0.9715 | 0.9780 |
| Shirt | 0.7163 | 0.7530 |
| Sneaker | 0.9522 | 0.9637 |
| Bag | 0.9732 | 0.9817 |
| Ankle boot | 0.9598 | 0.9658 |
| macro avg | 0.8907 | 0.9144 |
| weighted avg | 0.8907 | 0.9144 |

## 4. Experimental observations

- Both models were trained using the same dataset preparation and training configuration.
- VGG achieved higher overall test accuracy in this experiment.
- LeNet-5 has substantially fewer trainable parameters.
- VGG showed a larger final training-validation accuracy gap.
- The architectures differ in several design choices; this experiment does not isolate network depth as the sole cause of the performance difference.

## 5. Methodological note

The official Fashion-MNIST test set had been explored in an earlier experiment. Consequently, it should not be described as a completely untouched final holdout. The reported test metrics are presented as an evaluation and were not used to tune this experiment.
