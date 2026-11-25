"""
Compact VGG-11-inspired architecture adapted for Fashion-MNIST.

Architecture characteristics:
    - Eight 3x3 convolutional layers
    - ReLU activations
    - Five 2x2 max-pooling operations
    - Increasing feature-map depth
    - Global average pooling
    - Ten-class classification head

Expected input:
    (N, 1, 32, 32)

Output:
    (N, 10) raw classification logits
"""

import torch
import torch.nn as nn


# VGG architecture configuration
VGG_CONFIG = [
    (32, 1),
    (64, 1),
    (128, 2),
    (256, 2),
    (256, 2),
]


class VGG11FashionMNIST(nn.Module):
    """
    Compact VGG-11-inspired model for Fashion-MNIST.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        layers = []

        in_channels = 1

        # Build convolutional blocks
        for out_channels, num_convolutions in VGG_CONFIG:

            for _ in range(num_convolutions):

                layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    )
                )

                layers.append(nn.ReLU(inplace=True))

                in_channels = out_channels

            # Downsample after each convolutional block.
            layers.append(
                nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                )
            )

        self.features = nn.Sequential(*layers)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(
            in_features=256,
            out_features=num_classes
        )

    def forward(self, x):

        # Validate input dimensions.
        if x.ndim != 4 or x.shape[1:] != (1, 32, 32):
            raise ValueError(
                "VGG11FashionMNIST expects input shape "
                "(N, 1, 32, 32). "
                f"Received {tuple(x.shape)}."
            )

        # Convolutional feature extraction.
        x = self.features(x)

        # Global average pooling.
        x = self.avgpool(x)

        # Flatten feature representation.
        x = torch.flatten(x, start_dim=1)

        # Classification.
        x = self.classifier(x)

        return x


def count_parameters(model):
    """Count trainable model parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":

    model = VGG11FashionMNIST(num_classes=10)

    dummy_input = torch.randn(64, 1, 32, 32)

    output = model(dummy_input)

    convolution_layers = sum(
        isinstance(layer, nn.Conv2d)
        for layer in model.modules()
    )

    pooling_layers = sum(
        isinstance(layer, nn.MaxPool2d)
        for layer in model.modules()
    )

    print("=" * 45)
    print("VGG ARCHITECTURE VERIFICATION")
    print("=" * 45)

    print(f"Input shape:          {tuple(dummy_input.shape)}")
    print(f"Output shape:         {tuple(output.shape)}")
    print(f"Convolution layers:   {convolution_layers}")
    print(f"Max-pooling layers:   {pooling_layers}")

    print(
        f"Trainable parameters: {count_parameters(model):,}"
    )

    print("=" * 45)