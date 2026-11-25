
"""
LeNet-5 architecture adapted for Fashion-MNIST.

Expected input:
    Batch of grayscale images with shape (N, 1, 32, 32)

Output:
    Raw classification logits with shape (N, 10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrainableSubsampling(nn.Module):
    """
    LeNet-style trainable subsampling layer.

    Performs:
        1. Average pooling
        2. Trainable scaling
        3. Trainable bias
        4. Tanh activation
    """

    def __init__(self, channels):
        super().__init__()

        self.scale = nn.Parameter(
            torch.ones(1, channels, 1, 1)
        )

        self.bias = nn.Parameter(
            torch.zeros(1, channels, 1, 1)
        )

    def forward(self, x):

        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.scale * x + self.bias

        return torch.tanh(x)


class LeNet5(nn.Module):
    """
    LeNet-5 architecture with partial connectivity in C3.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # C1: First convolutional layer
        self.c1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5
        )

        # S2: First trainable subsampling layer
        self.s2 = TrainableSubsampling(channels=6)

        # C3: Partially connected convolutional layer
        # Original LeNet-5 C3 connection pattern.
        # Each output map receives input from selected
        # S2 feature maps rather than all six.
        self.c3_connections = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 0],
            [5, 0, 1],
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 0],
            [4, 5, 0, 1],
            [5, 0, 1, 2],
            [0, 1, 3, 4],
            [1, 2, 4, 5],
            [0, 2, 3, 5],
            [0, 1, 2, 3, 4, 5],
        ]

        # Create one independent convolution for
        # every connected input-output feature map pair.

        self.c3_weights = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=5,
                    bias=False
                )
                for _ in connected_maps
            ])
            for connected_maps in self.c3_connections
        ])

        # One bias per C3 output feature map.
        self.c3_bias = nn.Parameter(
            torch.zeros(1, 16, 1, 1)
        )

        # --------------------------------------------------
        # S4: Second trainable subsampling layer
        # --------------------------------------------------

        self.s4 = TrainableSubsampling(channels=16)

        # --------------------------------------------------
        # C5: Final convolutional layer
        # --------------------------------------------------

        self.c5 = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=5
        )

        # --------------------------------------------------
        # F6: Fully connected layer
        # --------------------------------------------------

        self.f6 = nn.Linear(
            in_features=120,
            out_features=84
        )

        # --------------------------------------------------
        # Output classification layer
        # --------------------------------------------------

        self.output = nn.Linear(
            in_features=84,
            out_features=num_classes
        )

    def forward(self, x):

        # Expect 32 x 32 input images.
        if x.ndim != 4 or x.shape[1:] != (1, 32, 32):
            raise ValueError(
                "LeNet5 expects input shape (N, 1, 32, 32). "
                f"Received {tuple(x.shape)}."
            )

        # C1 -> S2
        x = torch.tanh(self.c1(x))
        x = self.s2(x)

        # C3: Partial connectivity
        c3_outputs = []

        for output_map, connected_maps in enumerate(
            self.c3_connections
        ):

            feature_map = None

            for input_map, convolution in zip(
                connected_maps,
                self.c3_weights[output_map]
            ):

                input_feature = x[:, input_map:input_map + 1]

                contribution = convolution(input_feature)

                if feature_map is None:
                    feature_map = contribution
                else:
                    feature_map = feature_map + contribution

            c3_outputs.append(feature_map)

        x = torch.cat(c3_outputs, dim=1)

        x = torch.tanh(x + self.c3_bias)

        # S4 -> C5
        x = self.s4(x)
        x = torch.tanh(self.c5(x))

        # Flatten 120 feature maps into a vector.
        x = torch.flatten(x, start_dim=1)

        # F6 -> Output
        x = torch.tanh(self.f6(x))

        # Return raw logits for CrossEntropyLoss.
        x = self.output(x)

        return x


def count_parameters(model):
    """Count trainable model parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":

    model = LeNet5(num_classes=10)

    # Simulate a batch of 64 Fashion-MNIST images.
    dummy_input = torch.randn(64, 1, 32, 32)

    output = model(dummy_input)

    print("=" * 45)
    print("LENET-5 ARCHITECTURE VERIFICATION")
    print("=" * 45)

    print(f"Input shape:       {tuple(dummy_input.shape)}")
    print(f"Output shape:      {tuple(output.shape)}")

    print(
        "C3 connections:    ",
        sum(len(maps) for maps in model.c3_connections)
    )

    print(
        "Trainable params:  ",
        f"{count_parameters(model):,}"
    )

    print("=" * 45)