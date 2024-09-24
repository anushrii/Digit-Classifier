from torch import cat, nn


class DigitClassifier(nn.Module):
    """
    Digit Classifier model using CNN architecture.
    This model is designed to classify digit images from the MNIST dataset.

    The model uses two parallel convolutional blocks (Block 1 and Block 2) that process
    the input image separately.

    The outputs of these blocks are concatenated and passed through fully connected layers
    to produce the final classification output.
    """

    def __init__(self):
        super().__init__()

        # Convolutional layers.
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=1
        )  # kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)
        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)
        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1)
        self.conv3_2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1)

        # Pooling layer.
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 1: Conv1 -> MaxPool -> Conv2_1 -> MaxPool -> Conv3_1
        self.block_1 = nn.Sequential(
            self.conv1,
            self.maxpool,
            nn.ReLU(),
            self.conv2_1,
            self.maxpool,
            nn.ReLU(),
            self.conv3_1,
        )

        # Block 2: Conv1 -> MaxPool -> Conv2_2 -> Maxpool +  Conv3_2
        self.block_2 = nn.Sequential(
            self.conv1,
            self.maxpool,
            nn.ReLU(),
            self.conv2_2,
            self.maxpool,
            nn.ReLU(),
            self.conv3_2,
        )

        # Block 3: Conv3_1 concat Conv3_2 -> Linear -> Linear
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=25088, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=10),
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
        x (torch.Tensor): Input tensor representing a batch of images.

        Returns:
        torch.Tensor: Output tensor representing the class scores for each image in the batch.
        """
        x1 = self.block_1(x)
        x2 = self.block_2(x)
        x = cat([x1, x2], dim=1)
        x = self.fc_layers(x)
        return x
