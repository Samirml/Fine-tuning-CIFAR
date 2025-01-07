import torch
import torch.nn as nn
from torchvision import models


class SimpleCNNModel(nn.Module):
    """
    A simple CNN model with three convolutional layers and two fully connected layers.

    The model is designed for CIFAR-10 classification (10 classes).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Model predictions.
        """
        return self.fc(x)


class ModifiedResNet18(nn.Module):
    """
    A modified version of ResNet-18 pre-trained on ImageNet, fine-tuned for CIFAR-10 classification.

    The model freezes the convolutional layers and only trains the fully connected layers.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        for param in self.resnet18.parameters():
            param.requires_grad = False
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        """
        Forward pass of the modified ResNet-18 model.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Model predictions.
        """
        return self.resnet18(x)
