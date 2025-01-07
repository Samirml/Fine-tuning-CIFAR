import torchvision
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class CifarDataset(Dataset):
    """
    Custom dataset class for loading CIFAR-10 data and applying transformations.

    Args:
        image_size (int): Target size of images (default is 32).
        train (bool): If True, loads the training data, otherwise loads the test data (default is True).

    Methods:
        __getitem__(index) -> tuple: Returns a transformed image and its label.
        __len__() -> int: Returns the number of samples in the dataset.
    """

    def __init__(self, image_size: int = 32, train=True):
        self.dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True)
        self.transforms = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ToFloat(max_value=255),
            ToTensorV2(),
        ])

    def __getitem__(self, index) -> tuple:
        """
        Get the transformed image and its label at the given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: (transformed image, label)
        """
        image, label = self.dataset[index]
        if self.transforms:
            image = self.transforms(image=np.array(image))["image"]
        return image, label

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: The size of the dataset.
        """
        return len(self.dataset)
