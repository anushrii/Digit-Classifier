import logging
from typing import Tuple

import torchvision
from torch.utils.data import DataLoader


def load_MNIST_data(
    root: str = "./data", batch_size: int = 64
) -> Tuple[DataLoader, DataLoader]:
    """
    Load MNIST data from torch vision datasets using torch dataloader

    Args:
    root (str): Path to store the data
    batch_size (int): Batch size for the data loader

    Returns:
    Tuple[DataLoader, DataLoader]: Tuple of train and test data loaders.
    """

    train_mnist_data = torchvision.datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    test_mnist_data = torchvision.datasets.MNIST(
        root=root,
        download=True,
        train=False,
        transform=torchvision.transforms.ToTensor(),
    )
    print(
        f"Successfully loaded MNIST data. Train data shape: {train_mnist_data.data.shape}, Test data shape: {test_mnist_data.data.shape}"
    )
    train_data_loader = DataLoader(
        train_mnist_data, batch_size=batch_size, shuffle=True
    )
    test_data_loader = DataLoader(test_mnist_data, batch_size=batch_size, shuffle=True)

    return train_data_loader, test_data_loader
