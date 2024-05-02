import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2, ToTensor


def train_test_dataloaders(root: str) -> tuple[DataLoader, DataLoader]:
    """
    Obtain train and test data loaders for the Fashion-MNIST dataset.

    Args:
        root (str): Root directory where the Fashion-MNIST dataset will be saved.

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing the train and test data loaders.
    """
    train_transform = v2.Compose(
        [
            v2.RandomHorizontalFlip(),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.2860], [0.3530]),
            v2.RandomErasing(),
        ]
    )

    test_transform = v2.Compose(
        [
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.2860], [0.3530]),
        ]
    )

    train = FashionMNIST(root, train=True, transform=train_transform, download=True)
    test = FashionMNIST(root, train=False, transform=test_transform, download=True)

    # Return the dataloaders in a tuple[train_dataloader, test_dataloader]
    return DataLoader(train, 32, shuffle=True), DataLoader(test, 32, shuffle=False)


"""
def train_test_dataloaders(root: str) -> tuple[DataLoader, DataLoader]:
    train = FashionMNIST(root, train=True, transform=ToTensor(), download=True)
    test = FashionMNIST(root, train=False, transform=ToTensor(), download=True)

    # Return the dataloaders in a tuple[train_dataloader, test_dataloader]
    return DataLoader(train, 32, shuffle=True), DataLoader(test, 32, shuffle=False)
"""


def train_test_dataframes(root: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    def dataset_to_dataframe(dataset: FashionMNIST) -> pd.DataFrame:
        images: np.ndarray = dataset.data.flatten(1).numpy()
        n_images, n_pixels = images.shape  # Gather relevent dataset metadata

        images = images.reshape(n_images, -1)  # Flatten image arrays
        data = pd.DataFrame(images, columns=[f"pixel_{i}" for i in range(n_pixels)])
        data["label"] = dataset.targets.numpy()  # Append labels to the dataframe
        return data

    transform = v2.Compose([v2.ToPureTensor(), v2.ToDtype(torch.float32, scale=True)])
    train = FashionMNIST(root, train=True, transform=transform, download=True)
    test = FashionMNIST(root, train=False, transform=transform, download=True)
    return dataset_to_dataframe(train), dataset_to_dataframe(test)
