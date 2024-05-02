from typing import cast  # Python type hints
from torch.types import Device  # PyTorch type hints

import torch  # Used for accessing cuda/backend
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from tqdm import trange

from utils import train_test_dataloaders

class MyModel(nn.Module):
    def __init__(self, device: Device = None) -> None:
        super().__init__()  # Initalize module w/ PyTorch
        self._model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

        self.to(device)  # Shift the model to a device (if required)

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)


def train_model(
    model: nn.Module,
    device: Device,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    epochs: int = 11,
) -> None:
    """
    Train a PyTorch model using the specified optimizer and data loader.
    Parameters:
        model (nn.Module): The neural network model to be trained.
        device (Device): The device to perform computations on (e.g., 'cuda' or 'cpu').
        optimizer (optim.Optimizer): The optimization algorithm used for training.
        train_loader (DataLoader): The data loader providing batches of training data.
        epochs (int): The number of training epochs (default is 11).
    Notes:
        This function trains the specified model using the provided optimizer and data loader for
        the specified number of epochs. It computes the loss using cross-entropy and performs
        backpropagation to update the model parameters.
    Example usage:
        train_model(my_model, device, my_optimizer, train_loader)
    """
    model.train()  # Enable gradient tracking
    for epoch in trange(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Shift the tensors to the target device
            data = cast(Tensor, data).to(device)
            target = cast(Tensor, target).to(device)

            optimizer.zero_grad()  # Reset all optimized gradients
            output: Tensor = model(data)
            loss = F.cross_entropy(output, target)

            loss.backward()
            optimizer.step()


def test_model(model: MyModel, device: Device, test_loader: DataLoader) -> None:
    """
    Test the given PyTorch model on a test dataset.
    Args:
        model (nn.Module): The PyTorch model to be tested.
        device (Device): The device (CPU or GPU) where the model and data will be loaded.
        test_loader (DataLoader): DataLoader providing batches of test data.
    Notes:
        This function evaluates the performance of a PyTorch model on a given test dataset.
        It computes the average loss and accuracy over the test set.
    Example:
        test_model(my_model, device, test_loader)
    """
    with torch.inference_mode():
        test_loss, correct = 0, 0
        for data, target in test_loader:
            # Shift the tensors to the target device
            data = cast(Tensor, data).to(device)
            target = cast(Tensor, target).to(device)

            output: Tensor = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            test_loss += F.cross_entropy(output, target).item()
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def train_test_dataloaders(root: str) -> tuple[DataLoader, DataLoader]:
    train = FashionMNIST(root, train=True, transform=ToTensor(), download=True)
    test = FashionMNIST(root, train=False, transform=ToTensor(), download=True)

    # Return the dataloaders in a tuple[train_dataloader, test_dataloader]
    return DataLoader(train, 32, shuffle=True), DataLoader(test, 32, shuffle=False)


def main():
    if torch.cuda.is_available():
        gpu_score = torch.cuda.get_device_capability()
        if gpu_score >= (8, 0) and torch.backends.cuda.is_built():
            torch.backends.cuda.matmul.allow_tf32 = True

        device = "cuda"  # Utilize GPU resources for training/testing
    else:
        device = "cpu"  # No additional optimizations are avalible

    train, test = train_test_dataloaders("./datasets")  # Load datasets from PyTorch
    model = MyModel(device)  # Initalize model/optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    train_model(model, device, optimizer, train, 10)  # Train the model on the dataset
    test_model(model, device, test)


if __name__ == "__main__":
    main()