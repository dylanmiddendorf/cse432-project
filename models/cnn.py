from typing import cast  # Python type hints
from torch.types import Device  # PyTorch type hints

import torch  # Used for accessing cuda/backend
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from tqdm import trange

from sklearn.ensemble import RandomForestClassifier
from codecarbon import EmissionsTracker

from utils import train_test_dataloaders


class MyModel(nn.Module):
    def __init__(self, device: Device = None) -> None:
        super().__init__()  # Initalize module w/ PyTorch
        self._conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2),
        )

        self._embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9216, 1024),
            nn.Dropout(0.25),
        )

        self._classifier = nn.Sequential(
            self._embedding, nn.ReLU(), nn.Linear(1024, 10)
        )

        self.to(device)  # Shift the model to a device (if required)

    def forward(self, x: Tensor) -> Tensor:
        x = self._conv_block_1(x)
        return self._classifier(x)

    def drop_classifier(self) -> None:
        self._classifier = self._embedding


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

    Example:
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

            cross_entropy_loss = F.cross_entropy(output, target)
            loss = cross_entropy_loss  # Can be utilized for L1/L2 regularization
            loss.backward()
            optimizer.step()


def test_model(model: nn.Module, device: Device, test_loader: DataLoader) -> None:
    """
    Test the given PyTorch model on a test dataset.

    Args:
        model (nn.Module): The PyTorch model to be tested.
        device (Device): The device (CPU or GPU) where the model and data will be loaded.
        test_loader (DataLoader): DataLoader providing batches of test data.

    Notes:
        This function evaluates the performance of a PyTorch model on a given test dataset.
        It computes the average loss and accuracy over the test set.
    """
    with torch.inference_mode():
        test_loss, correct = 0.0, 0.0
        for data, target in test_loader:
            # Shift the tensors to the target device
            data = cast(Tensor, data).to(device)
            target = cast(Tensor, target).to(device)

            output: Tensor = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            test_loss += F.cross_entropy(output, target).item()
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= float(len(test_loader.dataset))

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / float(len(test_loader.dataset)),
        )
    )


def raw_inference(model: nn.Module, loader: DataLoader) -> tuple[Tensor, Tensor]:
    """
    Perform raw inference using the provided model over the entire dataset.

    Args:
        model (nn.Module): The neural network model to perform inference with.
        loader (DataLoader): The data loader containing the Fashion-MNIST dataset.

    Returns:
        tuple[Tensor, Tensor]: A tuple containing the raw outputs of the model and the corresponding labels.
            The first element is a Tensor containing the raw outputs of the model.
            The second element is a Tensor containing the labels of the dataset.
    """
    with torch.inference_mode():
        dataset: FashionMNIST = loader.dataset  # Type hint dataset for mypy
        data = (dataset.data / 255.0).unsqueeze(1)  # Normalize and format dataset
        return model(data), dataset.targets


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

    train_model(model, device, optimizer, train, 1)  # Train the model on the dataset
    test_model(model, device, test)  # Display the original accuracy w/ FC classifier

    model = model.to("cpu")  # Shift model to CPU for scikit-learn
    model.drop_classifier()  # Drop FC classifier in favor of ensemble

    X_train, y_train = raw_inference(model, train)
    X_test, y_test = raw_inference(model, test)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = torch.from_numpy(clf.predict(X_test))
    correct = y_pred.eq(y_test).sum().item()
    print(correct / y_test.shape[0])


if __name__ == "__main__":
    main()
