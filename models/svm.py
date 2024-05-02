import numpy as np
import pandas as pd

import torch
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA


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


def main():
    train, test = train_test_dataframes("./datasets")
    
    # Subsample the training dataset to reduce computational overhead
    train = train.sample(10_000)
    
    X_train, y_train = train.drop("label", axis=1), train["label"]
    X_test, y_test = test.drop("label", axis=1), test["label"]

    clf = SVC(cache_size=60000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    main()
