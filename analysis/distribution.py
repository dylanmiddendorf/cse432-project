# Obtained from: https://stackoverflow.com/q/63746182/19496432

from torchvision.datasets import FashionMNIST

trainset = FashionMNIST(root="./datasets", train=True, download=True)
print("Min Pixel Value:", trainset.data.min())
print("Max Pixel Value:", trainset.data.max())
print("Mean Pixel Value:", trainset.data.float().mean())
print("Pixel Values Stdev:", trainset.data.float().std())
print("Scaled Mean Pixel Value:", trainset.data.float().mean() / 255.0)
print("Scaled Pixel Values Stdev:", trainset.data.float().std() / 255.0)
