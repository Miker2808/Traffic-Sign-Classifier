import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms






def main():
    transform = transforms.ToTensor()
    train_data = datasets.GTSRB(root="gtsrb/train", split="train",
                                 download=True, transform=transform)
    test_data = datasets.GTSRB(root="gtsrb/test", split="test",
                                download=True, transform=transform)



if __name__ == "__main__":
    main()