from typing import Optional
from pathlib import Path

from badger_utils.torch.device_aware import default_device
from badger_utils.view.image_utils import ImageUtils

import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from gtn.datasets.random_dataset import RandomDataset, RandomDatasetWithTargets

project_path = Path(__file__).absolute().parent.parent.parent


class Datasets:
    num_workers = 0  # breaks PyCharm debugger when 1

    # @staticmethod
    # def mnist_dataloader(batch_size: int = 1):
    #     ds = datasets.MNIST(root='./data', train=True, download=True,
    #                         transform=lambda image: ImageUtils.image_to_tensor(image))
    #     return DataLoader(ds, batch_size=batch_size)

    @staticmethod
    def mnist_dataset(train: bool) -> Dataset:
        return datasets.MNIST(project_path/'data', train=train, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))

    @staticmethod
    def cifar10_dataset(train: bool) -> Dataset:
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        return datasets.CIFAR10(project_path/'data/cifar-10-python', 
                                train=train, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)
                              ]))

    @classmethod
    def mnist_dataloader(cls, batch_size: int, train: bool) -> DataLoader:
        dataset = Datasets.mnist_dataset(train)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=cls.num_workers)

    @classmethod
    def random_dataloader(cls, batch_size: int, item_size: int, count: int, device: Optional[str] = None) -> DataLoader:
        if device is None:
            device = default_device()
        dataset = RandomDataset(item_size, count, device)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cls.num_workers)

    @classmethod
    def random_dataloader_with_targets(cls, batch_size: int, random_item_size: int, count: int,
                                       target_classes: int, device: Optional[str] = None) -> DataLoader:
        if device is None:
            device = default_device()
        dataset = RandomDatasetWithTargets(random_item_size, count, target_classes, device)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cls.num_workers)
