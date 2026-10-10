"""Task and dataset utilities for the Flower/PyTorch experiment."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


_fds_cache: dict[tuple[object, ...], FederatedDataset] = {}

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def create_partitioner(
    partitioning_mode: str,
    num_partitions: int,
    dirichlet_alpha: float = 0.3,
    partition_seed: int = 42,
):
    """Create the requested deterministic partitioner."""
    if partitioning_mode == "iid":
        return IidPartitioner(num_partitions=num_partitions)
    if partitioning_mode == "dirichlet":
        if dirichlet_alpha <= 0:
            raise ValueError("Dirichlet alpha must be positive")
        return DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=dirichlet_alpha,
            seed=partition_seed,
        )
    raise ValueError(f"Unknown partitioning mode: {partitioning_mode}")


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    seed: int | None = None,
    partitioning_mode: str = "iid",
    dirichlet_alpha: float = 0.3,
    partition_seed: int = 42,
):
    """Load one CIFAR-10 partition using a reproducible partition scheme."""
    if partitioning_mode == "iid":
        cache_key = ("iid", num_partitions)
    elif partitioning_mode == "dirichlet":
        cache_key = ("dirichlet", num_partitions, dirichlet_alpha, partition_seed)
    else:
        raise ValueError(f"Unknown partitioning mode: {partitioning_mode}")
    if cache_key not in _fds_cache:
        partitioner = create_partitioner(
            partitioning_mode,
            num_partitions,
            dirichlet_alpha,
            partition_seed,
        )
        _fds_cache[cache_key] = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = _fds_cache[cache_key].load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    trainloader = DataLoader(
        partition_train_test["train"],
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset():
    """Load test set and return dataloader."""
    # Load entire test set
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
