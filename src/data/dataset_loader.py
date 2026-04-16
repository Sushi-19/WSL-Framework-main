import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import os
import urllib.request

def get_datasets(dataset_name, data_dir='./data'):
    """Returns train_dataset and test_dataset for the requested dataset_name."""
    os.makedirs(data_dir, exist_ok=True)
    
    if dataset_name == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
        
    elif dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    elif dataset_name == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=test_transform)

    elif dataset_name == 'svhn':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        train_dataset = datasets.SVHN(data_dir, split='train', download=True, transform=train_transform)
        test_dataset = datasets.SVHN(data_dir, split='test', download=True, transform=test_transform)
        # SVHN targets are stored in 'labels' as a numpy array, convert to tensor for consistency
        train_dataset.targets = torch.tensor(train_dataset.labels, dtype=torch.long)
        test_dataset.targets = torch.tensor(test_dataset.labels, dtype=torch.long)

    elif dataset_name == 'stl10':
        transform = transforms.Compose([
            transforms.Resize(32), # Resize to 32x32 for consistency
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.STL10(data_dir, split='train', download=True, transform=transform)
        test_dataset = datasets.STL10(data_dir, split='test', download=True, transform=transform)
        train_dataset.targets = torch.tensor(train_dataset.labels, dtype=torch.long)
        test_dataset.targets = torch.tensor(test_dataset.labels, dtype=torch.long)

    elif dataset_name == 'cifar10n':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
        
        cifar10n_path = os.path.join(data_dir, 'CIFAR-10_human.pt')
        if not os.path.exists(cifar10n_path):
            print("CIFAR-10_human.pt not found. For full CIFAR-10N, manually download it from github.com/UCSC-REAL/cifar-10-100n into data/. Using standard CIFAR-10 labels for now.")
        else:
            human_labels = torch.load(cifar10n_path)
            # Assuming worse_label is used as requested by standard noisy label simulations
            train_dataset.targets = human_labels['worse_label'].tolist()

    elif dataset_name == 'animal10n':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dir = os.path.join(data_dir, 'animal10n', 'training')
        test_dir = os.path.join(data_dir, 'animal10n', 'testing')
        if not os.path.exists(train_dir):
            print("Animal-10N not found. For full Animal-10N, manually download from github.com/UCSC-REAL/Animal-10N into data/animal10n/.")
            # Dummy fallback for mocking
            train_dataset = datasets.FakeData(size=1000, image_size=(3, 32, 32), num_classes=10, transform=transform)
            test_dataset = datasets.FakeData(size=200, image_size=(3, 32, 32), num_classes=10, transform=transform)
            train_dataset.targets = [label for _, label in train_dataset]
            test_dataset.targets = [label for _, label in test_dataset]
        else:
            train_dataset = datasets.ImageFolder(train_dir, transform=transform)
            test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_dataset, test_dataset
