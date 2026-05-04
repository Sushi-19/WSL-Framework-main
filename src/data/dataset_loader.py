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
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
        
    elif dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform)

    elif dataset_name == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.SVHN(data_dir, split='train', download=True, transform=transform)
        test_dataset = datasets.SVHN(data_dir, split='test', download=True, transform=transform)
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
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
        
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
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install the datasets library using: pip install datasets")

        print("Downloading Animal-10N dataset via Hugging Face...")
        hf_dataset = load_dataset("dgrnd4/animals-10")
        
        class HFDatasetWrapper(torch.utils.data.Dataset):
            def __init__(self, hf_ds, transform=None):
                self.hf_ds = hf_ds
                self.transform = transform
                self.targets = [item["label"] for item in self.hf_ds]
                
            def __len__(self):
                return len(self.hf_ds)
                
            def __getitem__(self, idx):
                item = self.hf_ds[idx]
                img = item["image"].convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, item["label"]
        
        # Check if test split exists, otherwise split train manually
        if 'test' in hf_dataset:
            train_split = hf_dataset['train']
            test_split = hf_dataset['test']
        else:
            print("No test split found, creating an 80/20 split from train data...")
            splits = hf_dataset['train'].train_test_split(test_size=0.2, seed=42)
            train_split = splits['train']
            test_split = splits['test']
            
        train_dataset = HFDatasetWrapper(train_split, transform)
        test_dataset = HFDatasetWrapper(test_split, transform)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_dataset, test_dataset
