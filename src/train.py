import sys
import os
# Force Linux to recognize the local src directory to resolve ModuleNotFoundError
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
import argparse
from torch.utils.data import DataLoader, random_split
import copy
from collections import Counter

from models.noise_robust_model import NoiseRobustModel
from training.train import Trainer
from data.dataset_loader import get_datasets
from adas_wsl.wene import WrongEventNoiseEstimator
from adas_wsl.sact import SelfAdaptiveClassThreshold
from adas_wsl.dual_axis_loss import DualAxisLoss
from adas_wsl.pasw import ProportionAlignedStrategyWeighting

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockPASW:
    def __init__(self, l1, l2, l3):
        self.l1, self.l2, self.l3 = l1, l2, l3
    def update(self, *args):
        return self.l1, self.l2, self.l3, 0.0
    def get_weights(self):
        return self.l1, self.l2, self.l3

def parse_args():
    parser = argparse.ArgumentParser(description="ADAS-WSL Framework trainer.")
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100', 'svhn', 'cifar10n', 'stl10', 'animal10n', 'mnist'])
    parser.add_argument('--model_type', type=str, default='simple_cnn',
                        choices=['simple_cnn', 'resnet', 'mlp'])
    parser.add_argument('--strategy', type=str, default='adas_wsl',
                        choices=['baseline', 'pseudo_labeling', 'consistency', 'co_training', 'adas_wsl'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default='matrix_results_50epochs',
                        help='Sub-directory under experiments/ to save results (e.g. matrix_results_50epochs)')
    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device} | Dataset: {args.dataset} | Model: {args.model_type} | Strategy: {args.strategy}")
    
    save_dir = Path(f'experiments/{args.output_dir}/{args.dataset}_{args.model_type}_{args.strategy}')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load requested dataset
    train_dataset, test_dataset = get_datasets(args.dataset, 'data')
    
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Compute Labeled Class Distribution for PASW
    class_counts = Counter([train_dataset[i][1] if isinstance(train_dataset[i][1], int) else train_dataset[i][1].item() for i in range(len(train_dataset))])
    num_classes = len(class_counts)
    if num_classes == 0:
        num_classes = 10 # fallback
    labeled_class_dist = [class_counts.get(i, 0) for i in range(num_classes)]
    if sum(labeled_class_dist) == 0:
        labeled_class_dist = [1] * num_classes

    # Initialize Modules conditionally based on strategy
    wene = None
    sact = None
    dual_axis_loss = None
    pasw = None
    
    if args.strategy != 'baseline':
        wene = WrongEventNoiseEstimator(num_samples=len(train_loader.dataset))
        sact = SelfAdaptiveClassThreshold(num_classes=num_classes)
        dual_axis_loss = DualAxisLoss(lambda_saf=0.1)
        
        if args.strategy == 'adas_wsl':
            pasw = ProportionAlignedStrategyWeighting(base_lambda1=0.4, base_lambda2=0.3, base_lambda3=0.3)
        elif args.strategy == 'pseudo_labeling':
            pasw = MockPASW(1.0, 0.0, 0.0)
        elif args.strategy == 'consistency':
            pasw = MockPASW(0.0, 1.0, 0.0)
        elif args.strategy == 'co_training':
            pasw = MockPASW(0.0, 0.0, 1.0)
    
    # Setup model parameters
    model = NoiseRobustModel(
        model_type=args.model_type,
        num_classes=num_classes,
        loss_type='cross_entropy',
        beta=0.95
    )
    teacher_model = copy.deepcopy(model) if args.strategy != 'baseline' else None
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        unlabeled_loader=train_loader, # Mocking unlabeled loader with train array for now
        optimizer=optimizer,
        device=device,
        save_dir=save_dir,
        wene=wene,
        sact=sact,
        dual_axis_loss=dual_axis_loss,
        pasw=pasw,
        labeled_class_dist=labeled_class_dist,
        teacher_model=teacher_model
    )
    
    logger.info("Starting training...")
    history = trainer.train(epochs=args.epochs, early_stopping_patience=10)
    
    torch.save(history, save_dir / 'training_history.pt')
    logger.info("Training completed!")

if __name__ == '__main__':
    main()