import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from typing import Dict, Any, List
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

# Import ADAS-WSL modules (assumed to be instantiated and passed in)
# WENE, SACT, DualAxisLoss, PASW

logger = logging.getLogger(__name__)

class Trainer:
    """Trainer class for ADAS-WSL enhanced models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        save_dir: Path,
        test_loader: DataLoader = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        # ADAS-WSL extensions:
        unlabeled_loader: DataLoader = None,
        wene = None,
        sact = None,
        dual_axis_loss = None,
        pasw = None,
        labeled_class_dist = None,
        teacher_model = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.unlabeled_loader = unlabeled_loader if unlabeled_loader else train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        
        self.wene = wene
        self.sact = sact
        self.dual_axis_loss = dual_axis_loss
        self.pasw = pasw
        self.labeled_class_dist = labeled_class_dist
        self.teacher_model = teacher_model
        
        # Move model to device
        self.model = self.model.to(device)
        if self.teacher_model:
            self.teacher_model = self.teacher_model.to(device)
        
        # Initialize metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _update_teacher(self):
        """EMA update for teacher model"""
        if self.teacher_model is None:
            return
        alpha = 0.99
        for param, teacher_param in zip(self.model.parameters(), self.teacher_model.parameters()):
            teacher_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    
    def train(self, epochs: int, early_stopping_patience: int = 10) -> Dict[str, List[float]]:
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            
            # ADAS-WSL Phase Transition
            if epoch == 31 and self.wene is not None:
                logger.info("Transitioning to Phase 2: Fitting WENE")
                self.wene.fit()
                
            train_loss, train_acc = self._train_epoch(epoch)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # SACT checkpoints
            if self.sact is not None:
                self.sact.update_thresholds(epoch)
                
            # PASW update at end of epoch
            if self.pasw is not None and self.unlabeled_loader is not None and self.labeled_class_dist is not None:
                new_l1, new_l2, new_l3, kl_div = self.pasw.update(
                    self.model, self.unlabeled_loader, self.labeled_class_dist, self.device
                )
                logger.info(f"PASW Updated lambdas: [{new_l1:.3f}, {new_l2:.3f}, {new_l3:.3f}] - KL: {kl_div:.4f}")
            
            val_loss, val_acc = self._evaluate_loader(self.val_loader)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            logger.info(
                f"Epoch {epoch}/{epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}"
            )
            
            if self.scheduler:
                self.scheduler.step()
                
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint(epoch, val_acc)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
                    
        self.plot_curves()
        self.evaluate()
        return history
    
    def _train_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        # Get lambdas from PASW
        if self.pasw:
            lambda1, lambda2, lambda3 = self.pasw.get_weights()
        else:
            lambda1, lambda2, lambda3 = 0.4, 0.3, 0.3
            
        for batch_idx, data in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
            # Extract inputs/targets; if dataset returns index, grab it.
            if len(data) == 3:
                inputs, noisy_targets, targets = data
                indices = torch.arange(batch_idx * len(inputs), batch_idx * len(inputs) + len(inputs))
            elif len(data) == 4: # If dataset was modified to return indices
                inputs, noisy_targets, targets, indices = data
            else:
                inputs, targets = data
                indices = torch.arange(batch_idx * len(inputs), batch_idx * len(inputs) + len(inputs))
                noisy_targets = targets
                
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            indices = indices.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            if epoch <= 30 or self.dual_axis_loss is None:
                # Phase 1
                loss = criterion(outputs, targets)
                if self.wene is not None:
                    preds = outputs.argmax(dim=1)
                    self.wene.update(indices.cpu().numpy(), preds.cpu().numpy(), targets.cpu().numpy())
                if self.sact is not None:
                    # Update EMA with unlabeled (here we just use the current batch as mock)
                    self.sact.update_ema(torch.softmax(outputs.detach(), dim=1))
            else:
                # Phase 2
                wi = self.wene.get_weights(indices.cpu().numpy()).to(self.device)
                probs = torch.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1)
                
                pseudo_mask = self.sact.filter_pseudo_labels(probs, preds).to(self.device)
                
                # Co-training / Consistency mock per sample metrics
                consist_loss_per_sample = torch.zeros(len(inputs), device=self.device)
                cotrain_loss_per_sample = torch.zeros(len(inputs), device=self.device)
                if self.teacher_model:
                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(inputs)
                    consist_loss_per_sample = F.mse_loss(torch.softmax(outputs, dim=1), 
                                                          torch.softmax(teacher_outputs, dim=1), reduction='none').mean(dim=1)

                loss = self.dual_axis_loss.compute(
                    labeled_logits=outputs, labeled_targets=targets,
                    pseudo_logits=outputs, pseudo_targets=preds, pseudo_mask=pseudo_mask, pseudo_wi=wi,
                    consist_loss_per_sample=consist_loss_per_sample, consist_wi=wi,
                    cotrain_loss_per_sample=cotrain_loss_per_sample, cotrain_wi=wi,
                    unlabeled_probs=probs,
                    lambda1=lambda1, lambda2=lambda2, lambda3=lambda3
                )
                
                if self.sact is not None:
                    self.sact.update_ema(probs.detach())
            
            loss.backward()
            self.optimizer.step()
            self._update_teacher()
            
            total_loss += loss.item()
            with torch.no_grad():
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        return total_loss / len(self.train_loader), correct / total

    def _evaluate_loader(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data in loader:
                if len(data) == 3:
                    inputs, _, targets = data
                else:
                    inputs, targets = data[0], data[1]
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return total_loss / len(loader), correct / total
    
    def _save_checkpoint(self, epoch: int, val_acc: float):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc
        }
        torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{epoch}.pt')
        # Also maintain a latest best
        torch.save(checkpoint, self.save_dir / 'best_model.pt')
    
    def plot_curves(self):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Training and Validation Loss')
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Acc')
        plt.plot(self.val_accs, label='Val Acc')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Training and Validation Accuracy')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png')
        plt.close()
    
    def evaluate(self):
        best_path = self.save_dir / 'best_model.pt'
        if best_path.exists():
            checkpoint = torch.load(best_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        eval_loader = self.test_loader if self.test_loader is not None else self.val_loader
        test_metrics = self._evaluate_loader(eval_loader)
        
        results = {
            'test_loss': test_metrics[0],
            'test_accuracy': test_metrics[1],
            'best_epoch': self.best_epoch,
            'best_val_acc': self.best_val_acc
        }
        with open(self.save_dir / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=4)