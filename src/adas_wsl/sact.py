import torch
import numpy as np

class SelfAdaptiveClassThreshold:
    def __init__(self, num_classes, ema_decay=0.99):
        self.num_classes = num_classes
        self.ema_decay = ema_decay
        # Initialize all thresholds and EMA trackers to 0.5
        self.tau_c = np.full(num_classes, 0.5, dtype=np.float32)
        self.class_conf_ema = np.full(num_classes, 0.5, dtype=np.float32)
        self.global_ema = 0.5
        self.update_epochs = [20, 50, 80]

    def update_ema(self, unlabeled_probs):
        # Called every batch with model predictions on unlabeled data
        # unlabeled_probs: tensor of shape (batch_size, num_classes)
        if unlabeled_probs is None or len(unlabeled_probs) == 0:
            return
        
        max_conf = unlabeled_probs.max(dim=1).values.mean().item()
        self.global_ema = self.ema_decay * self.global_ema + (1 - self.ema_decay) * max_conf
        for c in range(self.num_classes):
            class_avg = unlabeled_probs[:, c].mean().item()
            self.class_conf_ema[c] = self.ema_decay * self.class_conf_ema[c] + (1 - self.ema_decay) * class_avg

    def update_thresholds(self, epoch):
        # Called only at checkpoint epochs 20, 50, 80
        if epoch in self.update_epochs:
            max_conf = max(self.class_conf_ema)
            for c in range(self.num_classes):
                self.tau_c[c] = self.global_ema * (self.class_conf_ema[c] / (max_conf + 1e-8))
                self.tau_c[c] = max(0.3, min(0.95, self.tau_c[c]))

    def get_threshold(self, class_idx):
        # Returns tau_c for a given class index
        return float(self.tau_c[class_idx])

    def filter_pseudo_labels(self, probs, predicted_classes):
        # Returns boolean mask: True = sample passes threshold gate
        mask = torch.zeros(len(probs), dtype=torch.bool)
        for i, (prob, cls) in enumerate(zip(probs.max(dim=1).values, predicted_classes)):
            if prob.item() >= self.tau_c[cls.item()]:
                mask[i] = True
        return mask
