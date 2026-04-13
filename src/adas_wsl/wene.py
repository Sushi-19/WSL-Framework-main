import torch
import numpy as np
from scipy.stats import beta
from sklearn.mixture import BayesianGaussianMixture

class WrongEventNoiseEstimator:
    def __init__(self, num_samples):
        # Counter array: one slot per training sample
        self.wrong_event = np.zeros(num_samples, dtype=np.float32)
        self.weights = np.ones(num_samples, dtype=np.float32)
        self.fitted = False

    def update(self, indices, predictions, targets):
        # Called after every batch during Phase 1
        for idx, pred, target in zip(indices, predictions, targets):
            if pred != target:
                self.wrong_event[idx] += 1

    def fit(self):
        # Called once after Phase 1 ends
        counts = self.wrong_event.reshape(-1, 1)
        gmm = BayesianGaussianMixture(n_components=2, random_state=42)
        gmm.fit(counts)
        # Component with lower mean = clean group
        probs = gmm.predict_proba(counts)
        clean_component = np.argmin(gmm.means_)
        self.weights = probs[:, clean_component].astype(np.float32)
        self.fitted = True

    def get_weights(self, indices):
        # Returns wi for a batch of sample indices
        if not self.fitted:
            return torch.ones(len(indices))
        return torch.tensor(self.weights[indices], dtype=torch.float32)
