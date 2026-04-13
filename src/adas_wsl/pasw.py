import torch
import torch.nn.functional as F
import numpy as np

class ProportionAlignedStrategyWeighting:
    def __init__(self, base_lambda1=0.4, base_lambda2=0.3, base_lambda3=0.3,
                 beta=1.0, gamma=0.5):
        self.base_lambda1 = base_lambda1
        self.base_lambda2 = base_lambda2
        self.base_lambda3 = base_lambda3
        self.beta = beta
        self.gamma = gamma
        # Current weights start at base values
        self.lambda1 = base_lambda1
        self.lambda2 = base_lambda2
        self.lambda3 = base_lambda3

    def update(self, model, unlabeled_loader, labeled_class_dist, device):
        # Called at the end of every epoch
        model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in unlabeled_loader:
                images = batch[0].to(device)
                outputs = torch.softmax(model(images), dim=1)
                all_probs.append(outputs.cpu())
                if len(all_probs) * images.shape[0] >= 1000:
                    break
        
        if len(all_probs) > 0:
            all_probs = torch.cat(all_probs, dim=0)
            P_pred = all_probs.mean(dim=0)
            
            P_expected = torch.tensor(labeled_class_dist, dtype=torch.float32)
            P_expected = P_expected / P_expected.sum()
            # KL divergence: how different are predicted vs expected distributions
            kl_div = F.kl_div(P_pred.log() + 1e-8, P_expected, reduction='sum').item()
            # Adjust strategy weights based on KL divergence
            new_l1 = self.base_lambda1 * np.exp(-self.beta * kl_div)
            new_l2 = self.base_lambda2 * (1 + self.gamma * kl_div)
            new_l3 = self.base_lambda3
            # Normalize so weights sum to 1
            total = new_l1 + new_l2 + new_l3
            self.lambda1 = new_l1 / total
            self.lambda2 = new_l2 / total
            self.lambda3 = new_l3 / total
        else:
            kl_div = 0.0

        model.train()
        return self.lambda1, self.lambda2, self.lambda3, kl_div

    def get_weights(self):
        return self.lambda1, self.lambda2, self.lambda3
