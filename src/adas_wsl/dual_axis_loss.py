import torch
import torch.nn.functional as F

class DualAxisLoss:
    def __init__(self, lambda_saf=0.1):
        self.lambda_saf = lambda_saf

    def compute(self, labeled_logits, labeled_targets,
                pseudo_logits, pseudo_targets, pseudo_mask, pseudo_wi,
                consist_loss_per_sample, consist_wi,
                cotrain_loss_per_sample, cotrain_wi,
                unlabeled_probs,
                lambda1, lambda2, lambda3):

        # Term 1: Supervised loss on labeled data (unchanged from previous project)
        if len(labeled_logits) > 0:
            L_sup = F.cross_entropy(labeled_logits, labeled_targets)
        else:
            L_sup = torch.tensor(0.0, device=pseudo_logits.device)

        # Term 2: Pseudo-label loss with dual-axis filtering
        if pseudo_mask.sum() > 0:
            pl_loss = F.cross_entropy(pseudo_logits[pseudo_mask],
                                       pseudo_targets[pseudo_mask], reduction='none')
            pl_loss = (pl_loss * pseudo_wi[pseudo_mask]).mean()
        else:
            pl_loss = torch.tensor(0.0, device=pseudo_logits.device)

        # Term 3: Consistency loss weighted by wi (sample-axis scale)
        if consist_loss_per_sample is not None and len(consist_loss_per_sample) > 0:
            L_consist = (consist_loss_per_sample * consist_wi).mean()
        else:
            L_consist = torch.tensor(0.0, device=pseudo_logits.device)

        # Term 4: Co-training loss weighted by wi (sample-axis scale)
        if cotrain_loss_per_sample is not None and len(cotrain_loss_per_sample) > 0:
            L_cotrain = (cotrain_loss_per_sample * cotrain_wi).mean()
        else:
            L_cotrain = torch.tensor(0.0, device=pseudo_logits.device)

        # Term 5: Self-adaptive class fairness regularization
        if unlabeled_probs is not None and len(unlabeled_probs) > 0:
            avg_pred = unlabeled_probs.mean(dim=0)
            uniform = torch.ones_like(avg_pred) / avg_pred.shape[0]
            L_SAF = F.kl_div(avg_pred.log() + 1e-8, uniform, reduction='sum')
        else:
            L_SAF = torch.tensor(0.0, device=pseudo_logits.device)

        # Combined dual-axis loss
        L_total = L_sup + lambda1 * pl_loss + lambda2 * L_consist + lambda3 * L_cotrain + self.lambda_saf * L_SAF
        return L_total
