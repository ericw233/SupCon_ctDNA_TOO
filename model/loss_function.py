import torch
import torch.nn as nn
import torch.nn.functional as F

def supcon_loss(Zi, Zj, temperature=0.1):

    ### normalized temnperature-scaled cross-entropy loss
    Z_size = Zi.shape[0]
    Zi = F.normalize(Zi, dim=1)
    Zj = F.normalize(Zj, dim=1)

    Z =  torch.cat([Zi, Zj], dim=0)
    # Z = F.normalize(Z, dim=1)

    similarity_matrix = torch.matmul(Z, Z.T)
    mask = torch.eye(Z_size * 2).bool().to(Z.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

    positives = torch.sum(Zi * Zj, dim=1)
    positives = torch.cat([positives, positives], dim=0)

    numerator = torch.exp(positives / temperature)
    denominator = torch.sum(torch.exp(similarity_matrix / temperature), dim=1)

    loss = -torch.log(numerator / denominator)
    return loss.mean()

### supervised contrastive loss class
class SupConLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device

        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature
        
        batch_size = labels.size(0)
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        sim.masked_fill_(mask, -float('inf')) # exclude self comparisons

        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T) & ~mask # mask for positive pairs 

        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = -(log_prob * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1)
        return loss.mean()

