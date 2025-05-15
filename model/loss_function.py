import torch
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

