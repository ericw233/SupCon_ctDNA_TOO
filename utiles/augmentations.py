import torch

# def augment_feature(X, mask_prob = 0.2):

#     mask = torch.rand_like(X) < mask_prob
#     X_masked = X.masked_fill(mask, 0)

#     return X_masked

def augment_feature(X, mask_prob = 0.2, noise = 0.01):
    
    noise = torch.randn_like(X) * 0.01
    X = X + noise
    mask = torch.rand_like(X) < mask_prob
    X_masked = X.masked_fill(mask, 0)

    return X_masked
