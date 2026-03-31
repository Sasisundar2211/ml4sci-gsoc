import torch

def augment(x):
    # flips
    if torch.rand(1) > 0.5:
        x = torch.flip(x, dims=[2])

    if torch.rand(1) > 0.5:
        x = torch.flip(x, dims=[1])

    # stronger noise
    x = x + torch.randn_like(x) * 0.1

    # random crop
    x = x[:, :, 10:115, 10:115] # Fixed indexing for batch size dimension

    return x
