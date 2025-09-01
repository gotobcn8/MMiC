import torch

def compute_model_l2_norm(model):
    l2_norm = torch.norm(
        torch.stack([torch.norm(p, 2) for p in model.parameters() if p.requires_grad]), 2
    )
    return l2_norm.item()