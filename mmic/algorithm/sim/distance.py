import torch
from geomloss import SamplesLoss

def cosine_sim(x1, x2, dim=-1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return w12 / (w1 * w2).clamp(min=eps)


def wasserstein_distance(p,q,proximation = False):
    if proximation:
        loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, diameter = 0.1)  # Sinkhorn Wasserstein-2 distance
        bs = p.shape[0]
        wasserstein_dist = torch.zeros(bs, device=p.device)
        p,q = p[:,:10000],q[:,:10000]
        for i in range(bs):
            value = loss_fn(p[i].unsqueeze(0), q[i].unsqueeze(0))
            wasserstein_dist[i] = value  # compute it by each instance

        return wasserstein_dist
    p_sorted, _ = torch.sort(p, dim=1)
    q_sorted, _ = torch.sort(q, dim=1)
    return torch.mean(torch.abs(p_sorted - q_sorted), dim=1)


def z_score_normalize(x):
    """
    Z-score 标准化
    :param x: 输入数据
    :return: 标准化后的数据
    """
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    return (x - mean) / std