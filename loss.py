import torch
import torch.nn.functional as F

def k_max_loss(scores, label, k_ratio=1/16):
    k = int(len(scores) * k_ratio) + 1
    topk_scores, _ = torch.topk(scores, k)
    avg_topk = topk_scores.mean()
    target = torch.tensor(float(label)).to(scores.device)
    return F.binary_cross_entropy(avg_topk, target)
