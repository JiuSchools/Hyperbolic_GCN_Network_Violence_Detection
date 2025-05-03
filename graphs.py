import torch
import numpy as np

def cosine_similarity_matrix(X, tau=0.8):
    sim = torch.nn.functional.cosine_similarity(X.unsqueeze(1), X.unsqueeze(0), dim=-1)
    sim[sim < tau] = 0
    return torch.nn.functional.softmax(sim, dim=-1)

def temporal_relation_matrix(T, gamma=1.0, sigma=np.e):
    i, j = torch.meshgrid(torch.arange(T), torch.arange(T), indexing="ij")
    temporal_dist = torch.exp(-torch.abs(i - j).float() ** gamma / sigma)
    return temporal_dist
