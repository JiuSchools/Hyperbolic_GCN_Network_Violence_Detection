import torch
import torch.nn as nn
import geoopt
from .graphs import cosine_similarity_matrix, temporal_relation_matrix

class HyperbolicGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, manifold):
        super().__init__()
        self.manifold = manifold
        self.lin = nn.Linear(in_features, out_features)

    def forward(self, x, A):
        x_tangent = self.manifold.logmap0(x)
        x_transformed = self.lin(x_tangent)
        agg = torch.matmul(A, x_transformed)
        x_out = self.manifold.expmap0(agg)
        return x_out

class HyperbolicVideoAnomalyDetector(nn.Module):
    def __init__(self, dim, manifold=None):
        super().__init__()
        self.manifold = manifold or geoopt.PoincareBall()
        self.gcn1 = HyperbolicGCNLayer(dim, dim, self.manifold)
        self.gcn2 = HyperbolicGCNLayer(dim, dim, self.manifold)
        self.fc = nn.Linear(dim * 2, 1)

    def forward(self, X):
        A_fsg = cosine_similarity_matrix(X)
        A_trg = temporal_relation_matrix(X.size(0))
        X_fsg = self.gcn1(X, A_fsg)
        X_trg = self.gcn2(X, A_trg)
        X_concat = torch.cat([X_fsg, X_trg], dim=-1)
        score = torch.sigmoid(self.fc(X_concat)).squeeze()
        return score
