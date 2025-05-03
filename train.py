import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import geoopt
from .data_utils import VideoAnomalyDataset
from .model import HyperbolicVideoAnomalyDetector
from .loss import k_max_loss

def train_model(train_data, epochs=10, batch_size=1, dim=1024, lr=1e-4, use_cuda=False):
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    manifold = geoopt.PoincareBall()
    model = HyperbolicVideoAnomalyDetector(dim, manifold).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = VideoAnomalyDataset(train_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for features_batch, labels_batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            loss_batch = 0
            optimizer.zero_grad()
            for features, label in zip(features_batch, labels_batch):
                features = features.to(device)
                scores = model(features)
                bin_label = 1 if label > 0 else 0
                loss = k_max_loss(scores, bin_label)
                loss_batch += loss
            loss_batch /= batch_size
            loss_batch.backward()
            optimizer.step()
            total_loss += loss_batch.item()
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")
    return model
