import os
import numpy as np
import torch
from torch.utils.data import Dataset

def load_dataset_from_folder(base_path):
    data = []
    label_names = sorted(os.listdir(base_path))
    label_to_index = {name: idx for idx, name in enumerate(label_names)}

    for label_name in label_names:
        label_folder = os.path.join(base_path, label_name)
        for file in os.listdir(label_folder):
            if file.endswith('.npy'):
                path = os.path.join(label_folder, file)
                label = label_to_index[label_name]
                data.append((path, label))
    return data, label_to_index

class VideoAnomalyDataset(Dataset):
    def __init__(self, data, max_frames=300):
        self.data = data
        self.max_frames = max_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        features = np.load(path, mmap_mode='r')
        if features.shape[0] > self.max_frames:
            indices = np.linspace(0, features.shape[0] - 1, self.max_frames).astype(int)
            features = features[indices]
        features_tensor = torch.tensor(features, dtype=torch.float32)
        return features_tensor, label
