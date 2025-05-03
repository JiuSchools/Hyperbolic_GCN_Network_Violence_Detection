import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, test_data):
    device = next(model.parameters()).device
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for path, label in test_data:
            features = np.load(path, mmap_mode='r')
            features = torch.tensor(features, dtype=torch.float32).to(device)
            score = model(features)
            pred = int(score.mean() > 0.5)
            y_pred.append(pred)
            y_true.append(1 if label > 0 else 0)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Violence'], yticklabels=['Normal', 'Violence'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("Confusion_matrix.png")
