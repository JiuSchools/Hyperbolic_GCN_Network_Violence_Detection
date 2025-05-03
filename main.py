from data_utils import load_dataset_from_folder
from train import train_model
from evaluate import evaluate_model

if __name__ == "__main__":
    train_path = "/path/to/Train Features"
    test_path = "/path/to/Test Features"

    train_data, label_map = load_dataset_from_folder(train_path)
    test_data, _ = load_dataset_from_folder(test_path)

    model = train_model(train_data, epochs=3, batch_size=1, dim=1024, lr=1e-4, use_cuda=False)
    evaluate_model(model, test_data)
