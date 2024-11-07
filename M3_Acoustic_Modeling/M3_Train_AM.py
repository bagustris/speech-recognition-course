import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse

# Keep the same data directory structure and globals
data_dir = "../Experiments"
list_path = os.path.join(data_dir, "lists")
am_path = os.path.join(data_dir, "am")

globals = {
    "features_file": os.path.join(list_path, "feat_train.rscp"),
    "labels_file": os.path.join(am_path, "labels_all.cimlf"),
    "cv_features_file": os.path.join(list_path, "feat_dev.rscp"),
    "cv_labels_file": os.path.join(am_path, "labels_all.cimlf"),
    "label_mapping_file": os.path.join(am_path, "labels.ciphones"),
    "label_priors": os.path.join(am_path, "labels_ciprior.ascii"),
    "feature_mean_file": os.path.join(am_path, "feat_mean.ascii"),
    "feature_invstddev_file": os.path.join(am_path, "feat_invstddev.ascii"),
    "feature_dim": 40,
    "num_classes": 120,
}


class SpeechDataset(Dataset):
    def __init__(self, features_file, labels_file):
        self.features = np.load(features_file)
        self.labels = np.load(labels_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class DNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(DNNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class BLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(BLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)


def train_model(model_type, train_loader, val_loader, device, num_epochs):
    if model_type == "DNN":
        model = DNNModel(globals["feature_dim"], 512, globals["num_classes"]).to(device)
    else:
        model = BLSTMModel(globals["feature_dim"], 512, globals["num_classes"]).to(
            device
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    for epoch in range(num_epochs):
        model.train()
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                outputs = model(batch_features)
                val_loss += criterion(outputs, batch_labels).item()

        print(f"Epoch {epoch}: Val Loss = {val_loss/len(val_loader)}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--type",
        help="Network type to train (DNN or BLSTM)",
        default="DNN",
    )
    args = parser.parse_args()

    model_type = str.upper(args.type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup data loaders
    batch_size = 256 if model_type == "DNN" else 4096
    max_epochs = 100 if model_type == "DNN" else 1

    train_dataset = SpeechDataset(globals["features_file"], globals["labels_file"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = SpeechDataset(globals["cv_features_file"], globals["cv_labels_file"])
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = train_model(model_type, train_loader, val_loader, device, max_epochs)

    # Save model
    model_path = os.path.join(am_path, model_type)
    os.makedirs(model_path, exist_ok=True)
    torch.save(
        model.state_dict(), os.path.join(model_path, f"{model_type}_CE_model.pt")
    )


if __name__ == "__main__":
    main()
