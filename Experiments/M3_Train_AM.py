import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
from htk_featio import read_htk_user_feat
import time

# Keep the same data directory structure and globals
data_dir = "."
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
        # Read feature file paths
        with open(features_file, 'r') as f:
            self.feature_paths = [line.strip().split('=')[1].split('[')[0] for line in f.readlines()]
            
        # Read labels directly from the file
        with open(labels_file, 'r') as f:
            label_lines = f.readlines()
            self.labels = []
            current_labels = []
            for line in label_lines:
                parts = line.strip().split()
                if len(parts) >= 3:  # Assuming format: start_time end_time label score
                    current_labels.append(int(self._get_label_index(parts[2])))
                else:  # Empty line indicates end of utterance
                    if current_labels:
                        self.labels.append(current_labels)
                        current_labels = []
            if current_labels:  # Add final utterance
                self.labels.append(current_labels)
            
    def _get_label_index(self, label_str):
        # Convert label string to proper class index using the mapping
        # Read from label_mapping_file in globals
        with open(globals["label_mapping_file"], 'r') as f:
            label_map = {line.strip(): idx for idx, line in enumerate(f)}
        return label_map.get(label_str, 0)  # Return 0 for unknown labels
        
    def __len__(self):
        return len(self.feature_paths)
    
    def __getitem__(self, idx):
        # Load feature from HTK file
        feature = read_htk_user_feat(self.feature_paths[idx])
        # print(f"Feature stats - min: {np.min(feature)}, max: {np.max(feature)}, mean: {np.mean(feature)}")
        
        # Get corresponding label
        label = np.array(self.labels[idx])
        # print(f"Label stats - min: {np.min(label)}, max: {np.max(label)}")
            
        return torch.FloatTensor(feature), torch.LongTensor(label)


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
        model = BLSTMModel(globals["feature_dim"], 512, globals["num_classes"]).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    for epoch in range(num_epochs):
        model.train()
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start = time.time()

        train_loss = 0
        total_samples = 0
        correct = 0

        for batch_features, batch_labels, lengths in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs.view(-1, globals["num_classes"]), batch_labels.view(-1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_labels.numel()
            pred = outputs.argmax(dim=-1)
            errors = (pred != batch_labels).sum().item()  # Count misclassified frames
            total_samples += batch_labels.numel()

        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()
            epoch_time = start.elapsed_time(end) / 1000  # Convert to seconds
        else:
            epoch_time = time.time() - start

        samples_per_sec = total_samples / epoch_time
        error_rate = 100.0 * errors / total_samples  # Frame Error Rate percentage

        print(f"Finished Epoch[{epoch+1} of {num_epochs}]: [CE_Training] loss = {train_loss/total_samples:.6f} * {total_samples}, "
              f"metric = {error_rate:.2f}% * {total_samples} {epoch_time:.3f}s ({samples_per_sec:.1f} samples/s);")
        
        # Evaluate on dev set every 5 epochs (cv_checkpoint_interval = 5)
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_errors = 0
            val_samples = 0
            with torch.no_grad():
                for batch_features, batch_labels, lengths in val_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    outputs = model(batch_features)
                    pred = outputs.argmax(dim=-1)
                    val_errors += (pred != batch_labels).sum().item()
                    val_samples += batch_labels.numel()

            val_error_rate = 100.0 * val_errors / val_samples
            print(f"Finished Evaluation [{epoch+1}]: Minibatch[1-{len(val_loader)}]: metric = {val_error_rate:.2f}% * {val_samples};")

    return model


def collate_fn(batch):
    # Sort batch by sequence length (descending)
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    
    # Separate features and labels
    features, labels = zip(*batch)
    
    # Get sequence lengths
    lengths = [min(feat.shape[0], lab.shape[0]) for feat, lab in zip(features, labels)]
    max_len = max(lengths)
    
    # Pad sequences with matching dimensions
    features_padded = torch.zeros(len(features), max_len, features[0].shape[-1])
    labels_padded = torch.zeros(len(labels), max_len, dtype=torch.long)
    
    for i, ((feat, lab), length) in enumerate(zip(zip(features, labels), lengths)):
        features_padded[i, :length] = feat[:length]
        labels_padded[i, :length] = lab[:length]
        
    return features_padded, labels_padded, torch.tensor(lengths)


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
    max_epochs = 10 if model_type == "DNN" else 1

    train_dataset = SpeechDataset(globals["features_file"], globals["labels_file"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


    val_dataset = SpeechDataset(globals["cv_features_file"], globals["cv_labels_file"])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    model = train_model(model_type, train_loader, val_loader, device, max_epochs)

    # Save model
    model_path = os.path.join(am_path, model_type)
    os.makedirs(model_path, exist_ok=True)
    torch.save(
        model.state_dict(), os.path.join(model_path, f"{model_type}_CE_model.pt")
    )


if __name__ == "__main__":
    main()