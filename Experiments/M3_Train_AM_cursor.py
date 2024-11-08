import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import logging
import argparse
from torch.optim.lr_scheduler import ExponentialLR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Global configurations
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
    def __init__(self, features_file, labels_file, feature_mean, feature_invstd, context=(11,11)):
        self.features = self.load_features(features_file)
        self.labels = self.load_labels(labels_file)
        self.feature_mean = feature_mean
        self.feature_invstd = feature_invstd
        self.context = context
        
    def load_features(self, file_path):
        # Load features from RSCP file
        features = []
        with open(file_path, 'r') as f:
            for line in f:
                features.append(np.load(line.strip()))
        return features
    
    def load_labels(self, file_path):
        # Load labels from CIMLF file
        labels = []
        with open(file_path, 'r') as f:
            for line in f:
                labels.append(np.load(line.strip()))
        return labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        # Apply mean and variance normalization
        feature = (feature - self.feature_mean) * self.feature_invstd
        
        # Add context frames
        if self.context != (0,0):
            left, right = self.context
            padded_feature = np.pad(feature, ((left, right), (0, 0)), mode='edge')
            context_features = []
            for i in range(len(feature)):
                context_window = padded_feature[i:i+left+right+1]
                context_features.append(context_window.flatten())
            feature = np.array(context_features)
            
        return torch.FloatTensor(feature), torch.LongTensor(label), len(feature)

class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=4):
        super(DNN, self).__init__()
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.Sigmoid())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Sigmoid())
            
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, num_classes))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class BLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2):
        super(BLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

def train_model(model_type, train_loader, val_loader, device, num_epochs):
    # Set input dimensions based on model type
    if model_type == "DNN":
        input_dim = globals["feature_dim"] * 23  # 23 frames context
        model = DNN(input_dim, 512, globals["num_classes"]).to(device)
    else:
        input_dim = globals["feature_dim"]
        model = BLSTM(input_dim, 512, globals["num_classes"]).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (batch_features, batch_labels, lengths) in enumerate(train_loader):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            
            # Create mask for valid sequence lengths
            mask = torch.arange(batch_labels.size(1), device=device)[None, :] < lengths[:, None]
            outputs = outputs[mask]
            batch_labels = batch_labels[mask]
            
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
            
            if batch_idx % 100 == 0:
                logging.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}, '
                           f'Acc: {100.*correct/total:.2f}%')
        
        scheduler.step()
        
        # Validation
        if epoch % 5 == 0:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_features, batch_labels, lengths in val_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    outputs = model(batch_features)
                    mask = torch.arange(batch_labels.size(1), device=device)[None, :] < lengths[:, None]
                    outputs = outputs[mask]
                    batch_labels = batch_labels[mask]
                    
                    _, predicted = outputs.max(1)
                    val_total += batch_labels.size(0)
                    val_correct += predicted.eq(batch_labels).sum().item()
                    
            logging.info(f'Validation Accuracy: {100.*val_correct/val_total:.2f}%')
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", help="Network type to train (DNN or BLSTM)",
                       default="DNN")
    args = parser.parse_args()
    
    model_type = str.upper(args.type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load mean and variance normalization parameters
    feature_mean = np.loadtxt(globals["feature_mean_file"]).astype(np.float32)
    feature_invstd = np.loadtxt(globals["feature_invstddev_file"]).astype(np.float32)
    
    # Create datasets
    context = (11,11) if model_type == "DNN" else (0,0)
    train_dataset = SpeechDataset(globals["features_file"], globals["labels_file"],
                                 feature_mean, feature_invstd, context)
    val_dataset = SpeechDataset(globals["cv_features_file"], globals["cv_labels_file"],
                               feature_mean, feature_invstd, context)
    
    logging.info(f"Loaded {len(train_dataset)} features and {globals['num_classes']} labels")
    logging.info(f"Loaded {len(val_dataset)} features and {globals['num_classes']} labels")
    
    # Create data loaders
    batch_size = 256 if model_type == "DNN" else 4096
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Train model
    max_epochs = 100 if model_type == "DNN" else 1
    model = train_model(model_type, train_loader, val_loader, device, max_epochs)
    
    # Save model
    model_path = os.path.join(am_path, model_type)
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_path, f"{model_type}_CE.pt"))

if __name__ == "__main__":
    main()