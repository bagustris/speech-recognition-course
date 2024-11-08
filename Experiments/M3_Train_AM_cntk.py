import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
from htk_featio import read_htk_user_feat
from typing import List, Tuple
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Keep the same data directory structure and globals
data_dir = "./"
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
    def __init__(self, features_file: str, labels_file: str, feature_mean: np.ndarray, 
                 feature_invstd: np.ndarray, context: Tuple[int, int] = (0,0)):
        self.features = self.load_features(features_file)
        self.labels = self.load_labels(labels_file)
        self.feature_mean = feature_mean
        self.feature_invstd = feature_invstd
        self.context = context
        
    def load_features(self, file_path: str) -> List[np.ndarray]:
        features = []
        with open(file_path, 'r') as f:
            for line in f:
                features.append(read_htk_user_feat(line.strip()))
        return features
    
    def load_labels(self, file_path: str) -> List[np.ndarray]:
        labels = []
        current_labels = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '#!MLF!#':
                    continue
                if line.startswith('"'):  # New utterance
                    if current_labels:
                        labels.append(np.array(current_labels))
                    current_labels = []
                elif line == '.':  # End of utterance
                    if current_labels:
                        labels.append(np.array(current_labels))
                    current_labels = []
                else:  # Label line
                    try:
                        state_id = int(line.split()[0])
                        current_labels.append(state_id)
                    except:
                        continue
        return labels
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
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

class DNNModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 4):
        super(DNNModel, self).__init__()
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class BLSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2, dropout: float = 0.0):
        super(BLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, 
            bidirectional=True, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)

def train_model(model_type: str, train_loader: DataLoader, val_loader: DataLoader, 
                device: torch.device, num_epochs: int):
    # Initialize model
    if model_type == "DNN":
        model = DNNModel(globals["feature_dim"] * 23, 512, globals["num_classes"]).to(device)
    else:
        model = BLSTMModel(globals["feature_dim"], 512, globals["num_classes"]).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                   factor=0.5, patience=3, verbose=True)

    best_val_loss = float('inf')
    val_losses = []
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        start_time = time.time()
        
        for batch_idx, (batch_features, batch_labels, lengths) in enumerate(train_loader):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            
            if model_type == "DNN":
                batch_size, seq_len, feat_dim = batch_features.size()
                batch_features = batch_features.reshape(-1, feat_dim)
                batch_labels = batch_labels.reshape(-1)
                outputs = model(batch_features)
            else:
                outputs = model(batch_features)
                mask = torch.arange(batch_labels.size(1), device=device)[None, :] < lengths[:, None]
                outputs = outputs[mask]
                batch_labels = batch_labels[mask]
            
            loss = criterion(outputs, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                logging.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

        avg_train_loss = train_loss / len(train_loader)
        epoch_time = time.time() - start_time
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels, lengths in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                if model_type == "DNN":
                    batch_size, seq_len, feat_dim = batch_features.size()
                    batch_features = batch_features.reshape(-1, feat_dim)
                    batch_labels = batch_labels.reshape(-1)
                    outputs = model(batch_features)
                else:
                    outputs = model(batch_features)
                    mask = torch.arange(batch_labels.size(1), device=device)[None, :] < lengths[:, None]
                    outputs = outputs[mask]
                    batch_labels = batch_labels[mask]
                
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        logging.info(f'Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, '
                    f'Val Loss = {avg_val_loss:.4f}, Accuracy = {accuracy:.2f}%, '
                    f'Time = {epoch_time:.2f}s')
        
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            save_checkpoint(model, optimizer, epoch, avg_val_loss, 
                          os.path.join(am_path, f"{model_type}_CE.pt"), True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f'Early stopping triggered after {epoch + 1} epochs')
                break
    
    return model

def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, 
                   loss: float, path: str, is_best: bool = False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    if is_best:
        best_path = path.replace('.pt', '_best.pt')
        torch.save(checkpoint, best_path)

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
    
    # Save final model
    model_path = os.path.join(am_path, model_type)
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_path, f"{model_type}_CE.pt"))

if __name__ == "__main__":
    main()