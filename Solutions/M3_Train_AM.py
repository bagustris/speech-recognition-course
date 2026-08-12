import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
import time
from htk_featio import read_htk_user_feat

# Run from the Experiments/ directory (the data lives in ./lists and ./am).
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

# Padding sentinel for labels. It must NOT collide with a real class index
# (labels.ciphones maps aa_s2 -> 0, a genuine class), so we use PyTorch's
# default CrossEntropyLoss ignore_index and mask it out of the metrics.
PAD_ID = -100


def load_label_map(mapping_file):
    """Read labels.ciphones (one state name per line) into {name: index}."""
    with open(mapping_file, "r") as f:
        return {line.strip(): idx for idx, line in enumerate(f) if line.strip()}


class SpeechDataset(Dataset):
    def __init__(self, features_file, labels_file, label_map,
                 feature_mean, feature_invstd, context=(0, 0)):
        self.context = context
        self.feature_mean = feature_mean
        self.feature_invstd = feature_invstd

        # Parse the RSCP feature list: each line is "uttid=path[start,end]".
        feats = []
        with open(features_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                uttid, rhs = line.split("=", 1)
                path = rhs.split("[", 1)[0]
                # The RSCP key carries a ".feat" suffix (e.g. "utt1.feat"),
                # while the MLF is keyed by the bare utterance id ("utt1").
                # Strip the extension here using the same transform _load_mlf
                # applies, so the two sides join correctly.
                uttid = os.path.splitext(os.path.basename(uttid))[0]
                feats.append((uttid, self._resolve_rscp_path(path)))

        # Parse the MLF once, keyed by utterance id (its order differs from
        # the RSCP order, so labels must be looked up by id, not position).
        self.labels = self._load_mlf(labels_file, label_map)

        # Keep only utterances present in both, paired by id.
        self.items = [(uttid, path) for uttid, path in feats if uttid in self.labels]
        if not self.items:
            raise RuntimeError(
                "No utterances matched between the feature list '{}' and the "
                "label file '{}'. Check that the RSCP keys and MLF utterance "
                "ids refer to the same utterances.".format(features_file, labels_file)
            )

    @staticmethod
    def _resolve_rscp_path(path):
        # Drop the leading CNTK "..." relative-path marker if present.
        if path.startswith(".../"):
            path = path[len(".../"):]
        return path

    @staticmethod
    def _load_mlf(labels_file, label_map):
        """Parse an HTK MLF into {uttid: np.array([state_index, ...])}.

        Each utterance is introduced by a "<name>.lab" header line; each frame
        line is "start end state_name ...", so the label is column index 2.
        """
        labels = {}
        uttid = None
        current = []
        with open(labels_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line == "#!MLF!#":
                    continue
                if line.startswith('"'):
                    uttid = os.path.splitext(os.path.basename(line.strip('"')))[0]
                    current = []
                elif line == ".":
                    if uttid is not None:
                        labels[uttid] = np.array(current, dtype=np.int64)
                    uttid = None
                    current = []
                else:
                    parts = line.split()
                    if len(parts) >= 3:
                        idx = label_map.get(parts[2])
                        if idx is not None:
                            current.append(idx)
        return labels

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        uttid, path = self.items[idx]
        feature = read_htk_user_feat(path).astype(np.float32)  # (T, feature_dim)
        label = self.labels[uttid]

        # Mean / variance normalization (channel compensation).
        feature = (feature - self.feature_mean) * self.feature_invstd

        # Splice in left/right context frames for the DNN.
        if self.context != (0, 0):
            left, right = self.context
            padded = np.pad(feature, ((left, right), (0, 0)), mode="edge")
            T = feature.shape[0]
            feature = np.stack(
                [padded[i:i + left + right + 1].flatten() for i in range(T)]
            )

        length = min(feature.shape[0], label.shape[0])
        return torch.FloatTensor(feature), torch.LongTensor(label), length


def collate_fn(batch):
    """Pad a batch of variable-length utterances to a common length."""
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    features, labels, lengths = zip(*batch)
    lengths = [min(f.shape[0], l.shape[0]) for f, l in zip(features, labels)]
    max_len = max(lengths)
    feat_dim = features[0].shape[-1]

    features_padded = torch.zeros(len(features), max_len, feat_dim)
    labels_padded = torch.full((len(labels), max_len), PAD_ID, dtype=torch.long)
    for i, (feat, lab, n) in enumerate(zip(features, labels, lengths)):
        features_padded[i, :n] = feat[:n]
        labels_padded[i, :n] = lab[:n]

    return features_padded, labels_padded, torch.tensor(lengths)


class DNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=4):
        super(DNNModel, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Sigmoid()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()]
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class BLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2):
        super(BLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            bidirectional=True, batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)


def _forward(model, model_type, features):
    """Run a padded batch through the model, returning per-frame logits (B, T, C)."""
    if model_type == "DNN":
        batch, seq_len, feat_dim = features.shape
        return model(features.reshape(batch * seq_len, feat_dim)).reshape(batch, seq_len, -1)
    return model(features)


def train_model(model_type, train_loader, val_loader, device, num_epochs,
                feature_dim, num_classes, context):
    num_context_frames = 1 + context[0] + context[1]
    if model_type == "DNN":
        model = DNNModel(feature_dim * num_context_frames, 512, num_classes).to(device)
    else:
        model = BLSTMModel(feature_dim, 512, num_classes).to(device)

    # Real labels live in [0, num_classes-1]; padding uses PAD_ID and is masked
    # out below, so no valid class is ever excluded from the loss or metrics.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    for epoch in range(num_epochs):
        model.train()
        start = time.time()
        total_loss = 0.0
        total_frames = 0
        total_errors = 0

        for batch_features, batch_labels, lengths in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = _forward(model, model_type, batch_features)

            mask = batch_labels != PAD_ID
            valid_outputs = outputs[mask]
            valid_labels = batch_labels[mask]

            loss = criterion(valid_outputs, valid_labels)
            loss.backward()
            optimizer.step()

            n = valid_labels.numel()
            total_loss += loss.item() * n
            total_frames += n
            total_errors += (valid_outputs.argmax(dim=-1) != valid_labels).sum().item()

        epoch_time = time.time() - start
        avg_loss = total_loss / max(total_frames, 1)
        fer = 100.0 * total_errors / max(total_frames, 1)  # Frame Error Rate
        # CNTK-style log line so M3_Plot_Training.py can parse it.
        print(f"Finished Epoch[{epoch + 1} of {num_epochs}]: [CE_Training] "
              f"loss = {avg_loss:.6f} * {total_frames}, "
              f"metric = {fer:.2f}% * {total_frames} ({epoch_time:.1f}s);")

        # Evaluate on the dev set every 5 epochs.
        if (epoch + 1) % 5 == 0 and val_loader is not None:
            model.eval()
            val_frames = 0
            val_errors = 0
            with torch.no_grad():
                for batch_features, batch_labels, lengths in val_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    outputs = _forward(model, model_type, batch_features)
                    mask = batch_labels != PAD_ID
                    valid_outputs = outputs[mask]
                    valid_labels = batch_labels[mask]
                    val_frames += valid_labels.numel()
                    val_errors += (valid_outputs.argmax(dim=-1) != valid_labels).sum().item()
            val_fer = 100.0 * val_errors / max(val_frames, 1)
            print(f"Finished Evaluation [{epoch + 1}]: Minibatch[1-{len(val_loader)}]: "
                  f"metric = {val_fer:.2f}% * {val_frames};")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", help="Network type to train (DNN or BLSTM)",
                        default="DNN")
    args = parser.parse_args()

    model_type = str.upper(args.type)
    if model_type not in ("DNN", "BLSTM"):
        raise RuntimeError("type must be DNN or BLSTM")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the label mapping and normalization stats once, shared across datasets.
    label_map = load_label_map(globals["label_mapping_file"])
    feature_mean = np.loadtxt(globals["feature_mean_file"]).astype(np.float32)
    feature_invstd = np.loadtxt(globals["feature_invstddev_file"]).astype(np.float32)
    # log priors, baked into the saved model so the decoder can form scaled
    # log-likelihoods log p(x|s) = log p(s|x) - log p(s) (see StaticDecoder.py).
    prior = np.loadtxt(globals["label_priors"]).astype(np.float32)
    log_prior = np.log(np.maximum(prior, 1e-10)).astype(np.float32)

    context = (11, 11) if model_type == "DNN" else (0, 0)
    batch_size = 256 if model_type == "DNN" else 4096
    max_epochs = 100 if model_type == "DNN" else 1

    train_dataset = SpeechDataset(globals["features_file"], globals["labels_file"],
                                  label_map, feature_mean, feature_invstd, context)
    val_dataset = SpeechDataset(globals["cv_features_file"], globals["cv_labels_file"],
                                label_map, feature_mean, feature_invstd, context)
    print(f"Loaded {len(train_dataset)} training and {len(val_dataset)} dev utterances")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn)

    model = train_model(model_type, train_loader, val_loader, device, max_epochs,
                        globals["feature_dim"], globals["num_classes"], context)

    # Save a self-contained checkpoint: weights plus everything the decoder
    # needs to reproduce the acoustic scores (architecture, normalization
    # stats, and log priors). This is the PyTorch analogue of the CNTK model
    # that exposed a "ScaledLogLikelihood" output.
    model_path = os.path.join(am_path, model_type)
    os.makedirs(model_path, exist_ok=True)
    torch.save(
        {
            "model_type": model_type,
            "feature_dim": globals["feature_dim"],
            "num_classes": globals["num_classes"],
            "context": list(context),
            "state_dict": model.state_dict(),
            "feature_mean": feature_mean,
            "feature_invstd": feature_invstd,
            "log_prior": log_prior,
        },
        os.path.join(model_path, f"{model_type}_CE.pt"),
    )


if __name__ == "__main__":
    main()
