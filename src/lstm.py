import numpy as np
import torch
from torch import nn, optim
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib


class CDMPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

        self.feature_names = [
            'MISS_DISTANCE',
            'RELATIVE_SPEED',
            'RELATIVE_POSITION_R',
            'RELATIVE_POSITION_T',
            'RELATIVE_POSITION_N',
            'RELATIVE_VELOCITY_R',
            'RELATIVE_VELOCITY_T',
            'RELATIVE_VELOCITY_N',
            'OBJECT1_CR_R',
            'OBJECT1_CT_T',
            'OBJECT1_CN_N',
            'OBJECT2_CR_R'
        ]

    def extract_features(self, event):
        features = []

        for cdm in event:
            cdm_features = []
            for f in self.feature_names:
                value = cdm.get(f, 0.0)
                if value is None:
                    value = 0.0
                cdm_features.append(float(value))
            features.append(cdm_features)

        return np.array(features)

    def fit_transform(self, events):
        all_features = []

        for event in events:
            event_features = self.extract_features(event)
            all_features.extend(event_features)

        all_features = np.array(all_features)
        self.scaler.fit(all_features)
        self.is_fitted = True

        sequences = []
        for event in events:
            event_features = self.extract_features(event)
            if len(event_features) > 0:
                scaled_features = self.scaler.transform(event_features)
                sequences.append(scaled_features)

        return sequences

    def transform(self, events):
        if not self.is_fitted:
            raise RuntimeError("Call fit before transform")

        sequences = []
        for event in events:
            event_features = self.extract_features(event)
            if len(event_features) > 0:
                scaled_features = self.scaler.transform(event_features)
                sequences.append(scaled_features)

        return sequences

    def save(self, filepath):
        joblib.dump(
            {
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "is_fitted": self.is_fitted,
            }, filepath
        )

    def load(self, filepath):
        data = joblib.load(filepath)
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.is_fitted = data["is_fitted"]

class EventDataset(Dataset):
    def __init__(self, event_set, features):
        self._event_set = event_set
        self._features = features
        self._features_length = len(features)
        self._max_event_length = max(len(e) for e in event_set)
        self._features_stats = {"mean": [], "stddev": []}

        for f in self._features:
            values = []
            for event in event_set:
                for cdm in event:
                    val = cdm.get(f)
                    if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
                        values.append(val)
            self._features_stats["mean"].append(np.mean(values))
            self._features_stats["stddev"].append(np.std(values))

    def __getitem__(self, idx):
        event = self._event_set[idx]
        x = torch.zeros(self._max_event_length, self._features_length)

        for cdm_idx, cdm in enumerate(event):
            values = [
                self._normalize_feature(cdm.get(feature, 0.0), mean, std)
                for feature, mean, std in zip(self._features, self._features_stats["mean"], self._features_stats["stddev"])
            ]
            x[cdm_idx] = torch.tensor(values)
        label = 1 if event[-1]["MISS_DISTANCE"] < 100 else 0

        length = torch.tensor(len(event), dtype = torch.int64)
        label_tensor  = torch.tensor(label, dtype = torch.float32)
        return x, length, label_tensor

    def __len__(self):
        return len(self._event_set)

    def _normalize_feature(self, val, mean, std):
        if val is None or not isinstance(val, (int, float)) or np.isnan(val):
            val = 0.0

        return (val - mean) / (std + 1e-8)

def collate_fn(batch):
    xs, lengths, labels = zip(*batch)
    return (
        torch.stack(xs),
        torch.tensor(lengths, dtype=torch.int64),
        torch.tensor(labels, dtype=torch.float32)
    )

class CollisionRiskLSTM(nn.Module):
    def __init__(self, event_set, features, hidden_size = 64, num_layers = 1, dropout = 0.2):
        super().__init__()

        self._features = features
        self._features_length = len(features)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.dataset = EventDataset(event_set, features)

        self.lstm = nn.LSTM(
            input_size = self._features_length,
            hidden_size = self.hidden_size,
            num_layers = num_layers,
            batch_first = True
        )

        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        packed = rnn_utils.pack_padded_sequence(x, lengths, batch_first = True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first = True)

        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, self.hidden_size)
        final_outputs = out.gather(1, idx).squeeze(1)
        final_outputs = self.dropout(final_outputs)

        logits = self.fc(final_outputs)
        return self.sigmoid(logits).squeeze(1)

    def learn(self, num_epochs = 10, batch_size = 32, lr = 0.001, val_split = 0.15):
        total_size = len(self.dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True, collate_fn = collate_fn)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, collate_fn = collate_fn)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr = lr)

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            for x, lengths, labels in train_loader:
                x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
                optimizer.zero_grad()
                preds = self(x, lengths)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            self.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for x, lengths, labels in val_loader:
                    x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
                    preds = self(x, lengths)
                    pred_labels = (preds > 0.5).float()
                    correct += (pred_labels == labels).sum().item()
                    total += labels.size(0)

            accuracy = correct / total
            print(f"Epoch {epoch + 1:02d} | Train Loss: {avg_train_loss:.3f} | Val Accuracy: {accuracy:.3f}")