import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
import numpy as np

class EventDataset(Dataset):
    def __init__(self, event_set, features):
        self._event_set = event_set
        self._features = features
        self._features_length = len(features)
        self._max_event_length = max(len(e) for e in event_set)

        self._features_stats = {"mean":[], "stddev":[]}
        for f in self._features:
            values = []
            for e in event_set:
                for cdm in e:
                    val = cdm[f]
                    if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
                        values.append(val)

            self._features_stats["mean"].append(np.mean(values))
            self._features_stats["stddev"].append(np.std(values))

    def __len__(self):
        return len(self._event_set)

    def __getitem__(self, idx):
        e = self._event_set[idx]
        x = torch.zeros(self._max_event_length, self._features_length)
        for j, cdm in enumerate(e):
            x[j] = torch.tensor([
                ((cdm[f] if cdm[f] is not None else 0.0) - self._features_stats["mean"][k] + 1e-8)
                for k, f in enumerate(self._features)
            ])

            label = 1 if e[-1]["MISS_DISTANCE"] < 100 else 0

            return x, torch.tensor(len(e)), torch.tensor(label).float()

class LSTM(nn.Module):
    def __init__(self, event_set, features, hidden_size = 64, num_layers = 1, dropout = 0.2):
        super(LSTM, self).__init__()
        self._event_set = event_set
        self._features = features
        self._features_length = len(features)
        self.dataset = EventDataset(event_set, features)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size = self._features_length,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        packed = rnn_utils.pack_padded_sequence(x, lengths.cpu, batch_first = True, enforce_sorted = False)
        packed_out, _ = self.lstm(packed)
        out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first = True)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, self.lstm_size)
        final_outputs = out.gather(1, idx).squeeze(1)
        logits = self.fc(final_outputs)
        return self.sigmoid(logits).squeeze(1)

    def learn(self, num_epochs = 10, batch_size = 32, lr = 0.001, val_split = 0.15):
        total_size = len(self.dataset)
        val_size = int(val_split * total_size)
        train_size = total_size - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr = lr)
        criterion = nn.BCELoss()

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0

            for x, lengths, labels in train_loader:
                x = x.to(device)
                lengths = lengths.to(device, dtype=torch.int64)
                labels = labels.to(device)
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
            print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.3f} | Val Accuracy: {accuracy:.3f}")