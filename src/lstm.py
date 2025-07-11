import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import joblib

class CollisionRiskLSTM(nn.Module):
    def __init__(self, input_size = 12, hidden_size = 64, num_layers = 2, dropout = 0.2):
        super(CollisionRiskLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            dropout = dropout if num_layers > 1 else 0,
            batch_first = True
        )

        self.attention = nn.Linear(hidden_size, 1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        batch_size = x.size(0)

        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))

        attention_weights = torch.softmax(self.attention(lstm_out), dim = 1)
        context_vector = torch.sum(attention_weights * lstm_out, dim = 1)

        output = self.classifier(context_vector)
        return output, attention_weights

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
            raise RuntimeError("Preprocessor must be fitted before transform")

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

def create_collision_labels(events, threshold = 1000.0): #TODO: change threshold according to eval metrics
    labels = []

    for event in events:
        min_miss_distance = float("inf")

        for cdm in event:
            miss_distance = cdm.get("MISS_DISTANCE")
            if miss_distance is not None:
                min_miss_distance = min(min_miss_distance, float(miss_distance))

        if min_miss_distance < threshold:
            labels.append(1)
        else:
            labels.append(0)

    return labels


def pad_sequences(sequences, max_length=None):
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    padded = []
    for seq in sequences:
        if len(seq) >= max_length:
            padded.append(seq[-max_length:])
        else:
            padding = np.zeros((max_length - len(seq), seq.shape[1]))
            padded_seq = np.vstack([padding, seq])
            padded.append(padded_seq)

    return np.array(padded)