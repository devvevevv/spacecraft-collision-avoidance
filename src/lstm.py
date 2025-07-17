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

class CAM:
    @staticmethod
    def generate_cam(cdm, delta_v_mag=1.0):
        new_cdm = cdm.copy()

        cur_pos_r = cdm.get("RELATIVE_POSITION_R", 0.0)
        cur_pos_t = cdm.get("RELATIVE_POSITION_T", 0.0)
        cur_pos_n = cdm.get("RELATIVE_POSITION_N", 0.0)
        cur_vel_r = cdm.get("RELATIVE_VELOCITY_R", 0.0)
        cur_vel_t = cdm.get("RELATIVE_VELOCITY_T", 0.0)
        cur_vel_n = cdm.get("RELATIVE_VELOCITY_N", 0.0)
        cur_miss_dist = cdm.get("MISS_DISTANCE", 0.0)

        pos_mag = np.sqrt(cur_pos_r ** 2 + cur_pos_t ** 2 + cur_pos_n ** 2)

        if pos_mag < 1e-6:
            pos_mag = 1e-6

        pos_r_norm = cur_pos_r / pos_mag
        pos_t_norm = cur_pos_t / pos_mag
        pos_n_norm = cur_pos_n / pos_mag

        if abs(pos_r_norm) >= abs(pos_t_norm) and abs(pos_r_norm) >= abs(pos_n_norm):
            delta_v_r = 0.0
            delta_v_t = delta_v_mag if pos_t_norm >= 0 else -delta_v_mag
            delta_v_n = delta_v_mag * 0.3 if pos_n_norm >= 0 else -delta_v_mag * 0.3

        elif abs(pos_t_norm) >= abs(pos_n_norm):
            delta_v_r = delta_v_mag if pos_r_norm >= 0 else -delta_v_mag
            delta_v_t = 0.0
            delta_v_n = delta_v_mag * 0.3 if pos_n_norm >= 0 else -delta_v_mag * 0.3

        else:
            delta_v_r = delta_v_mag * 0.7 if pos_r_norm >= 0 else -delta_v_mag * 0.7
            delta_v_t = delta_v_mag * 0.7 if pos_t_norm >= 0 else -delta_v_mag * 0.7
            delta_v_n = 0.0

        rel_vel_mag = np.sqrt(cur_vel_r ** 2 + cur_vel_t ** 2 + cur_vel_n ** 2)
        if rel_vel_mag > 1e-6:
            time_to_ca = max(pos_mag / rel_vel_mag, 60.0)
            time_to_ca = min(time_to_ca, 7200.0)
        else:
            time_to_ca = 3600.0

        def update_state(sign=1):
            new_vel = {
                'r': cur_vel_r + sign * delta_v_r,
                't': cur_vel_t + sign * delta_v_t,
                'n': cur_vel_n + sign * delta_v_n,
            }
            new_speed = np.sqrt(new_vel['r'] ** 2 + new_vel['t'] ** 2 + new_vel['n'] ** 2)

            dt_hours = time_to_ca / 3600.0

            new_pos = {
                'r': cur_pos_r + new_vel['r'] * dt_hours,
                't': cur_pos_t + new_vel['t'] * dt_hours,
                'n': cur_pos_n + new_vel['n'] * dt_hours
            }

            new_miss_dist = np.sqrt(new_pos['r'] ** 2 + new_pos['t'] ** 2 + new_pos['n'] ** 2)
            return new_vel, new_speed, new_pos, new_miss_dist

        new_vel, new_speed, new_pos, new_miss_dist = update_state(sign=1)

        if new_miss_dist <= cur_miss_dist:
            new_vel_neg, new_speed_neg, new_pos_neg, new_miss_dist_neg = update_state(sign=-1)

            if new_miss_dist_neg > new_miss_dist:
                new_vel, new_speed, new_pos, new_miss_dist = new_vel_neg, new_speed_neg, new_pos_neg, new_miss_dist_neg

        if new_miss_dist < 500.0:
            safety_factor = 1000.0 / max(new_miss_dist, 1.0)
            new_miss_dist = max(new_miss_dist * safety_factor, 1000.0)

            if new_pos:
                pos_scale = new_miss_dist / np.sqrt(new_pos['r'] ** 2 + new_pos['t'] ** 2 + new_pos['n'] ** 2)
                new_pos['r'] *= pos_scale
                new_pos['t'] *= pos_scale
                new_pos['n'] *= pos_scale

        new_cdm.update({
            'RELATIVE_VELOCITY_R': new_vel['r'],
            'RELATIVE_VELOCITY_T': new_vel['t'],
            'RELATIVE_VELOCITY_N': new_vel['n'],
            'RELATIVE_SPEED': new_speed,
            'MISS_DISTANCE': new_miss_dist,
        })

        if new_pos:
            new_cdm.update({
                'RELATIVE_POSITION_R': new_pos['r'],
                'RELATIVE_POSITION_T': new_pos['t'],
                'RELATIVE_POSITION_N': new_pos['n'],
            })

        return new_cdm