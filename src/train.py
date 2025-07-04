from kessler.data import kelvins_to_event_dataset
from lstm import *
import torch
import os

def main():
    dataset_path = r"..\data\raw\train_data.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset not found.")

    event_dataset = kelvins_to_event_dataset(dataset_path)

    features = [
        "MISS_DISTANCE",
        "RELATIVE_SPEED",
        "COLLISION_PROBABILITY",
        "OBJECT1_WEIGHTED_RMS",
        "OBJECT2_WEIGHTED_RMS",
        "OBJECT1_OBS_USED",
        "OBJECT2_OBS_USED",
        "RELATIVE_POSITION_R",
        "RELATIVE_POSITION_T",
        "RELATIVE_POSITION_N",
        "RELATIVE_VELOCITY_R",
        "RELATIVE_VELOCITY_T",
        "RELATIVE_VELOCITY_N",
        "__DAYS_TO_TCA"
    ]

    model = LSTM(event_dataset, features)

    print('Training model...')
    model.learn(num_epochs = 10, batch_size = 32, lr = 0.001, val_split = 0.15)

    save_path = r"..\results\models\cdm_lstm.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()

