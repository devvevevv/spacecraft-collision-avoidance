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
        '__CREATION_DATE', '__TCA', 'MISS_DISTANCE',
        'RELATIVE_SPEED', 'RELATIVE_POSITION_R', 'RELATIVE_POSITION_T',
        'RELATIVE_POSITION_N', 'RELATIVE_VELOCITY_R', 'RELATIVE_VELOCITY_T',
        'RELATIVE_VELOCITY_N', 'OBJECT1_X', 'OBJECT1_Y',
        'OBJECT1_Z', 'OBJECT1_X_DOT', 'OBJECT1_Y_DOT',
        'OBJECT1_Z_DOT', 'OBJECT1_CR_R', 'OBJECT1_CT_R',
        'OBJECT1_CT_T', 'OBJECT1_CN_R', 'OBJECT1_CN_T',
        'OBJECT1_CN_N', 'OBJECT1_CRDOT_R', 'OBJECT1_CRDOT_T',
        'OBJECT1_CRDOT_N', 'OBJECT1_CRDOT_RDOT', 'OBJECT1_CTDOT_R',
        'OBJECT1_CTDOT_T', 'OBJECT1_CTDOT_N', 'OBJECT1_CTDOT_RDOT',
        'OBJECT1_CTDOT_TDOT', 'OBJECT1_CNDOT_R', 'OBJECT1_CNDOT_T',
        'OBJECT1_CNDOT_N', 'OBJECT1_CNDOT_RDOT', 'OBJECT1_CNDOT_TDOT',
        'OBJECT1_CNDOT_NDOT', 'OBJECT2_X', 'OBJECT2_Y',
        'OBJECT2_Z', 'OBJECT2_X_DOT', 'OBJECT2_Y_DOT',
        'OBJECT2_Z_DOT', 'OBJECT2_CR_R', 'OBJECT2_CT_R',
        'OBJECT2_CT_T', 'OBJECT2_CN_R', 'OBJECT2_CN_T',
        'OBJECT2_CN_N', 'OBJECT2_CRDOT_R', 'OBJECT2_CRDOT_T',
        'OBJECT2_CRDOT_N', 'OBJECT2_CRDOT_RDOT', 'OBJECT2_CTDOT_R',
        'OBJECT2_CTDOT_T', 'OBJECT2_CTDOT_N', 'OBJECT2_CTDOT_RDOT',
        'OBJECT2_CTDOT_TDOT', 'OBJECT2_CNDOT_R', 'OBJECT2_CNDOT_T',
        'OBJECT2_CNDOT_N', 'OBJECT2_CNDOT_RDOT', 'OBJECT2_CNDOT_TDOT',
        'OBJECT2_CNDOT_NDOT'
    ]

    model = LSTM(event_dataset, features)

    print('Training model...')
    model.learn(epochs = 10, batch_size = 32, lr = 0.001, val_split = 0.15)

    save_path = r"..\results\models\cdm_lstm.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()

