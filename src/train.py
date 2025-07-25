import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from kessler.data import kelvins_to_event_dataset
from lstm import CollisionRiskLSTM, CDMPreprocessor, create_collision_labels, pad_sequences

DATA_FILE = r'..\data\raw\train_data.csv'
NUM_EVENTS = 2000
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001

print("Loading data...")
events = kelvins_to_event_dataset(DATA_FILE, NUM_EVENTS, remove_outliers=True)

events = [e for e in events if 3 <= len(e) <= 20]
labels = create_collision_labels(events, threshold=1000.0)

train_events, test_events, y_train, y_test = train_test_split(
    events, labels, test_size=0.2, stratify=labels, random_state=42
)

preprocessor = CDMPreprocessor()
X_train = pad_sequences(preprocessor.fit_transform(train_events), 20)
X_test = pad_sequences(preprocessor.transform(test_events), 20)

train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
    batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
    batch_size=BATCH_SIZE
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CollisionRiskLSTM(input_size=X_train.shape[2]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

print(f"Training on {device}...")
for epoch in range(EPOCHS):
    model.train()
    train_loss, train_acc = 0, 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out, _ = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += (out.argmax(1) == y).sum().item()

    model.eval()
    test_acc = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            out, _ = model(X)
            test_acc += (out.argmax(1) == y).sum().item()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}: Train Loss={train_loss / len(train_loader):.3f}, "
            f"Train Acc={100 * train_acc / len(train_loader.dataset):.1f}%, "
            f"Test Acc={100 * test_acc / len(test_loader.dataset):.1f}%")

model.eval()
all_pred, all_true = [], []
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        all_pred.extend(model(X)[0].argmax(1).cpu().numpy())
        all_true.extend(y.numpy())

print("\nEvaluation:")
print(classification_report(all_true, all_pred, target_names=['Low Risk', 'High Risk']))

torch.save(model.state_dict(), f'../results/models/collision_lstm.pth')
preprocessor.save(f'../results/models/preprocessor.pkl')
print(f"Model saved!")