import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.0, activation='relu'):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.activation(out)
        out = self.fc(out)
        return out
    

def prepare_data(train_csv, test_csv, target_col='Activity'):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    le = LabelEncoder()
    train_df[target_col] = le.fit_transform(train_df[target_col])
    test_df[target_col] = le.transform(test_df[target_col])

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values
    X_val = test_df.drop(columns=[target_col])
    y_val = test_df[target_col].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    return X_train, y_train, X_val, y_val, le


def train_model(X_train, y_train, X_val, y_val, hidden_size=64, batch_size=64, lr=0.001, epochs=100, dropout=0.0, activation="relu", device='cpu', mlflow_logging=False):
    model = LSTMModel(input_size=X_train.shape[2], hidden_size=hidden_size, num_classes = len(torch.unique(torch.cat([y_train, y_val]))), dropout=dropout, activation=activation)
    model.to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_model_state = model.state_dict()
    best_labels = []
    best_preds = []
    patience = 20
    counter = 0

    final_val_loss = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        correct, total = 0, 0  # WICHTIG: total initialisieren
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss_batch = criterion(preds, yb).item()
                val_loss += loss_batch * yb.size(0)

                preds_labels = preds.argmax(dim=1)
                correct += (preds_labels == yb).sum().item()
                total += yb.size(0)  # WICHTIG: total aktualisieren!

                all_preds.extend(preds_labels.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total if total > 0 else 0

        if mlflow_logging:
            mlflow.log_metric("epoch_val_accuracy", val_acc, step=epoch)
            mlflow.log_metric("epoch_val_loss", val_loss, step=epoch)


        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            final_val_loss = val_loss
            best_model_state = model.state_dict()
            best_preds = all_preds
            best_labels = all_labels
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    # Sicherstellen, dass wir keine leeren Listen haben
    if not best_preds or not best_labels:
        best_preds = all_preds
        best_labels = all_labels
        final_val_loss = val_loss


    # Bestes Modell laden
    model.load_state_dict(best_model_state)

    precision = precision_score(best_labels, best_preds, average='weighted', zero_division=0)
    recall = recall_score(best_labels, best_preds, average='weighted', zero_division=0)

    return best_val_acc, final_val_loss, precision, recall, model
