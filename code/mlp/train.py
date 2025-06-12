import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from torch.utils.data import DataLoader, TensorDataset

from mlp import training_data_path, model_path, scaler_path, plot_path
from mlp import get_success_prob_columns, encode_fen, encode_moves, get_model

N_ROWS = 10000

if N_ROWS is not None:
    df = pd.read_csv(training_data_path, nrows=N_ROWS)
else:
    df = pd.read_csv(training_data_path)

success_prob_cols = get_success_prob_columns(df)

fen_features = np.array([encode_fen(fen) for fen in df['FEN']])
moves_features = np.array([encode_moves(m) for m in df['Moves']])
success_prob_features = df[success_prob_cols].values

X = np.hstack([fen_features, moves_features, success_prob_features])
y = df['Rating'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, scaler_path)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

BATCH_SIZE = 1024
torch_X_train = torch.tensor(X_train, dtype=torch.float32)
torch_y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
torch_X_val = torch.tensor(X_val, dtype=torch.float32)
torch_y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
train_dataset = TensorDataset(torch_X_train, torch_y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = get_model(X.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 500
best_val_loss = float('inf')
early_stop_counter = 0
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_X.size(0)
    epoch_loss /= len(train_loader.dataset)
    if (epoch+1) % 10 == 0:
        model.eval()
        val_output = model(torch_X_val)
        val_loss = criterion(val_output, torch_y_val)
        print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {epoch_loss:.2f} - Val Loss: {val_loss.item():.2f}")
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            early_stop_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            early_stop_counter += 1
        if early_stop_counter > 10:
            print("Early stopping triggered.")
            break

model.load_state_dict(torch.load(model_path))

model.eval()
val_preds = model(torch_X_val).detach().numpy().flatten()
val_true = y_val
mae = mean_absolute_error(val_true, val_preds)
mse = mean_squared_error(val_true, val_preds)
r2 = r2_score(val_true, val_preds)
print(f"Validation MAE: {mae:.2f}")
print(f"Validation MSE: {mse:.2f}")
print(f"Validation R^2: {r2:.4f}")

import matplotlib.pyplot as plt
plt.figure(figsize=(7,5))
plt.scatter(val_true, val_preds, alpha=0.5)
plt.xlabel("True Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Predicted vs. True Ratings")
min_val = min(val_true.min(), val_preds.min())
max_val = max(val_true.max(), val_preds.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
plt.legend()
plt.show()
plt.savefig(plot_path)

# Save model
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
print(f"Scaler saved to {scaler_path}")
