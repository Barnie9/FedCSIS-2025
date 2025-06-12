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

from cnn_mlp import training_data_path, model_path, scaler_path, device
from cnn_mlp import get_success_prob_columns, fen_to_tensor, chess_additional_features, encode_moves_enhanced
from cnn_mlp import ChessPuzzleCNN

N_ROWS = 1000000
BATCH_SIZE = 256
n_epochs = 1000
EARLY_STOP_PATIENCE = 20

if N_ROWS is not None:
    df = pd.read_csv(training_data_path, nrows=N_ROWS)
else:
    df = pd.read_csv(training_data_path)

success_prob_cols = get_success_prob_columns(df)

fen_tensor_features = np.stack([fen_to_tensor(fen) for fen in df['FEN']])
additional_features = np.array([chess_additional_features(fen) for fen in df['FEN']])
moves_features = np.array([encode_moves_enhanced(m) for m in df['Moves']])
success_prob_features = df[success_prob_cols].values

tabular_features = np.hstack([
    additional_features,
    moves_features,
    success_prob_features
])
y = df['Rating'].values

scaler = StandardScaler()
tabular_features_scaled = scaler.fit_transform(tabular_features)
joblib.dump(scaler, scaler_path)

X_board_train, X_board_val, X_tab_train, X_tab_val, y_train, y_val = train_test_split(
    fen_tensor_features, tabular_features_scaled, y, test_size=0.1, random_state=42
)

torch_X_board_train = torch.tensor(X_board_train, dtype=torch.float32)
torch_X_tab_train = torch.tensor(X_tab_train, dtype=torch.float32)
torch_y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
torch_X_board_val = torch.tensor(X_board_val, dtype=torch.float32)
torch_X_tab_val = torch.tensor(X_tab_val, dtype=torch.float32)
torch_y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(torch_X_board_train, torch_X_tab_train, torch_y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = ChessPuzzleCNN(num_tabular_features=tabular_features.shape[1]).to(device)

criterion = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

best_val_loss = float('inf')
early_stop_counter = 0

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_board, batch_tab, batch_y in train_loader:
        batch_board = batch_board.to(device)
        batch_tab = batch_tab.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_board, batch_tab)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_board.size(0)
    epoch_loss /= len(train_loader.dataset)
    if (epoch+1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_preds = model(torch_X_board_val.to(device), torch_X_tab_val.to(device))
            val_loss = criterion(val_preds, torch_y_val.to(device))
        print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {epoch_loss:.2f} - Val Loss: {val_loss.item():.2f}")
        scheduler.step(val_loss)
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            early_stop_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            early_stop_counter += 1
        if early_stop_counter > EARLY_STOP_PATIENCE:
            print("Early stopping triggered.")
            break

model.load_state_dict(torch.load(model_path))
model.eval()
with torch.no_grad():
    val_preds = model(torch_X_board_val.to(device), torch_X_tab_val.to(device)).cpu().numpy().flatten()
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

torch.save(model.state_dict(), model_path)
print(f"CNN model saved to {model_path}")
print(f"Scaler saved to {scaler_path}")
