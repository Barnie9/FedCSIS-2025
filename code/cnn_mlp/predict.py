import pandas as pd
import numpy as np
import torch
import joblib

from cnn_mlp import testing_data_path, model_path, scaler_path, submission_path, device
from cnn_mlp import get_success_prob_columns, fen_to_tensor, chess_additional_features, encode_moves_enhanced
from cnn_mlp import ChessPuzzleCNN

df = pd.read_csv(testing_data_path)

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

scaler = joblib.load(scaler_path)
tabular_features_scaled = scaler.transform(tabular_features)

model = ChessPuzzleCNN(num_tabular_features=tabular_features.shape[1])
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

with torch.no_grad():
    inputs_board = torch.tensor(fen_tensor_features, dtype=torch.float32).to(device)
    inputs_tab = torch.tensor(tabular_features_scaled, dtype=torch.float32).to(device)
    preds = model(inputs_board, inputs_tab).cpu().numpy().flatten()

submission = np.round(preds).astype(int)
assert len(submission) == 2235, f"Submission must have 2235 lines, got {len(submission)}"
with open(submission_path, 'w') as f:
    for val in submission:
        f.write(f"{val}\n")
print(f"Submission file '{submission_path}' created with 2235 predictions.")
