import pandas as pd
import numpy as np
import torch
import joblib

from mlp import testing_data_path, model_path, scaler_path, submission_path
from mlp import get_success_prob_columns, encode_fen, encode_moves, get_model

df = pd.read_csv(testing_data_path)

success_prob_cols = get_success_prob_columns(df)

fen_features = np.array([encode_fen(fen) for fen in df['FEN']])
moves_features = np.array([encode_moves(m) for m in df['Moves']])
success_prob_features = df[success_prob_cols].values

X = np.hstack([fen_features, moves_features, success_prob_features])

scaler = joblib.load(scaler_path)
X_scaled = scaler.transform(X)

model = get_model(X.shape[1])
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

with torch.no_grad():
    inputs = torch.tensor(X_scaled, dtype=torch.float32)
    preds = model(inputs).numpy().flatten()

submission = np.round(preds).astype(int)
assert len(submission) == 2235, f"Submission must have 2235 lines, got {len(submission)}"
with open(submission_path, 'w') as f:
    for val in submission:
        f.write(f"{val}\n")
print(f"Submission file '{submission_path}' created with 2235 predictions.")
