import os
import torch
import numpy as np
import chess
import torch.nn as nn

training_data_path = os.path.join('dataset', 'training_data.csv')
testing_data_path = os.path.join('dataset', 'testing_data.csv')
model_path = os.path.join('models', 'cnn_mlp_chess_rating_model.pt')
scaler_path = os.path.join('scalers', 'cnn_mlp_feature_scaler.save')
submission_path = os.path.join('submissions', 'cnn_mlp_chess_rating_submission.csv')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_success_prob_columns(df):
    return [col for col in df.columns if col.startswith('success_prob_')]

def fen_to_tensor(fen):
    board = chess.Board(fen)
    tensor = np.zeros((13, 8, 8), dtype=np.float32)
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            plane = piece_map[piece.symbol()]
            row = chess.square_rank(square)
            col = chess.square_file(square)
            tensor[plane][row][col] = 1
    tensor[12][:,:] = int(board.turn)
    return tensor

def chess_additional_features(fen):
    board = chess.Board(fen)
    piece_vals = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3.1,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }
    material_white = sum(len(board.pieces(pt, chess.WHITE)) * v for pt, v in piece_vals.items())
    material_black = sum(len(board.pieces(pt, chess.BLACK)) * v for pt, v in piece_vals.items())
    material_diff = material_white - material_black
    board_turn = board.turn
    board.turn = chess.WHITE
    white_mobility = len(list(board.legal_moves))
    board.turn = chess.BLACK
    black_mobility = len(list(board.legal_moves))
    board.turn = board_turn  # restore
    total_material = material_white + material_black
    if total_material > 35:
        phase = 0
    elif total_material > 20:
        phase = 1
    else:
        phase = 2
    return [material_diff, white_mobility, black_mobility, phase]

def encode_moves_enhanced(moves_str):
    moves = moves_str.split()
    num_moves = len(moves)
    has_promotion = int(any('=' in m for m in moves))
    num_captures = sum('x' in m for m in moves)
    num_checks = sum('+' in m for m in moves)
    max_move_len = max((len(m) for m in moves), default=0)
    return [num_moves, max_move_len, has_promotion, num_captures, num_checks]

class ChessPuzzleCNN(nn.Module):
    def __init__(self, num_tabular_features):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(13, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.cnn_out_dim = 64 * 2 * 2
        self.tabular_mlp = nn.Sequential(
            nn.Linear(num_tabular_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(self.cnn_out_dim + 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, board, tabular):
        x = self.cnn(board)
        x = x.view(x.size(0), -1)
        t = self.tabular_mlp(tabular)
        out = torch.cat([x, t], dim=1)
        out = self.head(out)
        return out