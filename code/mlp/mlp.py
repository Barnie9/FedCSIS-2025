import torch.nn as nn
import chess
import os

training_data_path = os.path.join('dataset', 'training_data.csv')
testing_data_path = os.path.join('dataset', 'testing_data.csv')
model_path = os.path.join('models', 'mlp_chess_rating_model.pt')
scaler_path = os.path.join('scalers', 'mlp_feature_scaler.save')
submission_path = os.path.join('submissions', 'mlp_submission.txt')
plot_path = os.path.join('plots', 'mlp_plot.png')

def get_success_prob_columns(df):
    return [col for col in df.columns if col.startswith('success_prob_')]

def encode_fen(fen):
    board = chess.Board(fen)
    piece_map = [
        (chess.PAWN, chess.BLACK), (chess.KNIGHT, chess.BLACK), (chess.BISHOP, chess.BLACK),
        (chess.ROOK, chess.BLACK), (chess.QUEEN, chess.BLACK), (chess.KING, chess.BLACK),
        (chess.PAWN, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.BISHOP, chess.WHITE),
        (chess.ROOK, chess.WHITE), (chess.QUEEN, chess.WHITE), (chess.KING, chess.WHITE),
    ]
    counts = [len(board.pieces(pt, color)) for pt, color in piece_map]
    turn = [1 if board.turn == chess.WHITE else 0]
    castling = [int(board.has_kingside_castling_rights(chess.WHITE)),
                int(board.has_queenside_castling_rights(chess.WHITE)),
                int(board.has_kingside_castling_rights(chess.BLACK)),
                int(board.has_queenside_castling_rights(chess.BLACK))]
    en_passant = [0 if board.ep_square is None else 1]
    return counts + turn + castling + en_passant

def encode_moves(moves_str):
    moves = moves_str.split()
    num_moves = len(moves)
    max_move_len = max((len(m) for m in moves), default=0)
    has_promotion = int(any('=' in m for m in moves))
    return [num_moves, max_move_len, has_promotion]

def get_model(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )