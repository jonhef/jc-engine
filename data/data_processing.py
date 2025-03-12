import chess
import numpy as np
import torch
from torch.utils.data import Dataset
import csv

def fen_to_tensor(fen):
    board = chess.Board(fen)
    # Изменяем numpy на torch
    tensor = torch.zeros((14, 8, 8), dtype=torch.float32)
    
    tensor = tensor / 10.0
    
    piece_to_idx = {
        'P': 0, 'R': 1, 'N': 2, 'B': 3, 'Q': 4, 'K': 5,
        'p': 6, 'r': 7, 'n': 8, 'b': 9, 'q': 10, 'k': 11
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = 7 - (square // 8), square % 8
            tensor[piece_to_idx[piece.symbol()], row, col] = 1.0
    
    # Добавляем дополнительные признаки
    tensor[12] = float(board.turn)
    tensor[13] = float(board.has_legal_en_passant())
    
    return tensor  # Теперь возвращает torch.Tensor

def move_to_indices(move_uci):
    move = chess.Move.from_uci(move_uci)
    return (move.from_square, move.to_square)

class ChessDataset(Dataset):
    def __init__(self, filename):
        self.data = []
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) == 2:
                    self.data.append((row[0], row[1]))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        fen, move = self.data[idx]
        x = fen_to_tensor(fen)
        from_sq, to_sq = move_to_indices(move)
        return (
            torch.tensor(x, dtype=torch.float32),
            (torch.tensor(from_sq, dtype=torch.long),
             torch.tensor(to_sq, dtype=torch.long))
        )