import chess
import numpy as np
import tensorflow as tf
from config.settings import Config

def fen_to_tensor(fen: str) -> tf.Tensor:
    board = chess.Board(fen)
    tensor = np.zeros(Config.INPUT_SHAPE, dtype=np.float32)
    
    piece_channels = {
        (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)  # Инверсия строк для FEN
            col = square % 8
            channel = piece_channels[(piece.piece_type, piece.color)]
            tensor[row, col, channel] = 1.0

    # Очередь хода (канал 12)
    turn_channel = 12
    tensor[:, :, turn_channel] = 1.0 if board.turn else 0.0
    
    # Рокировки (каналы 13-16)
    tensor[:, :, 13] = board.has_kingside_castling_rights(chess.WHITE)
    tensor[:, :, 14] = board.has_queenside_castling_rights(chess.WHITE)
    tensor[:, :, 15] = board.has_kingside_castling_rights(chess.BLACK)
    tensor[:, :, 16] = board.has_queenside_castling_rights(chess.BLACK)
    
    # Взятие на проходе (канал 17)
    if board.ep_square:
        ep_row = 7 - (board.ep_square // 8)
        ep_col = board.ep_square % 8
        tensor[ep_row, ep_col, 17] = 1.0
    
    return tf.convert_to_tensor(tensor)
