import chess
import numpy as np
import tensorflow as tf
from data.processors import MOVE_VOCAB

class ChessBoard:
    def __init__(self, fen=chess.STARTING_FEN):
        self.board = chess.Board(fen)
    
    def to_tensor(self) -> tf.Tensor:
        """Конвертирует доску в тензор формата 8x8x20"""
        tensor = np.zeros((8, 8, 20), dtype=np.float32)
        
        # Каналы 0-11: Фигуры
        piece_channels = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                channel = piece_channels[piece.piece_type]
                if piece.color == chess.BLACK:
                    channel += 6
                rank, file = chess.square_rank(square), chess.square_file(square)
                tensor[rank, file, channel] = 1.0
        
        # Канал 12: Цвет игрока (0 - белые, 1 - черные)
        tensor[:, :, 12] = 1.0 if self.board.turn == chess.BLACK else 0.0
        
        # Каналы 13-16: Рокировки
        castling_rights = [
            self.board.has_kingside_castling_rights(chess.WHITE),
            self.board.has_queenside_castling_rights(chess.WHITE),
            self.board.has_kingside_castling_rights(chess.BLACK),
            self.board.has_queenside_castling_rights(chess.BLACK)
        ]
        tensor[:, :, 13:17] = np.stack([castling_rights]*8*8, axis=0).reshape(8,8,4)
        
        # Канал 17: Взятие на проходе
        if self.board.ep_square:
            ep_rank = chess.square_rank(self.board.ep_square)
            ep_file = chess.square_file(self.board.ep_square)
            tensor[ep_rank, ep_file, 17] = 1.0
        
        # Каналы 18-19: Счетчик ходов (нормализованный)
        tensor[:, :, 18] = self.board.fullmove_number / 100.0
        tensor[:, :, 19] = self.board.halfmove_clock / 100.0
        
        return tf.convert_to_tensor(tensor)
    
    def legal_move_mask(self) -> tf.Tensor:
        """Маска легальных ходов размером 4672"""
        mask = tf.zeros(4672, dtype=tf.float32)
        indices = []
        
        for move in self.board.generate_legal_moves():
            uci = move.uci()
            index = MOVE_VOCAB.get(uci, -1)
            print(f"{uci} {index}")
            if 0 <= index and index < 4672:
                indices.append(index)
       
        updates = tf.ones(len(indices), dtype=tf.float32)
        return tf.tensor_scatter_nd_update(
            mask,
            tf.expand_dims(indices, 1),
            updates
        )
