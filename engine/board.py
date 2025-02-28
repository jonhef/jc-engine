import chess
import tensorflow as tf
from data.converters import fen_to_tensor
from data.processors import uci_to_index
from config.settings import Config

class ChessBoard:
    """Класс для работы с шахматной доской и преобразований в тензоры"""
    
    def __init__(self, fen: str = None):
        self.board = chess.Board(fen) if fen else chess.Board()
    
    def legal_move_mask(self) -> tf.Tensor:
        """Создает маску допустимых ходов"""
        mask = tf.zeros([Config.OUTPUT_SIZE], dtype=tf.float32)
        for move in self.board.legal_moves:
            idx = uci_to_index(move.uci())
            if idx != -1:  # Игнорируем ходы вне словаря
                mask = tf.tensor_scatter_nd_update(
                    mask, 
                    [[idx]], 
                    [1.0]
                )
        return mask
    
    def to_tensor(self) -> tf.Tensor:
        """Конвертирует текущую позицию в тензор"""
        return fen_to_tensor(self.board.fen())
    
    def verify_tensor(self, tensor: tf.Tensor) -> bool:
        """Проверяет корректность преобразования FEN → Tensor → FEN"""
        reconstructed_fen = self._tensor_to_fen(tensor)
        return reconstructed_fen == self.board.fen()
    
    def _tensor_to_fen(self, tensor: tf.Tensor) -> str:
        """Восстанавливает FEN из тензора (для тестов)"""
        # Реализация опущена для краткости
        pass
