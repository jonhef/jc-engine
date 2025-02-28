import unittest
import chess
import tensorflow as tf
from engine.board import ChessBoard
from data.converters import fen_to_tensor
from data.processors import uci_to_index

class TestConverters(unittest.TestCase):
    """Тесты для проверки конвертации FEN ↔ Tensor и валидации ходов"""
    
    def test_start_position_conversion(self):
        """Проверка начальной позиции"""
        board = ChessBoard()
        tensor = board.to_tensor()
        
        # Проверка белых фигур
        self.assertEqual(tensor[7, 0, 3].numpy(), 1.0)   # Белая ладья на a1
        self.assertEqual(tensor[7, 1, 1].numpy(), 1.0)   # Белый конь на b1
        self.assertEqual(tensor[6, 0, 0].numpy(), 1.0)   # Белая пешка на a2
        
        # Проверка черных фигур
        self.assertEqual(tensor[0, 0, 9].numpy(), 1.0)   # Черная ладья на a8
        self.assertEqual(tensor[0, 1, 7].numpy(), 1.0)   # Черный конь на b8
        
        # Дополнительные каналы
        self.assertEqual(tensor[0, 0, 13].numpy(), 1.0)  # Рокировка белых
        
    def test_legal_moves_startpos(self):
        """Проверка маски ходов в начальной позиции"""
        board = ChessBoard()
        mask = board.legal_move_mask()
        self.assertEqual(tf.reduce_sum(mask).numpy(), 20)  # 20 возможных ходов
        
    def test_promotion_position(self):
        """Тест позиции с пешкой на предпоследней горизонтали"""
        board = ChessBoard("8/P7/8/8/8/8/8/8 w - - 0 1")
        mask = board.legal_move_mask()
        
        # Пешка может превратиться в 4 фигуры + обычный ход
        self.assertEqual(tf.reduce_sum(mask).numpy(), 5)
        
    def test_uci_conversion(self):
        """Проверка преобразования UCI → Index"""
        self.assertGreater(uci_to_index("e2e4"), 0)
        self.assertGreater(uci_to_index("a7a8q"), 0)  # Промоушен
        self.assertEqual(uci_to_index("invalid"), -1)

    def test_en_passant(self):
        """Тест взятия на проходе"""
        board = ChessBoard("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1")
        tensor = board.to_tensor()
        
        # Канал взятия на проходе (17)
        self.assertEqual(tensor[4, 4, 17].numpy(), 1.0)  # e3

if __name__ == "__main__":
    unittest.main()
