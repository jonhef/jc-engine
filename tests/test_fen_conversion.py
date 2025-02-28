import unittest
from engine.board import ChessBoard
from data.converters import fen_to_tensor
import chess

class TestFENConversion(unittest.TestCase):
    def test_startpos_conversion(self):
        board = ChessBoard()
        tensor = board.to_tensor()
        
        # Проверка наличия белых фигур
        self.assertEqual(tensor[7, 0, 3].numpy(), 1.0)  # Ладья
        self.assertEqual(tensor[7, 1, 1].numpy(), 1.0)  # Конь
        
        # Проверка дополнительных каналов
        self.assertEqual(tensor[0, 0, 13].numpy(), 1.0)  # Рокировка

if __name__ == '__main__':
    unittest.main()
