import unittest
from engine.board import ChessBoard
import tensorflow as tf
import chess

class TestMoveMasking(unittest.TestCase):
    def test_initial_position_mask(self):
        board = ChessBoard()
        mask = board.legal_move_mask()
        self.assertEqual(tf.reduce_sum(mask).numpy(), 20)

if __name__ == '__main__':
    unittest.main()
