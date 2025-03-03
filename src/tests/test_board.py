import unittest
from engine.board import ChessBoard
import chess

class TestBoard(unittest.TestCase):
    def test_initial_position(self):
        board = ChessBoard()
        self.assertEqual(board.board.fen(), chess.STARTING_FEN)
    
    def test_move_generation(self):
        board = ChessBoard()
        self.assertEqual(len(list(board.board.legal_moves)), 20)

if __name__ == "__main__":
    unittest.main()
