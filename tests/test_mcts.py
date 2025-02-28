import unittest
from model.mcts import MCTS, MCTSNode
from engine.board import ChessBoard
from model.network import ChessModel
import chess

class TestMCTS(unittest.TestCase):
    def setUp(self):
        self.model = ChessModel()
        self.mcts = MCTS(self.model)

    def test_node_creation(self):
        board = ChessBoard()
        node = MCTSNode(board.board)
        self.assertFalse(node.is_terminal())

    def test_simulation(self):
        board = ChessBoard()
        root = MCTSNode(board.board)
        self.mcts.search(root, 10)
        self.assertGreater(root.visit_count, 0)

if __name__ == "__main__":
    unittest.main()
