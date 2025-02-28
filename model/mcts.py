import numpy as np
import chess
from engine.board import ChessBoard
from config.settings import Config

class MCTSNode:
    def __init__(self, board, parent=None):
        self.board = board.copy()
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = 0.0
    
    def is_terminal(self):
        return self.board.is_game_over()
    
    def ucb_score(self):
        if self.visit_count == 0:
            return float('inf')
        return (self.value_sum / self.visit_count) + Config.C_PUCT * self.prior * np.sqrt(self.parent.visit_count) / (1 + self.visit_count)

class MCTS:
    def __init__(self, model):
        self.model = model
    
    def search(self, root, simulations):
        for _ in range(simulations):
            node = root
            while not node.is_terminal() and node.children:
                node = max(node.children.values(), key=lambda n: n.ucb_score())
            
            if not node.is_terminal():
                policy, value = self.model.predict(node.board.to_tensor())
                self.expand(node, policy.numpy())
            
            self.backpropagate(node, value.numpy()[0])
        return root
    
    def expand(self, node, policy):
        legal_moves = [move.uci() for move in node.board.legal_moves]
        for move in legal_moves:
            new_board = node.board.copy()
            new_board.push_uci(move)
            node.children[move] = MCTSNode(new_board, node)
    
    def backpropagate(self, node, value):
        while node:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent
            value = -value
