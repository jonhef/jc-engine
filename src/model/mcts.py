import numpy as np
from engine.board import ChessBoard
import tensorflow as tf

class MCTSNode:
    def __init__(self, board, parent=None):
        self.board = board.copy()  # chess.Board, а не ChessBoard
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = 0.0
    
    def is_terminal(self):
        return self.board.is_game_over()
    
    def expanded(self):
        return len(self.children) > 0

class MCTS:
    def __init__(self, model):
        self.model = model
    
    def search(self, root, simulations):
        for _ in range(simulations):
            node = root
            search_path = [node]
            
            # Selection
            while not node.is_terminal() and node.expanded():
                node = max(node.children.values(), key=lambda n: n.ucb_score())
                search_path.append(node)
            
            # Expansion
            if not node.is_terminal():
                policy, value = self.model.predict(node.to_tensor())
                self.expand(node, policy)
                search_path.append(node)
            
            # Backpropagation
            self.backpropagate(search_path, value)
    
    def expand(self, node, policy):
        for move in node.board.legal_moves:
            child_board = node.board.copy()
            child_board.push(move)
            node.children[move.uci()] = MCTSNode(child_board, node)
    
    def backpropagate(self, path, value):
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value
