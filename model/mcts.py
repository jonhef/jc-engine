import math
import numpy as np
import chess
from collections import defaultdict

class MCTSNode:
    def __init__(self, parent=None, prior=0):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior  # Prior probability from neural network
    
    def expanded(self):
        return len(self.children) > 0
    
    def value(self):
        return self.total_value / (1 + self.visit_count) if self.visit_count > 0 else 0
    
    def ucb_score(self, exploration=1.5):
        if self.parent is None:
            return 0.0
        return self.value() + exploration * self.prior * math.sqrt(self.parent.visit_count) / (self.visit_count + 1)

class MCTS:
    def __init__(self, model, exploration=1.5, simulations=200):
        self.model = model
        self.exploration = exploration
        self.simulations = simulations
        self.move_map = self._create_move_mapping()
    
    def _create_move_mapping(self):
        """Create mapping from move index to chess.Move object"""
        move_map = {}
        idx = 0
        board = chess.Board()
        for move in board.legal_moves:
            move_map[idx] = move
            idx += 1
        return move_map
    
    def _policy_to_moves(self, policy, board):
        """Convert policy vector to legal moves dictionary"""
        legal_moves = {}
        for idx, prob in enumerate(policy):
            move = self.move_map.get(idx)
            if move and move in board.legal_moves:
                legal_moves[move] = prob
        return legal_moves
    
    def search(self, board):
        root = MCTSNode()
        
        for _ in range(self.simulations):
            node = root
            current_board = board.copy()
            search_path = [node]
            
            # Selection
            while node.expanded():
                moves, nodes = zip(*node.children.items())
                scores = [n.ucb_score(self.exploration) for n in nodes]
                best_idx = np.argmax(scores)
                move = moves[best_idx]
                node = nodes[best_idx]
                search_path.append(node)
                current_board.push(move)
            
            # Expansion
            if not current_board.is_game_over():
                policy, value = self.model.predict(current_board)
                legal_moves = self._policy_to_moves(policy, current_board)
                
                # Add Dirichlet noise for root node
                if node.parent is None:
                    dirichlet_noise = np.random.dirichlet([0.3]*len(legal_moves))
                    for i, (move, prob) in enumerate(legal_moves.items()):
                        legal_moves[move] = 0.75*prob + 0.25*dirichlet_noise[i]
                
                for move, prob in legal_moves.items():
                    node.children[move] = MCTSNode(node, prob)
                search_path[-1] = node
            
            # Backpropagation
            value = self._evaluate(current_board) if current_board.is_game_over() else value
            self._backpropagate(search_path, value)
        
        return self._select_move(root)
    
    def _evaluate(self, board):
        """Evaluate terminal state"""
        if board.is_checkmate():
            return 1.0 if board.turn != chess.WHITE else -1.0
        return 0.0  # Draw
    
    def _backpropagate(self, path, value):
        for node in reversed(path):
            node.visit_count += 1
            node.total_value += value
            value = -value  # Alternate perspective for players
    
    def _select_move(self, root, temperature=1.0):
        visits = [child.visit_count for child in root.children.values()]
        moves = list(root.children.keys())
        
        if temperature == 0:
            best_idx = np.argmax(visits)
            return moves[best_idx]
        else:
            visits = [v ** (1/temperature) for v in visits]
            probs = np.array(visits) / sum(visits)
            return np.random.choice(moves, p=probs)