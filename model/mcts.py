import math
import numpy as np
import chess
from collections import defaultdict
import data.settings
import logging

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
    def __init__(self, model, exploration=1.5, simulations=20):
        self.model = model
        self.exploration = exploration
        self.simulations = simulations
        self.move_map = self._create_move_mapping()
    
    def _create_move_mapping(self):
        """Создает полный mapping всех возможных шахматных ходов (как в AlphaZero)"""
        move_map = {}
        idx = 0
        
        # Все возможные направления для non-promotion и promotion
        for from_sq in chess.SQUARES:
            for to_sq in chess.SQUARES:
                if from_sq == to_sq:
                    continue
                # Non-promotion moves
                move = chess.Move(from_sq, to_sq)
                if move in move_map.values():
                    continue
                move_map[idx] = move
                idx +=1
                
        for i in range(48, 56):
            board = chess.Board()
            board.clear()
            board.set_piece_at(i, chess.Piece(chess.PAWN, chess.WHITE))
            try:
                board.set_piece_at(i-1+8, chess.Piece(chess.ROOK, chess.BLACK))
                board.set_piece_at(i+1+8, chess.Piece(chess.ROOK, chess.BLACK))
            except Exception as e:
                pass
            for move in board.legal_moves:
                move_map[idx] = move
                idx += 1
                
        for i in range(0, 8):
            board = chess.Board()
            board.clear()
            board.set_piece_at(i, chess.Piece(chess.PAWN, chess.BLACK))
            try:
                board.set_piece_at(i-1+8, chess.Piece(chess.ROOK, chess.WHITE))
                board.set_piece_at(i+1+8, chess.Piece(chess.ROOK, chess.WHITE))
            except Exception as e:
                pass
            for move in board.legal_moves:
                move_map[idx] = move
                idx += 1
                
        # print(len(move_map))
        logging.info(f"Loaded {len(move_map)} moves")
        
        return move_map
    
    def _policy_to_moves(self, policy, board):
        """Конвертирует policy vector в словарь легальных ходов"""
        legal_moves = {}
        for idx, prob in enumerate(policy):
            move = self.move_map.get(idx)
            if move and move in board.legal_moves:  # Проверяем легальность
                legal_moves[move] = prob
        # Если нет совпадений, возвращаем случайный легальный ход (fallback)
        if not legal_moves and board.legal_moves.count() > 0:
            for move in board.legal_moves:
                legal_moves[move] = 1.0  # Равномерное распределение
        return legal_moves

    def search(self, board):
        root = MCTSNode()
        
        for _ in range(self.simulations):
            node = root
            current_board = board.copy()
            search_path = [node]
            
            # Selection (без изменений)
            
            # Expansion
            if not current_board.is_game_over():
                policy, value = self.model.predict(current_board)
                legal_moves = self._policy_to_moves(policy, current_board)
                
                # Если legal_moves пуст (крайний случай)
                if not legal_moves:
                    continue  # Пропускаем симуляцию
                
                # Dirichlet noise (без изменений)
                
                for move, prob in legal_moves.items():
                    if move not in node.children:
                        node.children[move] = MCTSNode(node, prob)
                search_path[-1] = node
            
            # Backpropagation и остальное без изменений
        
        # Гарантия выбора, даже если root пуст (fallback)
        if not root.children:
            return next(iter(board.legal_moves))  # Возвращает первый легальный ход
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
    
    def _select_move(self, root, temperature=0.00001):
        visits = [child.visit_count for child in root.children.values()]
        moves = list(root.children.keys())
        
        if np.sum(visits) == 0:
            return np.random.choice(moves)
        
        if temperature == 0:
            best_idx = np.argmax(visits)
            return moves[best_idx]
        else:
            visits = [v ** (1/temperature) for v in visits]
            probs = np.array(visits) / sum(visits)
            try:
                return np.random.choice(moves, p=probs)
            except Exception as e:
                print(moves)
                print(probs)
                print(visits)
                raise e