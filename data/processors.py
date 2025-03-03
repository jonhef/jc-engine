import chess
from itertools import product

from chess import Move
from typing import Dict
import numpy as np

class MoveConverter:
    def __init__(self):
        self.move_vocab = self.create_move_vocabulary()
        self.reverse_vocab = {v: k for k, v in self.move_vocab.items()}

    def san_to_uci(self, san: str, board: chess.Board) -> str:
        try:
            move = board.parse_san(san)
            return move.uci()
        except:
            return None
        
    def create_move_vocabulary(self) -> Dict[str, int]:
        return create_move_vocabulary()

    def convert_san_sequence(self, sans: "List[str]") -> np.ndarray:
        board = chess.Board()
        uci_moves = []
        
        for san in sans:
            uci = self.san_to_uci(san, board)
            if uci:
                uci_moves.append(uci)
                board.push_uci(uci)
        
        return np.array([self.move_vocab.get(m, -1) for m in uci_moves])

def create_move_vocabulary():
    moves = {}
    index = 0
    
    # Все возможные комбинации from_sq (0-63) и to_sq (0-63)
    for from_sq, to_sq in product(range(64), repeat=2):
        from_rank = chess.square_rank(from_sq)
        to_rank = chess.square_rank(to_sq)
        
        # Определение необходимости промоушена
        promotions = []
        if (from_rank == 6 and to_rank == 7) or (from_rank == 1 and to_rank == 0):   
            promotions = [
                chess.QUEEN, chess.ROOK, 
                chess.BISHOP, chess.KNIGHT
            ]
        
        # Добавление хода без промоушена (если не промоушен)
        if not promotions:
            move = chess.Move(from_sq, to_sq)
            moves[move.uci()] = index
            index += 1
        else:
            # Добавление всех вариантов с промоушенами
            for promo in promotions:
                move = chess.Move(from_sq, to_sq, promotion=promo)
                moves[move.uci()] = index
                index += 1
    
    return moves

MOVE_VOCAB = create_move_vocabulary()

def uci_to_index(key: str):
    return MOVE_VOCAB.get(key, -1)
