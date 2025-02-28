import chess
from itertools import product

def create_move_vocab():
    moves = {}
    index = 0
    
    for from_sq in range(64):
        for to_sq in range(64):
            for promo in [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                move = chess.Move(from_sq, to_sq)
                if promo:
                    move.promotion = promo
                moves[move.uci()] = index
                index += 1
    return moves

MOVE_VOCAB = create_move_vocab()

def uci_to_index(uci_move: str) -> int:
    return MOVE_VOCAB[uci_move]
