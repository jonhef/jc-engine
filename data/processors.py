import chess
from itertools import product

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
