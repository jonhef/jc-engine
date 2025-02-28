import chess
from data.processors import uci_to_index, MOVE_VOCAB

def is_move_legal(board: chess.Board, uci_move: str) -> bool:
    """Проверяет, является ли ход легальным"""
    try:
        move = chess.Move.from_uci(uci_move)
        return move in board.legal_moves
    except:
        return False

def index_to_uci(index: int) -> str:
    """Преобразует индекс обратно в UCI-строку"""
    for uci, idx in MOVE_VOCAB.items():
        if idx == index:
            return uci
    return ""

def get_all_possible_moves():
    """Возвращает список всех возможных UCI-ходов"""
    return list(MOVE_VOCAB.keys())
