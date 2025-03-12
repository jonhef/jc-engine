import chess
from chess import svg
import torch
from model.model import ChessNet
from data.data_processing import fen_to_tensor
from model.predict import ChessEngine

class ChessGame:
    def __init__(self, model_path='chess_model.pth'):
        self.engine = ChessEngine(model_path)
        self.board = chess.Board()
        self.human_color = chess.WHITE  # Выберите цвет: chess.WHITE или chess.BLACK

    def print_board(self):
        """Отображает доску с юникод-символами"""
        print("\n" + "="*40)
        print(f"Ход {'белых' if self.board.turn else 'черных'}")
        print("="*40)
        
        unicode_pieces = {
            'r': '♖', 'n': '♘', 'b': '♗', 'q': '♕', 'k': '♔', 'p': '♙',
            'R': '♜', 'N': '♞', 'B': '♝', 'Q': '♛', 'K': '♚', 'P': '♟',
            '.': '·'
        }
        
        board_str = str(self.board)
        for char, piece in unicode_pieces.items():
            board_str = board_str.replace(char, piece)
        print(board_str)

    def get_user_move(self):
        """Получает и проверяет ход пользователя"""
        while True:
            try:
                move = input("Ваш ход (в формате UCI, например e2e4): ").strip()
                if move == "exit":
                    return None
                if chess.Move.from_uci(move) in self.board.legal_moves:
                    return move
                print("Недопустимый ход! Попробуйте снова.")
            except ValueError:
                print("Некорректный формат! Используйте UCI (например e2e4, g1f3)")

    def play(self):
        """Основной игровой цикл"""
        while not self.board.is_game_over():
            self.print_board()
            
            if self.board.turn == self.human_color:
                # Ход человека
                move = self.get_user_move()
                if not move:
                    print("Игра прервана.")
                    return
            else:
                # Ход нейросети
                move = self.engine.predict_move(self.board.fen())
                print(f"Ход нейросети: {move}")
                
            self.board.push_uci(move)
            
            # Проверка окончания игры
            if self.board.is_checkmate():
                print("Мат! Игра окончена.")
            elif self.board.is_stalemate():
                print("Пат! Ничья.")
            elif self.board.is_insufficient_material():
                print("Недостаточно материала! Ничья.")

        self.print_board()
        print("Результат:", self.board.result())

if __name__ == "__main__":
    game = ChessGame("checkpoint.pth")
    print("Шахматы против нейросети!")
    print("Введите ходы в формате UCI (например e2e4, g1f3)")
    print("Для выхода введите 'exit'")
    game.play()