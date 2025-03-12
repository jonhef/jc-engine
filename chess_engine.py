import sys
import chess
from model.predict import ChessEngine  # Ваша функция для предсказания хода

engine = ChessEngine("checkpoint.pth")  # Ваша функция для предсказания хода
predict_move = engine.predict_move

def main():
    board = chess.Board()
    while True:
        line = sys.stdin.readline().strip()
        if line == "uci":
            print("id name jc-engine")
            print("id author jonhef")
            print("uciok")
        elif line == "isready":
            print("readyok")
        elif line.startswith("position"):
            parts = line.split()
            if parts[1] == "startpos":
                board.reset()
                if len(parts) > 2 and parts[2] == "moves":
                    for move in parts[3:]:
                        board.push_uci(move)
            elif parts[1] == "fen":
                fen = " ".join(parts[2:8])
                board.set_fen(fen)
                if len(parts) > 8 and parts[8] == "moves":
                    for move in parts[9:]:
                        board.push_uci(move)
        elif line.startswith("go"):
            # Используйте вашу модель для предсказания хода
            best_move = predict_move(board.fen())  # Ваша функция
            print(f"bestmove {best_move}")
        elif line == "quit":
            break

if __name__ == "__main__":
    main()