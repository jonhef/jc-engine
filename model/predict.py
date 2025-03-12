import torch
import chess
from model.model import ChessNet
from data.data_processing import fen_to_tensor
import argparse
from model.mcts import MCTS, MCTSNode

class ChessEngine:
    def __init__(self, model_path, device='mps'):
        self.model = ChessNet()
        self.device = torch.device(device)
        
        # Загрузка чекпоинта
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Извлекаем только веса модели
        try:
            model_state_dict = checkpoint['model_state_dict']
        except KeyError:
            model_state_dict = checkpoint
        
        # Загружаем веса в модель
        self.model.load_state_dict(model_state_dict)
        
        self.model.to(self.device)
        self.model.eval()
        self.mcts = MCTS(self.model)
    
    def predict_move(self, fen):
        # Преобразование позиции в тензор
        board = chess.Board(fen)
        # Убираем .unsqueeze(0), если функция уже возвращает правильную размерность
        input_tensor = fen_to_tensor(fen).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            from_logits, to_logits = self.model(input_tensor)
        
        # Преобразование в вероятности
        from_probs = torch.softmax(from_logits, dim=1).cpu().numpy()[0]
        to_probs = torch.softmax(to_logits, dim=1).cpu().numpy()[0]
        
        # Поиск лучшего легального хода
        best_move = None
        best_score = -1
        
        for move in board.legal_moves:
            score = from_probs[move.from_square] * to_probs[move.to_square]
            if score > best_score:
                best_score = score
                best_move = move
                
        return self.mcts.search(board)

# Пример использования
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoint.pth")
    reader = parser.parse_args()
    engine = ChessEngine(reader.model)
    
    # Стартовая позиция
    start_pos = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    print(f"Best move: {engine.predict_move(start_pos)}")
    
    # Позиция с шахом
    check_pos = 'r1bqkb1r/pppp1ppp/2n2n2/4p3/2P5/2N3P1/PP1PPPBP/R1BQK1NR b KQkq - 2 4'
    print(f"Best move: {engine.predict_move(check_pos)}")