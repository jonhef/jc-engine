import chess.pgn
import csv
import os
from data.settings import Config
from pathlib import Path
from tqdm import tqdm  # Для прогресс-бара
import logging as log

logging = log.getLogger(__name__)

def process_pgn_file(pgn_path, writer):
    """Обрабатывает один PGN-файл"""
    with open(pgn_path, 'r', encoding='utf-8') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            
            # Проверка условий фильтрации
            if not validate_game(game):
                continue
                
            # Извлекаем данные из игры
            extract_game_data(game, writer)

def validate_game(game):
    """Проверяет соответствие игры критериям"""
    headers = game.headers
    
    # Проверка результата
    if Config.REQUIRE_RESULT and headers.get("Result", "*") == "*":
        return False
        
    # Проверка рейтингов
    if Config.MIN_ELO:
        try:
            white_elo = int(headers.get("WhiteElo", "0"))
            black_elo = int(headers.get("BlackElo", "0"))
            if white_elo < Config.MIN_ELO or black_elo < Config.MIN_ELO:
                return False
        except ValueError:
            return False
            
    return True

def extract_game_data(game, writer):
    """Извлекает позиции и ходы из игры"""
    board = game.board()
    for move in game.mainline_moves():
        # Сохраняем FEN до хода
        fen = board.fen()
        
        # Получаем ход в UCI-формате
        uci_move = move.uci()
        
        # Записываем в CSV
        writer.writerow([fen, uci_move])
        
        # Делаем ход на доске
        board.push(move)

def main():
    # Создаем выходной каталог
    Config.OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    # Получаем все PGN-файлы
    pgn_files = list(Config.PGN_DIRECTORY.rglob("*.pgn"))
    
    # Обработка файлов
    with open(Config.OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['fen', 'move'])  # Заголовок
        
        for pgn_path in tqdm(pgn_files, desc="Processing PGN files"):
            try:
                process_pgn_file(pgn_path, writer)
            except Exception as e:
                logging.error(f"Error processing {pgn_path}: {str(e)}")

if __name__ == "__main__":
    main()