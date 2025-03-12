import csv
import chess
import tqdm

path = "lichess_db_puzzle.csv"

with open(path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    with open("puzzles.csv", 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['fen', 'move'])
        
        for row in tqdm.tqdm(reader):
            try:
                board = chess.Board(row[1])
                for move in row[2].split(' '):
                    board.push_uci(move)
                    writer.writerow([board.fen(), move])
            except Exception as e:
                print(e)