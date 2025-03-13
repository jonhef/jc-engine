# data_fetching.py
import requests
import chess.pgn
import io
import csv
import time
import data.settings
import logging as log

logging = log.getLogger(__name__)

def get_top_players():
    response = requests.get("https://lichess.org/api/player/top/200/classical")
    return [user['username'] for user in response.json()['users']] if response.ok else []

def fetch_games(username, max_games=200):
    url = f"https://lichess.org/api/games/user/{username}"
    params = {
        'perfType': 'classical',
        'rated': 'true',
        'max': max_games,
        'clocks': 'false'
    }
    headers = {'Accept': 'application/x-chess-pgn'}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        return response.text.split('\n\n\n') if response.ok else []
    except Exception as e:
        print(f"Error fetching games for {username}: {str(e)}")
        return []

def process_game(pgn_text, username, writer):
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if not game:
            return

        board = game.board()
        player_color = None
        if game.headers['White'] == username:
            player_color = chess.WHITE
        elif game.headers['Black'] == username:
            player_color = chess.BLACK
        else:
            return

        for move in game.mainline_moves():
            if board.turn == player_color:
                writer.writerow([board.fen(), move.uci()])
            board.push(move)

    except Exception as e:
        logging.error(f"Error processing game: {str(e)}")
        logging.error(f"Problematic PGN fragment: {pgn_text[:200]}...")

def main():
    with open('chess_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['fen', 'move'])
        
        top_players = get_top_players()
        for i, username in enumerate(top_players[:200]):
            logging.info(f"Processing {username} ({i+1}/{len(top_players)})")
            games = fetch_games(username)
            
            for pgn in games:
                if pgn.strip():
                    process_game(pgn, username, writer)
            
            time.sleep(2)  # Соблюдение rate limit

if __name__ == '__main__':
    main()