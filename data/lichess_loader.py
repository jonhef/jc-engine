import requests
import chess.pgn
import io
from typing import List, Dict
from datetime import datetime
from config.settings import Config

class LichessLoader:
    def __init__(self):
        self.base_url = "https://lichess.org/api"

    def _make_request(self, endpoint: str, params: Dict) -> requests.Response:
        try:
            response = requests.get(
                f"{self.base_url}{endpoint}",
                params=params,
                headers={"Accept": "application/x-ndjson"},
                timeout=10
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None

    def load_games(
        self,
        username: str = None,
        since: datetime = None,
        until: datetime = None,
        max_games: int = 100,
        perf_type: str = "rapid,classical,blitz"
    ) -> List[str]:
        params = {
            "max": max_games,
            "perfType": perf_type,
            "pgnInJson": True,
            "clocks": True,
            "evals": True
        }

        if username:
            endpoint = f"/games/user/{username}"
            params["rated"] = True
        else:
            endpoint = "/games/export"
            params["since"] = since.timestamp() * 1000 if since else None
            params["until"] = until.timestamp() * 1000 if until else None

        response = self._make_request(endpoint, params)
        if not response:
            return []

        return [
            game["pgn"] 
            for game in response.json()
            if "pgn" in game
        ]

    @staticmethod
    def pgn_to_training_data(pgn: str) -> Dict:
        game = chess.pgn.read_game(io.StringIO(pgn))
        if not game:
            return None

        board = game.board()
        positions = []
        moves = []
        move_sans = []

        for node in game.mainline():
            move = node.move
            positions.append(board.fen())
            
            # Сохраняем оба формата
            moves.append(move.uci())
            move_sans.append(board.san(move))
            
            board.push(move)

        return {
            "positions": positions,
            "uci_moves": moves,
            "san_moves": move_sans,
            "metadata": {
                "result": game.headers.get("Result"),
                "white_rating": game.headers.get("WhiteElo"),
                "black_rating": game.headers.get("BlackElo"),
                "date": game.headers.get("Date")
            }
        }
