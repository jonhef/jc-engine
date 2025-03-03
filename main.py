import time
import requests
import json
import os
from datetime import datetime, timedelta
from data.lichess_loader import LichessLoader
from data.processors import MoveConverter
from config.settings import Config
import h5py
import numpy as np
from data.converters import fen_to_tensor

class ChessDatasetCreator:
    def __init__(self):
        self.loader = LichessLoader()
        self.converter = MoveConverter()
        self.dataset_path = "dataset/chess_dataset.h5"
        os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)

    def get_top_players(self, perf_type="classical", limit=200):
        """Получение топ-игроков с Lichess"""
        url = f"https://lichess.org/api/player/top/{limit}/{perf_type}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return [user["username"] for user in response.json()["users"]]
        except Exception as e:
            print(f"Error fetching top players: {e}")
            return []

    def process_user_games(self, username):
        """Обработка игр одного пользователя"""
        try:
            games = self.loader.load_games(
                username=username,
                max_games=300,
                perf_type="classical"
            )
            print(f"Found {len(games)} games for {username}")
            
            user_data = []
            for pgn in games:
                game_data = self.loader.pgn_to_training_data(pgn)
                if not game_data:
                    continue
                
                # Конвертация SAN → индексы
                indices = self.converter.convert_san_sequence(game_data["san_moves"])
                if len(indices) < 5:  # Пропускаем короткие игры
                    continue
                
                # Сохраняем данные
                user_data.append({
                    "positions": [fen_to_tensor(fen) for fen in game_data["positions"]],
                    "moves": indices,
                    "metadata": game_data["metadata"]
                })
            
            return user_data
        except Exception as e:
            print(f"Error processing {username}: {e}")
            return []

    def save_to_hdf5(self, data):
        """Сохранение данных в HDF5"""
        with h5py.File(self.dataset_path, "a") as hf:
            for idx, game in enumerate(data):
                grp = hf.create_group(f"game_{datetime.now().timestamp()}_{idx}")
                grp.create_dataset("positions", data=np.array(game["positions"]), compression="gzip")
                grp.create_dataset("moves", data=game["moves"], compression="gzip")
                grp.attrs["result"] = game["metadata"]["result"]
                grp.attrs["rating"] = f"{game['metadata']['white_rating']}-{game['metadata']['black_rating']}"
                grp.attrs["date"] = game["metadata"]["date"]

    def create_dataset(self):
        """Основной пайплайн создания датасета"""
        players = self.get_top_players(limit=200)
        if not players:
            return
        
        print(f"Found {len(players)} top players")
        
        for i, username in enumerate(players):
            print(f"\nProcessing {i+1}/{len(players)}: {username}")
            
            # Загрузка и обработка игр
            user_games = self.process_user_games(username)
            
            if not user_games:
                print(f"No valid games for {username}")
                continue
            
            # Сохранение данных
            self.save_to_hdf5(user_games)
            
            # Задержка для соблюдения лимитов API
            time.sleep(2)

if __name__ == "__main__":
    creator = ChessDatasetCreator()
    creator.create_dataset()
    print("Dataset creation completed!")
