import requests
import concurrent.futures
from typing import List, Dict
import time
import json
from tqdm import tqdm
from config.settings import Config

class LichessLoader:
    def __init__(self, max_workers=10, rate_limit_delay=1.0):
        self.base_url = "https://lichess.org/api"
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.headers = {"Accept": "application/json"}

    def get_top_players(self, perf_type="classical", limit=200) -> List[str]:
        url = f"{self.base_url}/player/top/{limit}/{perf_type}"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return [user["username"] for user in response.json()["users"]]
        except Exception as e:
            print(f"Error fetching top players: {e}")
            return []

    def _fetch_games(self, username: str) -> List[Dict]:
        time.sleep(self.rate_limit_delay)  # Задержка для соблюдения лимитов
        try:
            params = {
                "perfType": "classical",
                "max": 300,
                "rated": "true",
                "pgnInJson": "true",
                "clocks": "true"
            }
            response = requests.get(
                f"{self.base_url}/games/user/{username}",
                params=params,
                headers=self.headers,
                timeout=20
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching games for {username}: {e}")
            return []

    def get_games_parallel(self, usernames: List[str]) -> Dict[str, List[Dict]]:
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_user = {
                executor.submit(self._fetch_games, user): user 
                for user in usernames
            }
            
            with tqdm(total=len(usernames), desc="Downloading games") as pbar:
                for future in concurrent.futures.as_completed(future_to_user):
                    user = future_to_user[future]
                    try:
                        results[user] = future.result()
                    except Exception as e:
                        print(f"Error processing {user}: {e}")
                    finally:
                        pbar.update(1)
        return results

    def save_to_json(self, data: Dict, filename: str):
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

if __name__ == "__main__":
    loader = LichessLoader(max_workers=5, rate_limit_delay=1.5)
    
    # Шаг 1: Получение топ-200 игроков
    print("Fetching top players...")
    top_players = loader.get_top_players()
    
    # Шаг 2: Параллельная загрузка игр
    print(f"Downloading games for {len(top_players)} players...")
    all_games = loader.get_games_parallel(top_players)  # Тест на 5 игроках
    
    # Шаг 3: Сохранение результатов
    loader.save_to_json(all_games, "lichess_games.json")
    print("Data saved to lichess_games.json")
