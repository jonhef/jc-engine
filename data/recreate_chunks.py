# recreate_chunks.py
import pandas as pd
from pathlib import Path
from data.settings import BigDataConfig
import tqdm

def create_chunks():
    Path(BigDataConfig.CACHE_DIR).mkdir(exist_ok=True)
    
    # Читаем основной файл с указанием заголовков
    reader = pd.read_csv(
        "puzzles.csv",
        chunksize=BigDataConfig.CHUNK_SIZE,
        names=['fen', 'move'],  # Явно задаем названия столбцов
        header=None             # Если в исходном файле нет заголовков
    )
    
    for i, chunk in tqdm.tqdm(enumerate(reader)):
        chunk.to_csv(
            f"puzzles/chunk_{i}.csv",
            index=False,
            header=['fen', 'move']  # Сохраняем заголовки в каждый файл
        )

if __name__ == "__main__":
    create_chunks()