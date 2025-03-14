import torch

class BigDataConfig:
    CHUNK_SIZE = 1_000_000     # Размер чанка
    CHECKPOINT_INTERVAL = 1    # Сохранять чекпоинт каждые N чанков
    RESUME_FROM_CHUNK = 14      # Продолжить с чанка (для рестарта)
    CACHE_DIR = "./data_cache"  # Директория для кэша

class TrainingConfig:
    BATCH_SIZE = 768         # Увеличиваем размер батча
    NUM_EPOCHS = 5           # Количество эпох
    LEARNING_RATE = 0.0003     # Скорость обучения
    WEIGHT_DECAY = 0.01     # L2 регуляризация
    NUM_WORKERS = 8           # Количество процессов для загрузки данных
    ACCUMULATION_STEPS = 2    # Накопление градиентов для больших батчей
    LOG_INTERVAL = 2*BATCH_SIZE
    GRAD_CLIP = 1.0

class ModelConfig:
    INPUT_CHANNELS = 14       # Количество входных каналов
    BASE_CHANNELS = 64        # Базовое количество каналов
    DROPOUT_RATE = 0.3        # Регуляризация
    USE_AMP = True            # Использовать автоматическую смешанную точность
    HEAD_HIDDEN_SIZE = 4096
    ACTIVATION = "gelu"

class SystemConfig:
    DEVICE = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else 'cpu'
    PIN_MEMORY = True         # Ускорение передачи данных в GPU
    LOG_INTERVAL = TrainingConfig.BATCH_SIZE*2         # Логирование каждые N батчей
    
import pathlib

# Конфигурационные параметры
class Config:
    # Путь к папке с PGN-файлами
    PGN_DIRECTORY = pathlib.Path("./pgn_games")
    
    # Путь для сохранения CSV
    OUTPUT_CSV = pathlib.Path("chess_dataset.csv")
    
    # Минимальный рейтинг игроков (None если не фильтровать)
    MIN_ELO = None
    
    # Фильтровать только завершенные игры
    REQUIRE_RESULT = True
    
import logging

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s:%(name)s:%(levelname)s:%(message)s"
)