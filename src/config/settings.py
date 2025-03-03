import pathlib

class Config:
    INPUT_SHAPE = (8, 8, 18)
    OUTPUT_SIZE = 4672
    RESIDUAL_BLOCKS = 20
    FILTERS = 256
    SIMULATIONS = 800
    C_PUCT = 1.25
    STOCKFISH_PATH = str(pathlib.Path.home() / "stockfish/stockfish")
    MODEL_SAVE_PATH = "models/chess_ai.keras"
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001

config = Config()
