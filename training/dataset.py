import tensorflow as tf
import numpy as np

class ChessDataset:
    def __init__(self):
        self.positions = []
        self.policies = []
        self.values = []
    
    def add_game(self, game_data):
        self.positions.extend(game_data['positions'])
        self.policies.extend(game_data['policies'])
        self.values.extend(game_data['values'])
    
    def get_batch(self, batch_size):
        indices = np.random.choice(len(self.positions), batch_size, replace=False)
        return (
            tf.stack([self.positions[i] for i in indices]),
            tf.stack([self.policies[i] for i in indices]),
            tf.stack([self.values[i] for i in indices])
        )
