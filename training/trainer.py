import tensorflow as tf
from model.network import ChessModel
from config.settings import Config

class ChessTrainer:
    def __init__(self):
        self.model = ChessModel()
        self.optimizer = tf.keras.optimizers.Adam(Config.LEARNING_RATE)
    
    def train_step(self, states, masks, policies, values):
        with tf.GradientTape() as tape:
            pred_policies, pred_values = self.model(states)
            
            policy_loss = tf.keras.losses.categorical_crossentropy(
                policies, pred_policies, from_logits=True
            )
            value_loss = tf.keras.losses.mean_squared_error(values, pred_values)
            total_loss = tf.reduce_mean(policy_loss + value_loss)
        
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return total_loss.numpy()
