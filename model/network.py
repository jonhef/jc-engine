import tensorflow as tf
from model.layers import ResidualBlock, PolicyHead, ValueHead
from config.settings import Config

class ChessModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv_init = tf.keras.layers.Conv2D(Config.FILTERS, 3, padding='same')
        self.res_blocks = [ResidualBlock(Config.FILTERS) for _ in range(Config.RESIDUAL_BLOCKS)]
        self.policy_head = PolicyHead()
        self.value_head = ValueHead()
    
    def call(self, inputs):
        x = self.conv_init(inputs)
        for block in self.res_blocks:
            x = block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
