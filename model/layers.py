import tensorflow as tf
from config.settings import Config

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, 3, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return tf.nn.relu(x + inputs)

class PolicyHead(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(2, 1)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(Config.OUTPUT_SIZE)
    
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.flatten(x)
        return self.dense(x)

class ValueHead(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(1, 1)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='tanh')
    
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)
