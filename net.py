import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MyRNN(keras.Model):
    def __init__(self, units):
        super(MyRNN, self).__init__()
        self.fc1 = layers.Dense(64)       # 第一层全连接层，用于数据降维
        self.rnn = keras.Sequential([
            # layers.SimpleRNN(units, dropout=0.5, return_sequences=True, unroll=True),
            # layers.SimpleRNN(units, dropout=0.5, unroll=True)
            layers.BatchNormalization(),    # 防止梯度弥散
            layers.LSTM(units, dropout=0.2, return_sequences=True, unroll=True),   # 网络主体，三层LSTM层，加入dropout防止过拟合
            layers.LSTM(units, dropout=0.2, return_sequences=True, unroll=True),
            layers.LSTM(units, dropout=0.2, unroll=True),
            layers.BatchNormalization(),
        ])
        self.fc2 = layers.Dense(64, activation="sigmoid")
        self.outlayer = layers.Dense(1)       # 输出层

    def call(self, inputs, training=True):
        x = inputs
        x = self.fc1(x)
        x = self.rnn(x, training=training)
        x = self.fc2(x)
        x = self.outlayer(x)
        # prob = tf.sigmoid(x)

        return x
