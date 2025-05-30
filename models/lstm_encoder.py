import tensorflow as tf
from tensorflow.keras import layers

class LSTMEncoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim=128, lstm_units=256, dropout_rate=0.3):
        super(LSTMEncoder, self).__init__()
        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)
        self.lstm = layers.LSTM(lstm_units, return_sequences=False, dropout=dropout_rate)
        

    def call(self, inputs):
        x = self.embedding(inputs)
        output = self.lstm(x)
        return output
