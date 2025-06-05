# models.py
import tensorflow as tf
from tensorflow.keras import layers

class LSTMAutoencoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim=128, lstm_units=256, dropout_rate=0.3):
        super().__init__()
        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)
        self.encoder_lstm = layers.LSTM(lstm_units, return_sequences=True, return_state=True, dropout=dropout_rate)
        self.repeat_vector = layers.RepeatVector(300)
        self.decoder_lstm = layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate)
        self.output_layer = layers.TimeDistributed(layers.Dense(vocab_size, activation='softmax'))

    def call(self, inputs):
        x = self.embedding(inputs)
        encoder_output, h, c = self.encoder_lstm(x)
        decoded = self.repeat_vector(h)
        decoded = self.decoder_lstm(decoded, initial_state=[h, c])
        return self.output_layer(decoded)

    def encode(self, inputs):
        x = self.embedding(inputs)
        _, h, _ = self.encoder_lstm(x)
        return h
