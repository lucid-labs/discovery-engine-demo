import tensorflow as tf
from typing import List, Tuple

class NeuralArchitectureSearch:
    def __init__(self, input_shape: Tuple[int, int], output_shape: int):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def search(self, X_train, y_train, X_val, y_val) -> tf.keras.Model:
        architectures = [
            self._create_lstm_model,
            self._create_gru_model,
            self._create_transformer_model
        ]

        best_model = None
        best_val_loss = float('inf')

        for create_model in architectures:
            model = create_model()
            model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), verbose=0)
            val_loss = model.evaluate(X_val, y_val, verbose=0)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model

        return best_model

    def _create_lstm_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=self.input_shape, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(self.output_shape)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def _create_gru_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.GRU(64, input_shape=self.input_shape, return_sequences=True),
            tf.keras.layers.GRU(32),
            tf.keras.layers.Dense(self.output_shape)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def _create_transformer_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=self.input_shape),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.output_shape)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model