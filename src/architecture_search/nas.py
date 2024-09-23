import tensorflow as tf
from typing import Tuple
from tqdm import tqdm


class NeuralArchitectureSearch:
    def __init__(self, input_shape: Tuple[int, int], output_shape: int):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def search(self, X_train, y_train, X_val, y_val) -> tf.keras.Model:
        architectures = [
            self._create_lstm_model,
            self._create_gru_model,
            self._create_transformer_model,
        ]

        best_model = None
        best_val_loss = float("inf")

        with tqdm(total=len(architectures), desc="Neural Architecture Search") as pbar:
            for create_model in architectures:
                model = create_model()
                model.fit(
                    X_train,
                    y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    verbose=0,
                )
                val_loss = model.evaluate(X_val, y_val, verbose=0)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model

                pbar.update(1)
                pbar.set_postfix({"Best Val Loss": best_val_loss})

        return best_model

    def _create_lstm_model(self) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
        x = tf.keras.layers.LSTM(32)(x)
        outputs = tf.keras.layers.Dense(self.output_shape)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse")
        return model

    def _create_gru_model(self) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.layers.GRU(64, return_sequences=True)(inputs)
        x = tf.keras.layers.GRU(32)(x)
        outputs = tf.keras.layers.Dense(self.output_shape)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse")
        return model

    def _create_transformer_model(self) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation="relu")(inputs)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        outputs = tf.keras.layers.Dense(self.output_shape)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse")
        return model
