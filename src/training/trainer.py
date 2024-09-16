import tensorflow as tf
from typing import Tuple

class ModelTrainer:
    def __init__(self, model: tf.keras.Model, epochs: int = 100, batch_size: int = 32):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, X_train, y_train, X_val, y_val) -> tf.keras.callbacks.History:
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        return history