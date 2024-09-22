from typing import Tuple

import tensorflow as tf
from tqdm import tqdm


class ModelTrainer:
    def __init__(self, model: tf.keras.Model, epochs: int = 100, batch_size: int = 32):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, X_train, y_train, X_val, y_val) -> tf.keras.callbacks.History:
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        
        # Custom callback to update the progress bar
        class ProgressBarCallback(tf.keras.callbacks.Callback):
            def __init__(self, epochs):
                super(ProgressBarCallback, self).__init__()
                self.pbar = tqdm(total=epochs, desc="Training")

            def on_epoch_end(self, epoch, logs=None):
                self.pbar.update(1)
                self.pbar.set_postfix({
                    "Loss": f"{logs['loss']:.4f}",
                    "Val Loss": f"{logs['val_loss']:.4f}"
                })

            def on_train_end(self, logs=None):
                self.pbar.close()

        progress_bar = ProgressBarCallback(self.epochs)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, progress_bar],
            verbose=0  # Set to 0 as we're using our custom progress bar
        )
        return history