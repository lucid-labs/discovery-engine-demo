import pytest
import numpy as np
import tensorflow as tf
from src.training.trainer import ModelTrainer

@pytest.fixture
def sample_model():
    inputs = tf.keras.Input(shape=(5,))
    x = tf.keras.layers.Dense(10, activation='relu')(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

@pytest.fixture
def sample_data():
    X = np.random.rand(100, 5)
    y = np.random.rand(100, 1)
    return X, y

def test_train(sample_model, sample_data):
    X, y = sample_data
    trainer = ModelTrainer(sample_model, epochs=10, batch_size=32)
    history = trainer.train(X, y, X, y)
    
    assert isinstance(history, tf.keras.callbacks.History)
    assert len(history.history['loss']) <= 10  # Changed to <= because of early stopping
    assert len(history.history['val_loss']) <= 10  # Changed to <= because of early stopping