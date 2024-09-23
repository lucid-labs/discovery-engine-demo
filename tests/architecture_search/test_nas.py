import pytest
import numpy as np
import tensorflow as tf
from src.architecture_search.nas import NeuralArchitectureSearch


@pytest.fixture
def sample_data():
    X = np.random.rand(100, 10, 1)
    y = np.random.rand(100, 2)
    return X, y


def test_search(sample_data):
    X, y = sample_data
    nas = NeuralArchitectureSearch(input_shape=(10, 1), output_shape=2)
    best_model = nas.search(X, y, X, y)

    assert isinstance(best_model, tf.keras.Model)
    assert best_model.input_shape == (None, 10, 1)
    assert best_model.output_shape == (None, 2)


def test_create_models(sample_data):
    X, y = sample_data
    nas = NeuralArchitectureSearch(input_shape=(10, 1), output_shape=2)

    lstm_model = nas._create_lstm_model()
    gru_model = nas._create_gru_model()
    transformer_model = nas._create_transformer_model()

    models = [lstm_model, gru_model, transformer_model]

    for model in models:
        assert isinstance(model, tf.keras.Model)
        assert model.input_shape == (None, 10, 1)
        assert model.output_shape == (None, 2)
