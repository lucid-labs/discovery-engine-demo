import pytest
import pandas as pd
import numpy as np
from src.data_processing.data_preprocessor import DataPreprocessor

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2021-01-01', periods=100, freq='H'),
        'target1': np.random.rand(100),
        'target2': np.random.rand(100),
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100)
    })

def test_preprocess(sample_data):
    preprocessor = DataPreprocessor(['target1', 'target2'])
    processed_data = preprocessor.preprocess(sample_data)
    
    assert 'timestamp' not in processed_data.columns
    assert processed_data.index.name == 'timestamp'
    assert all(col in processed_data.columns for col in ['target1', 'target2', 'feature1', 'feature2'])

def test_split_data(sample_data):
    preprocessor = DataPreprocessor(['target1', 'target2'])
    processed_data = preprocessor.preprocess(sample_data)
    train_data, test_data = preprocessor.split_data(processed_data, train_ratio=0.8)
    
    assert len(train_data) == 80
    assert len(test_data) == 20
    assert all(col in train_data.columns for col in processed_data.columns)
    assert all(col in test_data.columns for col in processed_data.columns)