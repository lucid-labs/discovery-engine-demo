import pytest
import pandas as pd
import numpy as np
from src.feature_engineering.feature_transformer import FeatureTransformer

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100)
    })

def test_transform(sample_data):
    feature_columns = ['feature1', 'feature2']
    transformer = FeatureTransformer(feature_columns)
    transformed_data = transformer.transform(sample_data)
    
    expected_columns = (
        feature_columns +
        [f'{col}_log' for col in feature_columns] +
        [f'{col}_rolling_mean_7d' for col in feature_columns] +
        [f'{col}_rolling_mean_30d' for col in feature_columns] +
        [f'{col}_lag_1d' for col in feature_columns] +
        [f'{col}_lag_7d' for col in feature_columns]
    )
    
    assert all(col in transformed_data.columns for col in expected_columns)
    assert transformed_data.shape[1] == len(expected_columns)

def test_get_feature_names(sample_data):
    feature_columns = ['feature1', 'feature2']
    transformer = FeatureTransformer(feature_columns)
    feature_names = transformer.get_feature_names()
    
    expected_names = (
        feature_columns +
        [f'{col}_log' for col in feature_columns] +
        [f'{col}_rolling_mean_7d' for col in feature_columns] +
        [f'{col}_rolling_mean_30d' for col in feature_columns] +
        [f'{col}_lag_1d' for col in feature_columns] +
        [f'{col}_lag_7d' for col in feature_columns]
    )
    
    assert feature_names == expected_names