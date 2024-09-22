import pytest
import pandas as pd
import os
from src.data_processing.data_loader import DataLoader

@pytest.fixture
def sample_data_dir(tmp_path):
    # Create a temporary directory with sample CSV files
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create sample CSV files
    df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df2 = pd.DataFrame({'A': [7, 8, 9], 'B': [10, 11, 12]})
    
    df1.to_csv(data_dir / "file1.csv", index=False)
    df2.to_csv(data_dir / "file2.csv", index=False)
    
    return data_dir

def test_load_csv_files(sample_data_dir):
    loader = DataLoader(str(sample_data_dir))
    data_dict = loader.load_csv_files()
    
    assert len(data_dict) == 2
    assert all(isinstance(df, pd.DataFrame) for df in data_dict.values())
    assert list(data_dict.keys()) == ["file1.csv", "file2.csv"]

def test_merge_dataframes(sample_data_dir):
    loader = DataLoader(str(sample_data_dir))
    data_dict = loader.load_csv_files()
    merged_df = loader.merge_dataframes(data_dict)  # Pass the dictionary directly
    
    assert len(merged_df) == 6
    assert set(merged_df.columns) == {'A', 'B', 'protocol', 'asset'}  # Update expected columns