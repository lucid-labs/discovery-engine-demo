import pandas as pd
from typing import List

class DataPreprocessor:
    def __init__(self, target_columns: List[str]):
        self.target_columns = target_columns

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Remove rows with NaN values in target columns
        df = df.dropna(subset=self.target_columns)
        
        return df

    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.8) -> tuple:
        train_size = int(len(df) * train_ratio)
        train_data = df[:train_size]
        test_data = df[train_size:]
        return train_data, test_data