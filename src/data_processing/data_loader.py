import os
import pandas as pd
from typing import List, Dict

class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_csv_files(self) -> Dict[str, pd.DataFrame]:
        data_dict = {}
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.data_dir, filename)
                df = pd.read_csv(file_path)
                data_dict[filename] = df
        return data_dict

    def merge_dataframes(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(dataframes, axis=0, ignore_index=True)