import logging
import os
from typing import Dict

import pandas as pd

logging.basicConfig(level=logging.INFO)

class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        logging.info(f"DataLoader initialized with directory: {self.data_dir}")

    def load_csv_files(self) -> Dict[str, pd.DataFrame]:
        data_dict = {}
        logging.info(f"Searching for CSV files in {self.data_dir} and its subdirectories")
        
        if not os.path.exists(self.data_dir):
            logging.error(f"Directory does not exist: {self.data_dir}")
            return data_dict

        for root, dirs, files in os.walk(self.data_dir):
            for filename in files:
                if filename.endswith('.csv'):
                    file_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(file_path, self.data_dir)
                    logging.info(f"Loading file: {relative_path}")
                    try:
                        df = pd.read_csv(file_path)
                        # Extract protocol and asset information from the path
                        path_parts = relative_path.split(os.sep)
                        protocol = path_parts[0] if len(path_parts) > 1 else "Unknown"
                        asset = path_parts[1].split('-')[2] if len(path_parts) > 1 else "Unknown"
                        
                        # Add protocol and asset columns to the DataFrame
                        df['protocol'] = protocol
                        df['asset'] = asset
                        
                        data_dict[relative_path] = df
                        logging.info(f"Successfully loaded {relative_path} with shape {df.shape}")
                    except Exception as e:
                        logging.error(f"Error loading {relative_path}: {str(e)}")
        
        if not data_dict:
            logging.warning("No CSV files were loaded.")
        else:
            logging.info(f"Loaded {len(data_dict)} CSV files.")
        
        return data_dict

    def merge_dataframes(self, dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not dataframes:
            logging.error("No dataframes to merge.")
            raise ValueError("No objects to concatenate")
        
        logging.info(f"Merging {len(dataframes)} dataframes")
        merged_df = pd.concat(dataframes.values(), axis=0, ignore_index=True)
        logging.info(f"Merged dataframe shape: {merged_df.shape}")
        return merged_df