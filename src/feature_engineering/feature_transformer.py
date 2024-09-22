from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureTransformer:
    def __init__(self, feature_columns: List[str]):
        self.feature_columns = feature_columns
        self.scaler = StandardScaler()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Create a new DataFrame to avoid SettingWithCopyWarning
        transformed_df = df.copy()

        # Apply log transformation to numerical columns
        for col in self.feature_columns:
            transformed_df[f'{col}_log'] = np.log1p(transformed_df[col])

        # Create rolling mean features
        for col in self.feature_columns:
            transformed_df[f'{col}_rolling_mean_7d'] = transformed_df[col].rolling(window=7).mean()
            transformed_df[f'{col}_rolling_mean_30d'] = transformed_df[col].rolling(window=30).mean()

        # Create lagged features
        for col in self.feature_columns:
            transformed_df[f'{col}_lag_1d'] = transformed_df[col].shift(1)
            transformed_df[f'{col}_lag_7d'] = transformed_df[col].shift(7)

        # Standardize features
        scaled_features = self.scaler.fit_transform(transformed_df[self.feature_columns])
        transformed_df[self.feature_columns] = scaled_features

        return transformed_df.dropna()  # Remove rows with NaN values after feature creation

    def get_feature_names(self) -> List[str]:
        return self.feature_columns + [f'{col}_log' for col in self.feature_columns] + \
               [f'{col}_rolling_mean_7d' for col in self.feature_columns] + \
               [f'{col}_rolling_mean_30d' for col in self.feature_columns] + \
               [f'{col}_lag_1d' for col in self.feature_columns] + \
               [f'{col}_lag_7d' for col in self.feature_columns]