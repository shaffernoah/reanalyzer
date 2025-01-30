"""
Feature engineering and data enrichment for real estate market analysis.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        
    def create_time_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Extract temporal features from date column.
        """
        df = df.copy()
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['quarter'] = df[date_column].dt.quarter
        df['month_of_quarter'] = df[date_column].dt.month % 3 + 1
        df['day_of_week'] = df[date_column].dt.dayofweek
        
        # Create cyclical features for temporal patterns
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        return df

    def calculate_rolling_statistics(self, df: pd.DataFrame, 
                                   value_column: str,
                                   windows: List[int] = [3, 6, 12]) -> pd.DataFrame:
        """
        Calculate rolling statistics for the value column.
        """
        df_stats = df.copy()
        
        # Skip if DataFrame is empty or value column is not present
        if df_stats.empty or value_column not in df_stats.columns:
            return df_stats
            
        # Convert datetime column to numeric if needed
        if pd.api.types.is_datetime64_any_dtype(df_stats[value_column]):
            df_stats[value_column] = pd.to_numeric(df_stats[value_column].astype(np.int64))
            
        for window in windows:
            prefix = f'{window}m'
            df_stats[f'{prefix}_rolling_mean'] = df_stats[value_column].rolling(window).mean()
            df_stats[f'{prefix}_rolling_std'] = df_stats[value_column].rolling(window).std()
            df_stats[f'{prefix}_rolling_min'] = df_stats[value_column].rolling(window).min()
            df_stats[f'{prefix}_rolling_max'] = df_stats[value_column].rolling(window).max()
            
        return df_stats

    def calculate_market_indicators(self, df: pd.DataFrame, 
                                  value_column: str) -> pd.DataFrame:
        """
        Calculate various market indicators.
        """
        df_indicators = df.copy()
        
        # Skip if DataFrame is empty or value column is not present
        if df_indicators.empty or value_column not in df_indicators.columns:
            return df_indicators
            
        # Convert datetime column to numeric if needed
        if pd.api.types.is_datetime64_any_dtype(df_indicators[value_column]):
            df_indicators[value_column] = pd.to_numeric(df_indicators[value_column].astype(np.int64))
            
        # Calculate percentage changes
        df_indicators['pct_change'] = df_indicators[value_column].pct_change()
        df_indicators['pct_change_yoy'] = df_indicators[value_column].pct_change(periods=12)
        
        # Calculate momentum indicators
        df_indicators['momentum'] = df_indicators[value_column].diff()
        df_indicators['acceleration'] = df_indicators['momentum'].diff()
        
        # Calculate volatility
        df_indicators['volatility'] = df_indicators['pct_change'].rolling(window=12).std()
        
        # Calculate moving averages
        df_indicators['sma_short'] = df_indicators[value_column].rolling(window=3).mean()
        df_indicators['sma_long'] = df_indicators[value_column].rolling(window=12).mean()
        df_indicators['trend_strength'] = df_indicators['sma_short'] / df_indicators['sma_long'] - 1
        
        return df_indicators

    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       n_features: int = 10) -> pd.DataFrame:
        """
        Select most important features using univariate feature selection.
        """
        selector = SelectKBest(score_func=f_regression, k=n_features)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)

    def scale_features(self, df: pd.DataFrame, exclude_columns: List[str] = None) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        """
        if df.empty:
            return df
            
        # Create a copy of the DataFrame
        df_scaled = df.copy()
        
        # If no exclude columns specified, initialize empty list
        if exclude_columns is None:
            exclude_columns = []
            
        # Get numeric columns that are not in exclude_columns
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        columns_to_scale = [col for col in numeric_columns if col not in exclude_columns]
        
        if not columns_to_scale:
            return df_scaled
            
        # Scale only numeric columns
        scaled_values = self.scaler.fit_transform(df_scaled[columns_to_scale])
        df_scaled[columns_to_scale] = scaled_values
        
        return df_scaled

    def create_interaction_features(self, df: pd.DataFrame, 
                                  feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction features between specified pairs of features.
        """
        df = df.copy()
        
        for feat1, feat2 in feature_pairs:
            interaction_name = f'{feat1}_{feat2}_interaction'
            df[interaction_name] = df[feat1] * df[feat2]
            
        return df
