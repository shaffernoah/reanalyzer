"""
Advanced data quality and integrity checks for real estate market analysis.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class DataQualityChecker:
    def __init__(self):
        self.anomaly_threshold = 3  # Standard deviations for Z-score
        self.missing_threshold = 0.2  # 20% missing data threshold

    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Comprehensive data quality assessment.
        Returns a dictionary of quality metrics for each column.
        """
        quality_report = {}
        
        for column in df.columns:
            metrics = {
                'missing_rate': self._calculate_missing_rate(df[column]),
                'unique_count': df[column].nunique(),
                'data_type': str(df[column].dtype)
            }
            
            if pd.api.types.is_numeric_dtype(df[column]):
                metrics.update(self._numeric_column_checks(df[column]))
            
            quality_report[column] = metrics
            
        return quality_report

    def detect_outliers(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Detect outliers using Z-score method for specified numeric columns.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
            
        outliers_mask = pd.DataFrame()
        for column in columns:
            z_scores = np.abs(stats.zscore(df[column].fillna(df[column].mean())))
            outliers_mask[column] = z_scores > self.anomaly_threshold
            
        return outliers_mask

    def validate_time_series(self, df: pd.DataFrame, date_column: str) -> Tuple[bool, List[str]]:
        """
        Validate time series data for completeness and consistency.
        """
        issues = []
        
        # Check date sorting
        if not df[date_column].is_monotonic_increasing:
            issues.append("Dates are not in chronological order")
            
        # Check for gaps
        date_diff = df[date_column].diff()
        if date_diff.nunique() > 1:
            issues.append("Irregular time intervals detected")
            
        # Check for duplicates
        if df[date_column].duplicated().any():
            issues.append("Duplicate dates found")
            
        return len(issues) == 0, issues

    @staticmethod
    def _calculate_missing_rate(series: pd.Series) -> float:
        """Calculate the rate of missing values in a series."""
        return series.isna().mean()

    @staticmethod
    def _numeric_column_checks(series: pd.Series) -> Dict:
        """Perform statistical checks on numeric columns."""
        return {
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'skewness': series.skew(),
            'kurtosis': series.kurtosis()
        }

    def handle_missing_data(self, df: pd.DataFrame, strategy: str = 'interpolate') -> pd.DataFrame:
        """
        Handle missing data using specified strategy.
        Strategies: 'interpolate', 'mean', 'median', 'forward', 'backward'
        """
        df_clean = df.copy()
        
        if strategy == 'interpolate':
            df_clean = df_clean.interpolate(method='time')
        elif strategy == 'mean':
            df_clean = df_clean.fillna(df_clean.mean())
        elif strategy == 'median':
            df_clean = df_clean.fillna(df_clean.median())
        elif strategy == 'forward':
            df_clean = df_clean.fillna(method='ffill')
        elif strategy == 'backward':
            df_clean = df_clean.fillna(method='bfill')
            
        return df_clean
