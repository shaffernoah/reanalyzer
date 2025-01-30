"""
Advanced time series analysis for real estate market data.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class TimeSeriesAnalyzer:
    def __init__(self):
        self.decomposition_results = {}
        
    def decompose_series(self, series: pd.Series, 
                        period: int = 12) -> Dict[str, pd.Series]:
        """
        Perform STL decomposition of time series.
        """
        # Handle empty series
        if series.empty or series.isna().all():
            return {
                'trend': pd.Series(),
                'seasonal': pd.Series(),
                'residual': pd.Series()
            }
            
        # Convert index to datetime if it's not already
        if not isinstance(series.index, pd.DatetimeIndex):
            try:
                # Try to convert index to datetime
                series.index = pd.to_datetime(series.index)
            except (ValueError, TypeError):
                # If conversion fails, create a dummy datetime index
                logger.warning("Could not convert index to datetime. Creating dummy index.")
                series.index = pd.date_range(start='2000-01-01', periods=len(series), freq='M')
            
        # Handle irregular time intervals by resampling to regular frequency
        if not series.index.is_monotonic_increasing:
            series = series.sort_index()
        
        # Remove duplicate indices by keeping the last value
        series = series[~series.index.duplicated(keep='last')]
        
        # Ensure regular frequency by resampling
        freq = pd.infer_freq(series.index)
        if freq is None:
            # If frequency cannot be inferred, default to monthly
            series = series.resample('M').mean()
            
        # Fill any missing values created by resampling
        series = series.interpolate(method='linear')
            
        # Perform STL decomposition
        try:
            stl = STL(series, period=period)
            result = stl.fit()
            
            components = {
                'trend': result.trend,
                'seasonal': result.seasonal,
                'residual': result.resid
            }
        except Exception as e:
            logger.warning(f"STL decomposition failed: {str(e)}")
            components = {
                'trend': series,
                'seasonal': pd.Series(0, index=series.index),
                'residual': pd.Series(0, index=series.index)
            }
        
        # Store results
        self.decomposition_results = components
        
        return components

    def detect_seasonality(self, series: pd.Series, 
                          periods: List[int] = [3, 6, 12]) -> Dict[int, float]:
        """
        Detect seasonality strength at different periods.
        """
        # Handle empty series
        if series.empty or series.isna().all():
            return {period: 0.0 for period in periods}
            
        # Convert index to datetime if it's not already
        if not isinstance(series.index, pd.DatetimeIndex):
            try:
                # Try to convert index to datetime
                series.index = pd.to_datetime(series.index)
            except (ValueError, TypeError):
                # If conversion fails, create a dummy datetime index
                logger.warning("Could not convert index to datetime. Creating dummy index.")
                series.index = pd.date_range(start='2000-01-01', periods=len(series), freq='M')
            
        # Handle irregular time intervals by resampling to regular frequency
        if not series.index.is_monotonic_increasing:
            series = series.sort_index()
            
        # Remove duplicate indices
        series = series[~series.index.duplicated(keep='last')]
        
        # Ensure regular frequency by resampling
        freq = pd.infer_freq(series.index)
        if freq is None:
            # If frequency cannot be inferred, default to monthly
            series = series.resample('M').mean()
            
        # Fill any missing values created by resampling
        series = series.interpolate(method='linear')
            
        seasonality_strengths = {}
        
        for period in periods:
            if len(series) < period * 2:
                seasonality_strengths[period] = 0.0
                continue
                
            try:
                # Calculate autocorrelation at the given period
                acf = pd.Series(series).autocorr(lag=period)
                seasonality_strengths[period] = acf if not pd.isna(acf) else 0.0
            except Exception as e:
                logger.warning(f"Failed to calculate seasonality for period {period}: {str(e)}")
                seasonality_strengths[period] = 0.0
            
        return seasonality_strengths

    def test_stationarity(self, series: pd.Series) -> Tuple[bool, Dict]:
        """
        Perform Augmented Dickey-Fuller test for stationarity.
        """
        # Handle empty series
        if series.empty or series.isna().all():
            return False, {
                'test_statistic': None,
                'p_value': None,
                'critical_values': {}
            }
            
        # Convert index to datetime if it's not already
        if not isinstance(series.index, pd.DatetimeIndex):
            try:
                # Try to convert index to datetime
                series.index = pd.to_datetime(series.index)
            except (ValueError, TypeError):
                # If conversion fails, create a dummy datetime index
                logger.warning("Could not convert index to datetime. Creating dummy index.")
                series.index = pd.date_range(start='2000-01-01', periods=len(series), freq='M')
            
        # Handle irregular time intervals by resampling to regular frequency
        if not series.index.is_monotonic_increasing:
            series = series.sort_index()
            
        # Remove duplicate indices
        series = series[~series.index.duplicated(keep='last')]
        
        # Ensure regular frequency by resampling
        freq = pd.infer_freq(series.index)
        if freq is None:
            # If frequency cannot be inferred, default to monthly
            series = series.resample('M').mean()
            
        # Fill any missing values created by resampling
        series = series.interpolate(method='linear')
            
        # Drop NA values and ensure enough data points
        clean_series = series.dropna()
        if len(clean_series) < 3:  # Minimum required for ADF test
            return False, {
                'test_statistic': None,
                'p_value': None,
                'critical_values': {}
            }
            
        try:
            result = adfuller(clean_series)
            
            test_results = {
                'test_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4]
            }
            
            # Series is considered stationary if p-value < 0.05
            is_stationary = result[1] < 0.05
            
            return is_stationary, test_results
            
        except Exception as e:
            logger.warning(f"Stationarity test failed: {str(e)}")
            return False, {
                'test_statistic': None,
                'p_value': None,
                'critical_values': {}
            }

    def cluster_regions(self, df: pd.DataFrame, 
                       n_clusters: int = 5,
                       method: str = 'ward') -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Perform hierarchical clustering on regions based on price trends.
        """
        # Handle empty DataFrame
        if df.empty:
            return np.array([]), pd.DataFrame()
            
        # Prepare data for clustering
        X = df.values.T  # Transpose to cluster regions
        
        # Check if we have enough data points
        if X.shape[0] < 2:  # Need at least 2 regions to cluster
            return np.array([0]), pd.DataFrame({'Cluster_0': {
                'size': 1,
                'mean_price': df.mean().mean() if not df.empty else np.nan,
                'std_price': df.std().mean() if not df.empty else np.nan,
                'trend': df.mean().diff().mean() if not df.empty else np.nan
            }})
            
        # Adjust number of clusters if we have fewer regions
        n_clusters = min(n_clusters, X.shape[0])
        
        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(n_clusters=n_clusters, 
                                           linkage=method)
        labels = clustering.fit_predict(X)
        
        # Calculate cluster characteristics
        cluster_stats = pd.DataFrame()
        for i in range(n_clusters):
            mask = labels == i
            cluster_data = df.iloc[:, mask]
            
            cluster_stats[f'Cluster_{i}'] = {
                'size': mask.sum(),
                'mean_price': cluster_data.mean().mean(),
                'std_price': cluster_data.std().mean(),
                'trend': cluster_data.mean().diff().mean()
            }
            
        return labels, cluster_stats

    def calculate_trend_metrics(self, series: pd.Series, 
                              window: int = 12) -> Dict[str, float]:
        """
        Calculate various trend metrics for the time series.
        """
        metrics = {
            'overall_trend': series.diff().mean(),
            'recent_trend': series.diff().tail(window).mean(),
            'acceleration': series.diff().diff().mean(),
            'volatility': series.std() / series.mean(),
            'peak_value': series.max(),
            'trough_value': series.min(),
            'peak_to_trough_ratio': series.max() / series.min()
        }
        
        return metrics

    def plot_decomposition(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot the decomposition components.
        """
        if not self.decomposition_results:
            raise ValueError("No decomposition results available. Run decompose_series first.")
            
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        # Plot original data
        axes[0].plot(self.decomposition_results['trend'] + 
                    self.decomposition_results['seasonal'] + 
                    self.decomposition_results['residual'])
        axes[0].set_title('Original Time Series')
        
        # Plot components
        axes[1].plot(self.decomposition_results['trend'])
        axes[1].set_title('Trend')
        
        axes[2].plot(self.decomposition_results['seasonal'])
        axes[2].set_title('Seasonal')
        
        axes[3].plot(self.decomposition_results['residual'])
        axes[3].set_title('Residual')
        
        plt.tight_layout()
        return fig
