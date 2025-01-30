"""
Enhanced Zillow market analyzer with advanced analytics capabilities.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analyze_zillow import ZillowAnalyzer
from advanced_analytics.data_quality import DataQualityChecker
from advanced_analytics.feature_engineering import FeatureEngineer
from advanced_analytics.time_series_analysis import TimeSeriesAnalyzer
from advanced_analytics.predictive_modeling import MarketPredictor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedZillowAnalyzer(ZillowAnalyzer):
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        
        # Initialize advanced analytics components
        self.quality_checker = DataQualityChecker()
        self.feature_engineer = FeatureEngineer()
        self.ts_analyzer = TimeSeriesAnalyzer()
        self.predictor = MarketPredictor()
        
        # Store analysis results
        self.quality_reports: Dict[str, Dict] = {}
        self.feature_importance: Dict[str, pd.Series] = {}
        self.forecasts: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
        self.model_metrics: Dict[str, Dict[str, float]] = {}
        
    def enhanced_pipeline(self, forecast_periods: int = 12):
        """Run the enhanced analysis pipeline with advanced analytics."""
        try:
            # Run basic pipeline first
            super().run_pipeline()
            
            # Process each dataset with advanced analytics
            for name, df in self.data.items():
                logger.info(f"Running advanced analytics for {name}")
                
                # 1. Data Quality Analysis
                self._run_quality_analysis(name, df)
                
                # 2. Feature Engineering
                df_engineered = self._run_feature_engineering(name, df)
                
                # 3. Time Series Analysis
                self._run_time_series_analysis(name, df_engineered)
                
                # 4. Predictive Modeling
                self._run_predictive_modeling(name, df_engineered, forecast_periods)
                
                # 5. Generate Visualizations
                self._generate_visualizations(name, df_engineered)
                
            logger.info("Enhanced pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Error in enhanced pipeline: {str(e)}")
            raise
            
    def _run_quality_analysis(self, name: str, df: pd.DataFrame):
        """Run comprehensive data quality analysis."""
        logger.info(f"Running data quality analysis for {name}")
        
        # Check data quality
        quality_report = self.quality_checker.check_data_quality(df)
        self.quality_reports[name] = quality_report
        
        # Detect and log outliers
        outliers = self.quality_checker.detect_outliers(df)
        logger.info(f"Found {outliers.sum().sum()} outliers in {name}")
        
        # Validate time series
        is_valid, issues = self.quality_checker.validate_time_series(df, 'date')
        if not is_valid:
            logger.warning(f"Time series issues in {name}: {issues}")
            
    def _run_feature_engineering(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations."""
        logger.info(f"Running feature engineering for {name}")
        
        # Create time features
        df = self.feature_engineer.create_time_features(df, 'date')
        
        # Calculate rolling statistics
        df = self.feature_engineer.calculate_rolling_statistics(
            df, 'value', windows=[3, 6, 12]
        )
        
        # Calculate market indicators
        df = self.feature_engineer.calculate_market_indicators(df, 'value')
        
        # Scale features
        exclude_cols = ['date', 'RegionID', 'RegionName', 'value']
        df = self.feature_engineer.scale_features(df, exclude_columns=exclude_cols)
        
        return df
        
    def _run_time_series_analysis(self, name: str, df: pd.DataFrame):
        """Perform advanced time series analysis."""
        logger.info(f"Running time series analysis for {name}")
        
        # Decompose time series
        components = self.ts_analyzer.decompose_series(df['value'])
        
        # Detect seasonality
        seasonality = self.ts_analyzer.detect_seasonality(df['value'])
        logger.info(f"Seasonality strengths for {name}: {seasonality}")
        
        # Test stationarity
        is_stationary, test_results = self.ts_analyzer.test_stationarity(df['value'])
        logger.info(f"Stationarity test for {name}: {'Stationary' if is_stationary else 'Non-stationary'}")
        
        # Cluster regions if multiple regions exist
        if 'RegionName' in df.columns and df['RegionName'].nunique() > 1:
            labels, cluster_stats = self.ts_analyzer.cluster_regions(
                df.pivot(index='date', columns='RegionName', values='value')
            )
            logger.info(f"Region clustering stats for {name}:\n{cluster_stats}")
            
    def _run_predictive_modeling(self, name: str, df: pd.DataFrame, 
                               forecast_periods: int):
        """Train and evaluate predictive models."""
        logger.info(f"Running predictive modeling for {name}")
        
        # Prepare features for prediction
        target_col = 'value'
        feature_cols = [col for col in df.columns 
                       if col not in ['date', 'RegionID', 'RegionName', target_col]]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = self.predictor.train_test_split(
            pd.concat([X, y], axis=1), target_col
        )
        
        # Train models
        rf_model = self.predictor.train_random_forest(X_train, y_train)
        gb_model = self.predictor.train_gradient_boosting(X_train, y_train)
        
        # Evaluate models
        for model_name in ['random_forest', 'gradient_boosting']:
            metrics = self.predictor.evaluate_model(model_name, X_test, y_test)
            self.model_metrics[f"{name}_{model_name}"] = metrics
            
        # Get feature importance
        self.feature_importance[name] = self.predictor.feature_importance('random_forest')
        
        # Generate forecasts
        X_future = X_test.copy()  # In practice, you'd prepare future feature values
        predictions, conf_intervals = self.predictor.forecast_future(
            'random_forest', X_future, return_conf_int=True
        )
        self.forecasts[name] = (predictions, conf_intervals)
        
    def _generate_visualizations(self, name: str, df: pd.DataFrame):
        """Generate comprehensive visualizations."""
        logger.info(f"Generating visualizations for {name}")
        
        # 1. Time Series Plot with Trend
        fig = make_subplots(rows=3, cols=1, 
                           subplot_titles=('Time Series with Trend', 
                                         'Feature Importance',
                                         'Model Performance Comparison'))
        
        # Add time series plot
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['value'], 
                      name='Actual', mode='lines'),
            row=1, col=1
        )
        
        # Add trend line
        trend = df['value'].rolling(window=12).mean()
        fig.add_trace(
            go.Scatter(x=df['date'], y=trend, 
                      name='12-Month Trend', line=dict(dash='dash')),
            row=1, col=1
        )
        
        # 2. Feature Importance Plot
        importance = self.feature_importance.get(name)
        if importance is not None:
            fig.add_trace(
                go.Bar(x=importance.index[:10], y=importance.values[:10],
                      name='Feature Importance'),
                row=2, col=1
            )
        
        # 3. Model Performance Comparison
        metrics = pd.DataFrame([
            self.model_metrics.get(f"{name}_random_forest", {}),
            self.model_metrics.get(f"{name}_gradient_boosting", {})
        ], index=['Random Forest', 'Gradient Boosting'])
        
        for metric in metrics.columns:
            fig.add_trace(
                go.Bar(x=metrics.index, y=metrics[metric],
                      name=metric),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(height=1200, width=1000, title=f'Analysis Results for {name}',
                         showlegend=True)
        
        # Save plot
        fig.write_html(f'analysis_results_{name}.html')
        
    def get_analysis_summary(self, name: str) -> Dict:
        """Get a comprehensive summary of all analyses for a dataset."""
        return {
            'quality_report': self.quality_reports.get(name),
            'model_metrics': {k: v for k, v in self.model_metrics.items() 
                            if k.startswith(name)},
            'top_features': self.feature_importance.get(name).head(),
            'forecast_summary': {
                'mean_forecast': self.forecasts.get(name)[0].mean(),
                'forecast_std': self.forecasts.get(name)[0].std()
            } if name in self.forecasts else None
        }

def main():
    """Run the enhanced Zillow market analyzer."""
    try:
        # Initialize with correct data directory
        data_dir = "/Users/noahshaffer/Desktop/CascadeProjects/real_estate_analyzer/data/Zillow Data"
        analyzer = EnhancedZillowAnalyzer(data_dir)
        analyzer.enhanced_pipeline()
        
        # Print summary for each dataset
        for name in analyzer.data.keys():
            summary = analyzer.get_analysis_summary(name)
            logger.info(f"\nAnalysis Summary for {name}:")
            logger.info(f"Quality Metrics: {summary['quality_report']}")
            logger.info(f"Model Performance: {summary['model_metrics']}")
            logger.info(f"Top Features: {summary['top_features']}")
            logger.info(f"Forecast Summary: {summary['forecast_summary']}")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
