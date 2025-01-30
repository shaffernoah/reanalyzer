"""
Predictive modeling capabilities for real estate market analysis.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

class MarketPredictor:
    def __init__(self):
        self.models = {}
        self.model_metrics = {}
        self.imputer = SimpleImputer(strategy='mean')
        
    def train_test_split(self, df: pd.DataFrame, 
                        target_col: str,
                        test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                        pd.Series, pd.Series]:
        """
        Split data into training and testing sets, preserving time order.
        """
        # Handle empty DataFrame
        if df.empty:
            empty_df = pd.DataFrame(columns=[str(col) for col in df.columns if col != target_col])
            empty_series = pd.Series(dtype=float)
            return empty_df, empty_df, empty_series, empty_series
            
        # Convert column names to strings
        df = df.copy()
        df.columns = df.columns.astype(str)
        
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if str(target_col) not in numeric_cols:
            numeric_cols = numeric_cols.append(pd.Index([str(target_col)]))
            
        # Keep only numeric columns
        df = df[numeric_cols]
        
        split_idx = int(len(df) * (1 - test_size))
        
        X = df.drop(columns=[str(target_col)])
        y = df[str(target_col)]
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test

    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                          n_estimators: int = 100) -> RandomForestRegressor:
        """
        Train a Random Forest model for price prediction.
        """
        # Handle empty data
        if X_train.empty or y_train.empty:
            rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            self.models['random_forest'] = rf_model
            return rf_model
            
        # Ensure all feature names are strings and data is numeric
        X_train = X_train.copy()
        X_train.columns = X_train.columns.astype(str)
        
        # Get numeric columns only
        numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
        X_train = X_train[numeric_cols]
        
        # Handle missing values
        X_train_imputed = pd.DataFrame(self.imputer.fit_transform(X_train),
                                     columns=X_train.columns,
                                     index=X_train.index)
        y_train_imputed = pd.Series(self.imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel(),
                                  index=y_train.index)
        
        rf_model = RandomForestRegressor(n_estimators=n_estimators, 
                                       random_state=42)
        rf_model.fit(X_train_imputed, y_train_imputed)
        
        self.models['random_forest'] = rf_model
        return rf_model

    def train_gradient_boosting(self, X_train: pd.DataFrame, y_train: pd.Series,
                              n_estimators: int = 100) -> GradientBoostingRegressor:
        """
        Train a Gradient Boosting model for price prediction.
        """
        # Handle empty data
        if X_train.empty or y_train.empty:
            gb_model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
            self.models['gradient_boosting'] = gb_model
            return gb_model
            
        # Ensure all feature names are strings and data is numeric
        X_train = X_train.copy()
        X_train.columns = X_train.columns.astype(str)
        
        # Get numeric columns only
        numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
        X_train = X_train[numeric_cols]
        
        # Handle missing values
        X_train_imputed = pd.DataFrame(self.imputer.fit_transform(X_train),
                                     columns=X_train.columns,
                                     index=X_train.index)
        y_train_imputed = pd.Series(self.imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel(),
                                  index=y_train.index)
        
        gb_model = GradientBoostingRegressor(n_estimators=n_estimators,
                                           random_state=42)
        gb_model.fit(X_train_imputed, y_train_imputed)
        
        self.models['gradient_boosting'] = gb_model
        return gb_model

    def evaluate_model(self, model_name: str, X_test: pd.DataFrame, 
                      y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance using multiple metrics.
        """
        # Handle empty data
        if X_test.empty or y_test.empty:
            metrics = {
                'rmse': np.nan,
                'mae': np.nan,
                'mape': np.nan,
                'r2': np.nan
            }
            self.model_metrics[model_name] = metrics
            return metrics
            
        # Ensure all feature names are strings and data is numeric
        X_test = X_test.copy()
        X_test.columns = X_test.columns.astype(str)
        
        # Get numeric columns only
        numeric_cols = X_test.select_dtypes(include=['int64', 'float64']).columns
        X_test = X_test[numeric_cols]
        
        # Handle missing values
        X_test_imputed = pd.DataFrame(self.imputer.transform(X_test),
                                    columns=X_test.columns,
                                    index=X_test.index)
        y_test_imputed = pd.Series(self.imputer.transform(y_test.values.reshape(-1, 1)).ravel(),
                                 index=y_test.index)
        
        model = self.models.get(model_name)
        if model is None:
            logger.warning(f"Model {model_name} not found")
            return {
                'rmse': np.nan,
                'mae': np.nan,
                'mape': np.nan,
                'r2': np.nan
            }
            
        try:
            y_pred = model.predict(X_test_imputed)
            
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_test_imputed, y_pred)),
                'mae': mean_absolute_error(y_test_imputed, y_pred),
                'mape': np.mean(np.abs((y_test_imputed - y_pred) / y_test_imputed)) * 100,
                'r2': r2_score(y_test_imputed, y_pred)
            }
            
            self.model_metrics[model_name] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {
                'rmse': np.nan,
                'mae': np.nan,
                'mape': np.nan,
                'r2': np.nan
            }

    def compare_models(self) -> pd.DataFrame:
        """
        Compare performance metrics across all trained models.
        """
        if not self.model_metrics:
            return pd.DataFrame()
            
        return pd.DataFrame.from_dict(self.model_metrics, orient='index')

    def feature_importance(self, model_name: str) -> pd.Series:
        """
        Get feature importance for tree-based models.
        """
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found.")
            
        # Handle unfit models
        if not hasattr(model, 'feature_importances_') or not hasattr(model, 'feature_names_in_'):
            return pd.Series(dtype=float)
            
        importance = pd.Series(model.feature_importances_,
                             index=model.feature_names_in_,
                             name='importance')
        return importance.sort_values(ascending=False)

    def forecast_future(self, model_name: str, 
                       X_future: pd.DataFrame,
                       return_conf_int: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate forecasts for future data points.
        """
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found.")
            
        # Handle unfit models
        if not hasattr(model, 'feature_importances_') or not hasattr(model, 'feature_names_in_'):
            empty_pred = np.zeros(len(X_future)) if not X_future.empty else np.array([])
            empty_conf = np.zeros((len(X_future), 2)) if not X_future.empty else np.array([])
            return empty_pred, empty_conf if return_conf_int else None
            
        # Ensure all feature names are strings and data is numeric
        X_future = X_future.copy()
        X_future.columns = X_future.columns.astype(str)
        
        # Get numeric columns only
        numeric_cols = X_future.select_dtypes(include=['int64', 'float64']).columns
        X_future = X_future[numeric_cols]
        
        # Handle missing values
        X_future_imputed = pd.DataFrame(self.imputer.transform(X_future),
                                      columns=X_future.columns,
                                      index=X_future.index)
        
        predictions = model.predict(X_future_imputed)
        
        # Simple confidence intervals based on training residuals
        conf_intervals = None
        if return_conf_int and hasattr(model, 'predict'):
            try:
                y_train_pred = model.predict(model.feature_names_in_)
                residuals = np.abs(y_train_pred - model.y_)
                conf_width = np.percentile(residuals, 95)
                conf_intervals = np.vstack([
                    predictions - conf_width,
                    predictions + conf_width
                ]).T
            except:
                # If confidence interval calculation fails, return None
                conf_intervals = None
            
        return predictions, conf_intervals
