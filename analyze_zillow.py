"""
Core Zillow data analysis functionality.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ZillowDataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_files: Dict[str, pd.DataFrame] = {}
        
    def load_all_files(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV and Excel files from the data directory."""
        try:
            for file_path in self.data_dir.glob('**/*.*'):
                if file_path.suffix.lower() in ['.csv', '.xlsx']:
                    file_name = file_path.stem
                    logger.info(f"Loading {file_name}")
                    
                    if file_path.suffix.lower() == '.csv':
                        self.data_files[file_name] = pd.read_csv(file_path)
                    else:  # Excel file
                        self.data_files[file_name] = pd.read_excel(file_path)
                        
            return self.data_files
        except Exception as e:
            logger.error(f"Error loading files: {str(e)}")
            raise

class DataCleaner:
    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean a single dataframe."""
        try:
            # Create a copy to avoid modifying the original
            df_clean = df.copy()
            
            # Convert date columns to datetime if they're not already
            date_columns = [col for col in df_clean.columns 
                          if isinstance(col, str) and '-' in col]
            for col in date_columns:
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col])
                except:
                    # If conversion fails, try to handle it as a numeric column
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Handle missing values
            df_clean = df_clean.fillna({
                col: df_clean[col].mean() if df_clean[col].dtype.kind in 'fc'
                else df_clean[col].mode()[0]
                for col in df_clean.columns
            })
            
            return df_clean
        except Exception as e:
            logger.error(f"Error cleaning dataframe: {str(e)}")
            raise

class DataProcessor:
    def __init__(self):
        self.processed_data: Dict[str, pd.DataFrame] = {}
        
    def process_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process time series data into a more analysis-friendly format."""
        try:
            # Identify date columns (either string dates or datetime objects)
            date_cols = [col for col in df.columns 
                        if isinstance(col, str) and '-' in col
                        or isinstance(col, pd.Timestamp)]
            
            # Melt the dataframe to convert wide format to long format
            id_vars = [col for col in df.columns if col not in date_cols]
            df_melted = pd.melt(
                df,
                id_vars=id_vars,
                value_vars=date_cols,
                var_name='date',
                value_name='value'
            )
            
            # Convert date column to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df_melted['date']):
                df_melted['date'] = pd.to_datetime(df_melted['date'])
            
            return df_melted
        except Exception as e:
            logger.error(f"Error processing time series: {str(e)}")
            raise
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic metrics for the data."""
        try:
            metrics = {
                'mean': df['value'].mean(),
                'median': df['value'].median(),
                'std': df['value'].std(),
                'min': df['value'].min(),
                'max': df['value'].max()
            }
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise

class ZillowAnalyzer:
    def __init__(self, data_dir: str):
        self.loader = ZillowDataLoader(data_dir)
        self.cleaner = DataCleaner()
        self.processor = DataProcessor()
        self.data: Dict[str, pd.DataFrame] = {}
        
    def run_pipeline(self):
        """Run the complete data processing pipeline."""
        try:
            # Load data
            logger.info("Starting data loading...")
            raw_data = self.loader.load_all_files()
            
            # Clean and process each dataset
            for name, df in raw_data.items():
                logger.info(f"Processing {name}")
                # Clean the data
                cleaned_df = self.cleaner.clean_dataframe(df)
                # Process time series
                processed_df = self.processor.process_time_series(cleaned_df)
                # Store processed data
                self.data[name] = processed_df
                
                # Calculate and log basic metrics
                metrics = self.processor.calculate_metrics(processed_df)
                logger.info(f"Metrics for {name}: {metrics}")
            
            logger.info("Pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    # Initialize and run the pipeline
    data_dir = "/Users/noahshaffer/Desktop/CascadeProjects/real_estate_analyzer/data/Zillow Data"
    analyzer = ZillowAnalyzer(data_dir)
    analyzer.run_pipeline()
    
    # Example: Print summary of processed data
    for name, df in analyzer.data.items():
        print(f"\nDataset: {name}")
        print(f"Shape: {df.shape}")
        print("\nSample data:")
        print(df.head())

if __name__ == "__main__":
    main()
