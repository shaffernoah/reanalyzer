"""
Test script for enhanced Zillow analyzer
"""
import os
import logging
from enhanced_analyzer import EnhancedZillowAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Initialize analyzer with data directory
        data_dir = os.path.join(os.getcwd(), 'data')
        logger.info(f"Using data directory: {data_dir}")
        
        analyzer = EnhancedZillowAnalyzer(data_dir)
        
        # Run enhanced pipeline
        analyzer.enhanced_pipeline()
        
        # Print summaries for each dataset
        for name in analyzer.data.keys():
            summary = analyzer.get_analysis_summary(name)
            logger.info(f"\nAnalysis Summary for {name}:")
            logger.info(f"Quality Metrics: {summary['quality_report']}")
            logger.info(f"Model Performance: {summary['model_metrics']}")
            logger.info(f"Top Features: {summary['top_features']}")
            logger.info(f"Forecast Summary: {summary['forecast_summary']}")
            
    except Exception as e:
        logger.error(f"Error in test script: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
