from market_analyzer import MarketAnalyzer
import json

def print_analysis(data: dict, zip_code: str):
    """Pretty print the market analysis results."""
    print(f"\nAnalysis for ZIP Code: {zip_code}")
    print("=" * 50)
    
    if "error" in data:
        print(f"Error: {data['error']}")
        if "details" in data:
            print("\nError Details:")
            print(data["details"])
        return
    
    # Print location info
    print(f"Location: {data.get('location', 'Unknown')}")
    print(f"Data Timespan: {data.get('data_timespan', 'Unknown')}")
    print("\nMarket Metrics:")
    print("-" * 30)
    print(f"Current Median Price: ${data['current_median_price']:,.2f}")
    print(f"1-Year Appreciation: {data['1yr_appreciation']}%")
    print(f"2-Year Appreciation: {data['2yr_appreciation']}%")
    print(f"Price Volatility: {data['price_volatility']}%")
    print(f"Price Momentum: {data['price_momentum']}%")
    print(f"Market Health Score: {data['market_health']}/100")

def main():
    # Initialize the analyzer
    analyzer = MarketAnalyzer()
    
    # Test with a few different ZIP codes
    test_zips = [
        "94025",  # Menlo Park, CA
        "10001",  # New York, NY
        "98004",  # Bellevue, WA
        "60614",  # Chicago, IL
        "33139"   # Miami Beach, FL
    ]
    
    for zip_code in test_zips:
        market_data = analyzer.get_market_metrics(zip_code)
        print_analysis(market_data, zip_code)

if __name__ == "__main__":
    main()
