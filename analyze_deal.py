from real_estate_analyzer import analyze_property
from market_analyzer import MarketAnalyzer

def print_analysis(results):
    """Pretty print the analysis results."""
    print("\nReal Estate Investment Analysis")
    print("=" * 50)
    
    # Print financial metrics
    print("\nFinancial Analysis:")
    print("-" * 30)
    for metric, value in results.items():
        if metric != "Market Analysis":
            print(f"{metric}: ${value:,.2f}")
    
    # Print market analysis if available
    if "Market Analysis" in results:
        market_data = results["Market Analysis"]
        print("\nMarket Analysis")
        print("-" * 30)
        
        if "error" in market_data:
            print(f"Market Analysis Error: {market_data['error']}")
            return
            
        print(f"Location: {market_data.get('location', 'Unknown')}")
        print(f"Data Timespan: {market_data.get('data_timespan', 'Unknown')}")
        
        print("\nMarket Metrics:")
        print(f"Current Median Price: ${market_data['market_metrics']['current_median_price']:,.2f}")
        print(f"1-Year Appreciation: {market_data['market_metrics']['1yr_appreciation']}%")
        print(f"2-Year Appreciation: {market_data['market_metrics']['2yr_appreciation']}%")
        print(f"Price Volatility: {market_data['market_metrics']['price_volatility']}%")
        print(f"Price Momentum: {market_data['market_metrics']['price_momentum']}%")
        print(f"Market Health Score: {market_data['market_metrics']['market_health']}/100")
        
        print(f"\nInvestment Score: {market_data['investment_score']}")
        print(f"Recommendation: {market_data['recommendation']}")

def main():
    # Example property in Panama City Beach, FL
    results = analyze_property(
        purchase_price=400000,  # $400K purchase price
        monthly_rent=2500,      # $2500/month rent
        down_payment_percent=0.20,  # 20% down payment
        interest_rate=0.065,     # 6.5% interest rate
        zip_code="32413",       # Panama City Beach, FL
        property_tax_rate=0.0125,  # 1.25% property tax
        insurance_cost=1200,     # $1200/year insurance
        maintenance_percent=0.01,  # 1% maintenance
        vacancy_rate=0.05        # 5% vacancy rate
    )
    
    print_analysis(results)

if __name__ == "__main__":
    main()
