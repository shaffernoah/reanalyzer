import dataclasses
from typing import Dict, Optional
from market_analyzer import MarketAnalyzer

@dataclasses.dataclass
class RealEstateProperty:
    purchase_price: float
    monthly_rent: float
    property_tax_rate: float = 0.012  # Annual rate, default 1.2%
    insurance_cost: float = 1200  # Annual insurance cost
    maintenance_percent: float = 0.01  # 1% of purchase price annually
    vacancy_rate: float = 0.05  # 5% vacancy rate
    closing_costs: float = 0.03  # 3% of purchase price
    zip_code: str = ""

    def calculate_monthly_mortgage_payment(self, down_payment_percent: float, interest_rate: float, loan_term_years: int = 30) -> float:
        """Calculate the monthly mortgage payment."""
        loan_amount = self.purchase_price * (1 - down_payment_percent)
        monthly_rate = interest_rate / 12
        num_payments = loan_term_years * 12
        
        if interest_rate == 0:
            return loan_amount / num_payments
            
        mortgage_payment = (loan_amount * 
                          (monthly_rate * (1 + monthly_rate)**num_payments) / 
                          ((1 + monthly_rate)**num_payments - 1))
        return mortgage_payment

    def analyze_deal(self, down_payment_percent: float, interest_rate: float, market_analyzer: Optional[MarketAnalyzer] = None) -> Dict[str, float]:
        """Analyze the real estate deal and return key metrics."""
        # Get basic financial metrics
        basic_metrics = self._calculate_financial_metrics(down_payment_percent, interest_rate)
        
        # If market analyzer is provided, get market and location data
        if market_analyzer and self.zip_code:
            market_analysis = market_analyzer.analyze_investment_potential(self.zip_code)
            return {
                **basic_metrics,
                "Market Analysis": market_analysis
            }
        
        return basic_metrics
    
    def _calculate_financial_metrics(self, down_payment_percent: float, interest_rate: float) -> Dict[str, float]:
        """Calculate basic financial metrics."""
        # Monthly costs
        mortgage_payment = self.calculate_monthly_mortgage_payment(down_payment_percent, interest_rate)
        monthly_property_tax = (self.purchase_price * self.property_tax_rate) / 12
        monthly_insurance = self.insurance_cost / 12
        monthly_maintenance = (self.purchase_price * self.maintenance_percent) / 12
        
        # Monthly income
        effective_rent = self.monthly_rent * (1 - self.vacancy_rate)
        
        # Calculate monthly cash flow
        total_monthly_expenses = (mortgage_payment + 
                                monthly_property_tax + 
                                monthly_insurance + 
                                monthly_maintenance)
        monthly_cash_flow = effective_rent - total_monthly_expenses
        
        # Calculate ROI metrics
        down_payment = self.purchase_price * down_payment_percent
        total_investment = down_payment + (self.purchase_price * self.closing_costs)
        
        annual_cash_flow = monthly_cash_flow * 12
        cash_on_cash_roi = (annual_cash_flow / total_investment) * 100
        
        # Calculate cap rate (NOI / Purchase Price)
        annual_noi = (effective_rent * 12) - (
            (monthly_property_tax + monthly_insurance + monthly_maintenance) * 12
        )
        cap_rate = (annual_noi / self.purchase_price) * 100
        
        return {
            "Monthly Mortgage Payment": round(mortgage_payment, 2),
            "Monthly Cash Flow": round(monthly_cash_flow, 2),
            "Annual Cash Flow": round(annual_cash_flow, 2),
            "Cash on Cash ROI %": round(cash_on_cash_roi, 2),
            "Cap Rate %": round(cap_rate, 2),
            "Total Investment Required": round(total_investment, 2)
        }

def analyze_property(
    purchase_price: float,
    monthly_rent: float,
    down_payment_percent: float,
    interest_rate: float,
    zip_code: str = "",
    census_api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Analyze a real estate property investment.
    
    Args:
        purchase_price: The purchase price of the property
        monthly_rent: Expected monthly rental income
        down_payment_percent: Down payment as a decimal (e.g., 0.20 for 20%)
        interest_rate: Annual interest rate as a decimal (e.g., 0.05 for 5%)
        zip_code: Property ZIP code for market analysis
        census_api_key: Optional Census API key for detailed market analysis
        **kwargs: Additional property parameters
    
    Returns:
        Dictionary containing various investment metrics and market analysis
    """
    property_params = {
        "purchase_price": purchase_price,
        "monthly_rent": monthly_rent,
        "zip_code": zip_code,
        **kwargs
    }
    
    property = RealEstateProperty(**property_params)
    market_analyzer = MarketAnalyzer(census_api_key) if zip_code else None
    return property.analyze_deal(down_payment_percent, interest_rate, market_analyzer)

# Example usage
if __name__ == "__main__":
    # You'll need to get a Census API key from: https://api.census.gov/data/key_signup.html
    CENSUS_API_KEY = None  # Replace with your API key
    
    results = analyze_property(
        purchase_price=300000,
        monthly_rent=2500,
        down_payment_percent=0.20,  # 20% down payment
        interest_rate=0.065,  # 6.5% interest rate
        zip_code="94105",  # Example ZIP code for San Francisco
        census_api_key=CENSUS_API_KEY
    )
    
    print("\nReal Estate Investment Analysis")
    print("=" * 30)
    
    # Print financial metrics
    for metric, value in results.items():
        if metric != "Market Analysis":
            print(f"{metric}: ${value:,.2f}")
    
    # Print market analysis if available
    if "Market Analysis" in results:
        market_data = results["Market Analysis"]
        print("\nMarket Analysis")
        print("-" * 30)
        print(f"Investment Score: {market_data['investment_score']}")
        print(f"Recommendation: {market_data['recommendation']}")
        
        print("\nMarket Metrics:")
        for metric, value in market_data['market_metrics'].items():
            if metric != "error" and metric != "market_health":
                print(f"- {metric}: {value}")
        
        print("\nLocation Metrics:")
        for metric, value in market_data['location_metrics'].items():
            if metric != "location_score":
                print(f"- {metric}: {value}")
