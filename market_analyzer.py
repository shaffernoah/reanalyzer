import pandas as pd
import requests
from typing import Dict, List, Optional
import json
from datetime import datetime
import census
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import base64
from datetime import datetime, timedelta

class MarketAnalyzer:
    def __init__(self, census_api_key: Optional[str] = None):
        """
        Initialize MarketAnalyzer with API keys.
        Get your Census API key from: https://api.census.gov/data/key_signup.html
        """
        self.census_api_key = census_api_key
        if census_api_key:
            self.c = census.Census(census_api_key)
        self.data_path = os.path.join(os.path.dirname(__file__), "data", "Zillow Data", "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv")  # Path to Zillow ZHVI data
    
    def _calculate_market_health(self, appreciation: float, volatility: float, momentum: float) -> float:
        """
        Calculate overall market health score (0-100) based on key metrics.
        
        Args:
            appreciation: Year-over-year price appreciation rate
            volatility: Price volatility measure
            momentum: Recent price momentum
            
        Returns:
            float: Market health score from 0-100
        """
        # Appreciation score (0-40 points)
        appreciation_score = min(max(appreciation, 0) * 2, 40)
        
        # Volatility score (0-30 points, lower volatility is better)
        volatility_score = max(30 - volatility * 2, 0)
        
        # Momentum score (0-30 points)
        momentum_score = min(max(momentum, -15) * 1.5 + 22.5, 30)
        
        # Total score
        return round(appreciation_score + volatility_score + momentum_score, 1)

    def analyze_market(self, address: str, city: str, state: str, zip_code: Optional[str] = None) -> Dict:
        """
        Analyze market conditions using Zillow Home Value Index data.
        
        Args:
            address: Street address
            city: City name
            state: State abbreviation
            zip_code: Optional ZIP code (not used currently)
            
        Returns:
            Dict containing market analysis metrics
        """
        try:
            # Read the data file
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Zillow data file ({self.data_path}) not found")

            # Read data
            df = pd.read_csv(self.data_path)
            
            # Create metro area string in the same format as the data
            metro_area = f"{city}, {state}"
            
            # Filter for specific metro area
            metro_data = df[df['RegionName'] == metro_area]
            if metro_data.empty:
                raise ValueError(f"No data found for metro area {metro_area}")
            
            # Get date columns (YYYY-MM format)
            date_columns = [col for col in df.columns if isinstance(col, str) and col[0].isdigit()]
            date_columns.sort()
            
            if len(date_columns) < 12:  # Require at least 12 months of data
                raise ValueError("Insufficient historical data")
            
            # Get recent price history
            recent_columns = date_columns[-24:]  # Last 24 months
            price_history = metro_data[recent_columns]
            recent_prices = price_history.iloc[0]
            
            # Convert to numeric and handle missing values
            recent_prices = pd.to_numeric(recent_prices, errors='coerce')
            recent_prices = recent_prices.ffill()
            recent_prices = recent_prices.values
            
            # Calculate key metrics
            current_price = recent_prices[-1]
            year_ago_price = recent_prices[-13] if len(recent_prices) >= 13 else recent_prices[0]
            
            # Price changes
            yoy_appreciation = ((current_price / year_ago_price) - 1) * 100
            
            # Volatility (standard deviation of monthly changes)
            monthly_changes = np.diff(recent_prices) / recent_prices[:-1]
            volatility = np.std(monthly_changes) * 100
            
            # Calculate momentum (recent 6-month trend vs previous 6-month trend)
            recent_6m = recent_prices[-6:]
            prev_6m = recent_prices[-12:-6]
            recent_trend = (recent_6m[-1] / recent_6m[0] - 1) * 100
            prev_trend = (prev_6m[-1] / prev_6m[0] - 1) * 100
            momentum = recent_trend - prev_trend
            
            # Calculate market health score
            market_health = self._calculate_market_health(yoy_appreciation, volatility, momentum)
            
            results = {
                'address': f"{address}, {city}, {state} {zip_code if zip_code else ''}",
                'metro_area': metro_area,
                'current_price': round(current_price, 2),
                'yoy_appreciation': round(yoy_appreciation, 2),
                'volatility': round(volatility, 2),
                'momentum': round(momentum, 2),
                'market_health': market_health,
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }
            
            # Create visualization
            html_content = self.create_visualization(results, price_history)
            
            # Save the HTML report
            report_path = os.path.join(os.path.dirname(__file__), 'reports', f'market_analysis_{city}_{state}.html')
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, 'w') as f:
                f.write(html_content)
                
            results['report_path'] = report_path
            return results
            
        except Exception as e:
            raise Exception(f"Error analyzing market: {str(e)}")

    def create_visualization(self, results: Dict, price_history: pd.DataFrame) -> str:
        """
        Create an HTML visualization of the market analysis results.
        
        Args:
            results: Dictionary containing market analysis results
            price_history: DataFrame containing historical price data
            
        Returns:
            str: HTML content
        """
        # Create price trend figure
        fig1 = go.Figure()
        dates = pd.to_datetime(price_history.columns)
        prices = price_history.iloc[0].values
        
        fig1.add_trace(
            go.Scatter(x=dates, y=prices, name='Home Price',
                      line=dict(color='blue'))
        )
        
        fig1.update_layout(
            title="Home Price Trend",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400
        )
        
        # Create YoY changes figure
        fig2 = go.Figure()
        yoy_changes = []
        for i in range(12, len(prices)):
            yoy_change = ((prices[i] / prices[i-12]) - 1) * 100
            yoy_changes.append(yoy_change)
        
        fig2.add_trace(
            go.Scatter(x=dates[12:], y=yoy_changes, name='YoY Change',
                      line=dict(color='green'))
        )
        
        fig2.update_layout(
            title="Year-over-Year Price Changes",
            xaxis_title="Date",
            yaxis_title="YoY Change (%)",
            height=400
        )
        
        # Create momentum figure
        fig3 = go.Figure()
        momentum = []
        for i in range(6, len(prices)):
            mom = ((prices[i] / prices[i-6]) - 1) * 100
            momentum.append(mom)
            
        fig3.add_trace(
            go.Scatter(x=dates[6:], y=momentum, name='6-Month Momentum',
                      line=dict(color='orange'))
        )
        
        fig3.update_layout(
            title="Price Momentum",
            xaxis_title="Date",
            yaxis_title="6-Month Change (%)",
            height=400
        )
        
        # Create gauge chart
        fig4 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=results['market_health'],
            title={'text': "Market Health Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "red"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "green"}
                ]
            }
        ))
        
        fig4.update_layout(
            height=400
        )
        
        # Create HTML content
        html_content = f"""
        <html>
        <head>
            <title>Market Analysis Report - {results['metro_area']}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .metrics {{
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .metric-card {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #0066cc;
                }}
                .metric-label {{
                    color: #666;
                    margin-top: 5px;
                }}
                .charts-grid {{
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Real Estate Market Analysis Report</h1>
                    <p>Generated on {results['last_updated']}</p>
                </div>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value">${results['current_price']:,.2f}</div>
                        <div class="metric-label">Current Median Price</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{results['yoy_appreciation']}%</div>
                        <div class="metric-label">Year-over-Year Appreciation</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{results['market_health']}/100</div>
                        <div class="metric-label">Market Health Score</div>
                    </div>
                </div>
                
                <div class="charts-grid">
                    <div>{fig1.to_html(full_html=False, include_plotlyjs=False)}</div>
                    <div>{fig2.to_html(full_html=False, include_plotlyjs=False)}</div>
                    <div>{fig3.to_html(full_html=False, include_plotlyjs=False)}</div>
                    <div>{fig4.to_html(full_html=False, include_plotlyjs=False)}</div>
                </div>
                
                <div style="margin-top: 30px;">
                    <h3>Analysis Summary</h3>
                    <p>Property: {results['address']}</p>
                    <p>Metro Area: {results['metro_area']}</p>
                    <p>Price Volatility: {results['volatility']}%</p>
                    <p>Price Momentum: {results['momentum']}%</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content

    def get_location_metrics(self, zip_code: str) -> Dict:
        """
        Fetch demographic and economic data for a given ZIP code using Census API.
        """
        if not self.census_api_key:
            return {"error": "Census API key required for location metrics"}
        
        try:
            # Get the latest ACS 5-year estimates
            acs5_data = self.c.acs5.zipcode(
                ['B01003_001E',  # Total population
                 'B19013_001E',  # Median household income
                 'B25077_001E',  # Median home value
                 'B25064_001E',  # Median gross rent
                 'B25002_001E',  # Total housing units
                 'B25002_003E'   # Vacant housing units
                ],
                zip_code
            )
            
            if not acs5_data:
                return {"error": "No census data available for this ZIP code"}
            
            data = acs5_data[0]
            vacancy_rate = (data['B25002_003E'] / data['B25002_001E']) * 100 if data['B25002_001E'] else 0
            
            return {
                "population": data['B01003_001E'],
                "median_household_income": data['B19013_001E'],
                "median_home_value": data['B25077_001E'],
                "median_gross_rent": data['B25064_001E'],
                "vacancy_rate": round(vacancy_rate, 2),
                "location_score": self._calculate_location_score(data)
            }
            
        except Exception as e:
            return {"error": f"Error fetching census data: {str(e)}"}

    def _calculate_location_score(self, census_data: Dict) -> float:
        """
        Calculate location score (0-100) based on census data.
        """
        # Simplified scoring model - can be enhanced with more metrics
        scores = []
        
        # Income score (up to 40 points)
        if census_data['B19013_001E']:
            income_score = min(census_data['B19013_001E'] / 100000 * 40, 40)
            scores.append(income_score)
        
        # Vacancy rate score (up to 30 points)
        if census_data['B25002_001E'] and census_data['B25002_003E']:
            vacancy_rate = (census_data['B25002_003E'] / census_data['B25002_001E']) * 100
            vacancy_score = max(30 - vacancy_rate * 3, 0)
            scores.append(vacancy_score)
        
        # Population score (up to 30 points)
        if census_data['B01003_001E']:
            pop_score = min(census_data['B01003_001E'] / 50000 * 30, 30)
            scores.append(pop_score)
        
        return round(sum(scores) / len(scores) if scores else 0, 2)

    def analyze_investment_potential(self, zip_code: str) -> Dict:
        """
        Comprehensive analysis combining market and location metrics.
        """
        market_data = self.analyze_market("", "", "", zip_code)
        location_data = self.get_location_metrics(zip_code)
        
        if "error" in market_data or "error" in location_data:
            return {**market_data, **location_data}
        
        # Calculate overall investment score
        market_weight = 0.6
        location_weight = 0.4
        investment_score = (
            market_data["market_health"] * market_weight +
            location_data["location_score"] * location_weight
        )
        
        return {
            "investment_score": round(investment_score, 2),
            "market_metrics": market_data,
            "location_metrics": location_data,
            "recommendation": self._get_recommendation(investment_score)
        }

    def _get_recommendation(self, score: float) -> str:
        """Generate investment recommendation based on score."""
        if score >= 80:
            return "Strong Buy - Excellent market conditions and location metrics"
        elif score >= 60:
            return "Buy - Good investment potential with manageable risks"
        elif score >= 40:
            return "Hold - Average market conditions, consider other factors"
        else:
            return "Caution - High risk market conditions, detailed due diligence recommended"

def main():
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python market_analyzer.py <address> <city> <state> [zip_code]")
        sys.exit(1)
        
    address = sys.argv[1]
    city = sys.argv[2]
    state = sys.argv[3]
    zip_code = sys.argv[4] if len(sys.argv) > 4 else None
    
    analyzer = MarketAnalyzer()
    try:
        results = analyzer.analyze_market(address, city, state, zip_code)
        print("\nMarket Analysis Results:")
        print("-----------------------")
        print(f"Property: {results['address']}")
        print(f"Metro Area: {results['metro_area']}")
        print(f"Current Metro Area Median Price: ${results['current_price']:,.2f}")
        print(f"Year-over-Year Appreciation: {results['yoy_appreciation']}%")
        print(f"Price Volatility: {results['volatility']}%")
        print(f"Price Momentum: {results['momentum']}%")
        print(f"Market Health Score: {results['market_health']}/100")
        print(f"Last Updated: {results['last_updated']}")
        print(f"\nDetailed report saved to: {results['report_path']}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
