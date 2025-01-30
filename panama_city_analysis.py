from real_estate_analyzer import analyze_property
from market_analyzer import MarketAnalyzer
import json
from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

class DetailedDealAnalyzer:
    def __init__(self, deal_data: Dict):
        self.metrics = deal_data['financialMetrics']
        self.capital_stack = deal_data['capitalStack']
        self.cash_flows = deal_data['forecastedCashFlows']
        self.investment_details = deal_data['investmentDetails']
        self.source_and_use = deal_data['sourceAndUseOfFunds']
        self.growth_assumptions = deal_data['growthAssumptions']

    def analyze_capital_structure(self) -> Dict:
        total_capital = (self.capital_stack['newDebt'] + 
                        self.capital_stack['preferredEquity'] + 
                        self.capital_stack['originalEquity'])
        
        return {
            "Total Capital Structure": total_capital,
            "Debt Percentage": (self.capital_stack['newDebt'] / total_capital) * 100,
            "Preferred Equity Percentage": (self.capital_stack['preferredEquity'] / total_capital) * 100,
            "Original Equity Percentage": (self.capital_stack['originalEquity'] / total_capital) * 100
        }

    def analyze_cash_flows(self) -> Dict:
        years = len(self.cash_flows)
        total_noi = sum(cf['netOperatingIncome'] for cf in self.cash_flows)
        avg_noi = total_noi / years
        
        # Calculate NOI growth rate
        noi_growth_rates = []
        for i in range(1, years):
            prev_noi = self.cash_flows[i-1]['netOperatingIncome']
            current_noi = self.cash_flows[i]['netOperatingIncome']
            growth_rate = ((current_noi - prev_noi) / prev_noi) * 100
            noi_growth_rates.append(growth_rate)
        
        return {
            "Average Annual NOI": avg_noi,
            "Average NOI Growth Rate": sum(noi_growth_rates) / len(noi_growth_rates),
            "Total Cash Flow (excl. sale)": sum(cf['cashFlow'] for cf in self.cash_flows),
            "Debt Service Coverage Ratio": avg_noi / self.cash_flows[0]['debtService']
        }

    def calculate_investment_returns(self) -> Dict:
        # Extract final year's net sales proceeds
        exit_proceeds = self.cash_flows[-1].get('netSalesProceeds', 0)
        
        # Calculate cash-on-cash returns for each year
        total_equity = self.capital_stack['preferredEquity'] + self.capital_stack['originalEquity']
        cash_on_cash_returns = [
            (cf['cashFlow'] / total_equity) * 100 
            for cf in self.cash_flows
        ]
        
        return {
            "IRR": self.metrics['IRR'],
            "Equity Multiple": self.metrics['equityMultiple'],
            "Cap Rate": self.metrics['capRate'],
            "Preferred Return": self.metrics['preferredReturn'],
            "Average Cash-on-Cash Return": sum(cash_on_cash_returns) / len(cash_on_cash_returns),
            "Exit Value": exit_proceeds
        }

    def analyze_operating_metrics(self) -> Dict:
        first_year = self.cash_flows[0]
        expense_ratio = (first_year['expenses'] / first_year['grossOperatingIncome']) * 100
        
        return {
            "First Year NOI": first_year['netOperatingIncome'],
            "Operating Expense Ratio": expense_ratio,
            "Revenue Growth Rate": self.growth_assumptions['revenue']['grossPotentialRent'],
            "Expense Growth Rate": np.mean([
                rate for rate in self.growth_assumptions['expenses'].values()
            ])
        }

def millions_formatter(x, pos):
    """Format numbers in millions with $M"""
    return f'${x/1e6:.1f}M'

def create_visualizations(deal_data: Dict, analysis_results: Dict):
    """Create and save visualizations for cash flows and market data."""
    # Set style
    plt.style.use('default')
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. NOI and Cash Flow Progression
    ax1 = plt.subplot(221)
    years = range(1, len(deal_data['forecastedCashFlows']) + 1)
    noi_values = [cf['netOperatingIncome'] for cf in deal_data['forecastedCashFlows']]
    cash_flows = [cf['cashFlow'] for cf in deal_data['forecastedCashFlows']]
    
    ax1.plot(years, noi_values, marker='o', linewidth=2, label='NOI')
    ax1.plot(years, cash_flows, marker='s', linewidth=2, label='Cash Flow')
    ax1.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
    ax1.set_xlabel('Year')
    ax1.set_title('NOI and Cash Flow Progression')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Capital Stack Waterfall
    ax2 = plt.subplot(222)
    capital_components = ['Debt', 'Preferred Equity', 'Original Equity']
    capital_values = [
        deal_data['capitalStack']['newDebt'],
        deal_data['capitalStack']['preferredEquity'],
        deal_data['capitalStack']['originalEquity']
    ]
    
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    bottom = 0
    for i, (value, color) in enumerate(zip(capital_values, colors)):
        ax2.bar(0, value, bottom=bottom, color=color, label=capital_components[i])
        bottom += value
    
    ax2.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
    ax2.set_title('Capital Stack')
    ax2.set_xticks([])
    ax2.legend()
    
    # 3. Operating Metrics Over Time
    ax3 = plt.subplot(223)
    expenses = [cf['expenses'] for cf in deal_data['forecastedCashFlows']]
    revenue = [cf['grossOperatingIncome'] for cf in deal_data['forecastedCashFlows']]
    
    ax3.stackplot(years, [expenses, np.array(revenue) - np.array(expenses)], 
                 labels=['Expenses', 'Net Operating Income'],
                 colors=['#e74c3c', '#2ecc71'])
    ax3.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
    ax3.set_xlabel('Year')
    ax3.set_title('Revenue and Expense Composition')
    ax3.legend()
    
    # 4. Key Metrics Summary
    ax4 = plt.subplot(224)
    metrics = {
        'IRR': deal_data['financialMetrics']['IRR'],
        'Cap Rate': deal_data['financialMetrics']['capRate'],
        'DSCR': analysis_results['cash_flow_analysis']['Debt Service Coverage Ratio'],
        'NOI Growth': analysis_results['cash_flow_analysis']['Average NOI Growth Rate']
    }
    
    bars = ax4.bar(range(len(metrics)), list(metrics.values()), 
                  color=['#3498db', '#2ecc71', '#e67e22', '#9b59b6'])
    ax4.set_xticks(range(len(metrics)))
    ax4.set_xticklabels(metrics.keys(), rotation=45)
    ax4.set_title('Key Performance Metrics (%)')
    
    # Add value labels on the bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save to desktop
    output_path = '/Users/noahshaffer/Desktop/panama_city_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def print_detailed_analysis(analysis_results: Dict):
    """Print comprehensive analysis results with enhanced market focus."""
    print("\nPanama City Beach Investment Analysis")
    print("=" * 50)
    
    # Market Analysis
    print("\nMarket Analysis")
    print("-" * 30)
    if "market_analysis" in analysis_results:
        market = analysis_results["market_analysis"]
        print(f"Location: {market['location']}")
        print(f"ZIP Code: {market['zip_code']}")
        
        metrics = market["market_metrics"]
        if metrics:
            print("\nMarket Metrics:")
            print(f"Price Trend: {metrics.get('price_trend', 'N/A')}%")
            print(f"Appreciation Rate: {metrics.get('appreciation_rate', 'N/A')}%")
            print(f"Market Volatility: {metrics.get('volatility', 'N/A')}")
            print(f"Market Score: {metrics.get('market_score', 'N/A')}/100")
    
    if "property_analysis" in analysis_results:
        prop = analysis_results["property_analysis"]
        print("\nProperty Market Analysis:")
        if "market_metrics" in prop:
            metrics = prop["market_metrics"]
            print(f"Market Health: {metrics.get('market_health', 'N/A')}/100")
            print(f"1-Year Appreciation: {metrics.get('1yr_appreciation', 'N/A')}%")
            print(f"Price Momentum: {metrics.get('price_momentum', 'N/A')}%")
        if "recommendation" in prop:
            print(f"\nMarket Recommendation: {prop['recommendation']}")
    
    # Investment Returns
    print("\nInvestment Returns")
    print("-" * 30)
    returns = analysis_results["investment_returns"]
    print(f"IRR: {returns['IRR']}%")
    print(f"Equity Multiple: {returns['Equity Multiple']}x")
    print(f"Cap Rate: {returns['Cap Rate']}%")
    print(f"Preferred Return: {returns['Preferred Return']}%")
    print(f"Average Cash-on-Cash Return: {returns['Average Cash-on-Cash Return']:.2f}%")
    
    # Capital Structure
    print("\nCapital Structure")
    print("-" * 30)
    capital = analysis_results["capital_structure"]
    print(f"Total Capital: ${capital['Total Capital Structure']:,.2f}")
    print(f"Debt: {capital['Debt Percentage']:.1f}%")
    print(f"Preferred Equity: {capital['Preferred Equity Percentage']:.1f}%")
    print(f"Original Equity: {capital['Original Equity Percentage']:.1f}%")
    
    # Operating Metrics
    print("\nOperating Metrics")
    print("-" * 30)
    ops = analysis_results["operating_metrics"]
    print(f"First Year NOI: ${ops['First Year NOI']:,.2f}")
    print(f"Operating Expense Ratio: {ops['Operating Expense Ratio']:.1f}%")
    print(f"Revenue Growth Rate: {ops['Revenue Growth Rate']}%")
    print(f"Average Expense Growth Rate: {ops['Expense Growth Rate']:.1f}%")
    
    # Cash Flow Analysis
    print("\nCash Flow Metrics")
    print("-" * 30)
    cf = analysis_results["cash_flow_analysis"]
    print(f"Average Annual NOI: ${cf['Average Annual NOI']:,.2f}")
    print(f"NOI Growth Rate: {cf['Average NOI Growth Rate']:.1f}%")
    print(f"Debt Service Coverage Ratio: {cf['Debt Service Coverage Ratio']:.2f}x")
    print(f"Total Cash Flow (excl. sale): ${cf['Total Cash Flow (excl. sale)']:,.2f}")

def main():
    # Load and analyze the deal data
    deal_data = {
        "financialMetrics": {
            "IRR": 17.0,
            "equityMultiple": 2.6,
            "capRate": 5.0,
            "preferredReturn": 14.0
        },
        "capitalStack": {
            "newDebt": 52002000,
            "preferredEquity": 19327767,
            "originalEquity": 31650000
        },
        "forecastedCashFlows": [
            {"year": 1, "grossOperatingIncome": 6404019, "expenses": 2668479, "netOperatingIncome": 3735540, "debtService": 2548098, "cashFlow": 1187442},
            {"year": 2, "grossOperatingIncome": 6724328, "expenses": 2743367, "netOperatingIncome": 3980961, "debtService": 2548098, "cashFlow": 1432863},
            {"year": 3, "grossOperatingIncome": 7009069, "expenses": 2819615, "netOperatingIncome": 4189455, "debtService": 2548098, "cashFlow": 1641357},
            {"year": 4, "grossOperatingIncome": 7272981, "expenses": 2897679, "netOperatingIncome": 4375302, "debtService": 2548098, "cashFlow": 1827204},
            {"year": 5, "grossOperatingIncome": 7525709, "expenses": 2977893, "netOperatingIncome": 4547816, "debtService": 2548098, "cashFlow": 1999718},
            {"year": 6, "grossOperatingIncome": 7773596, "expenses": 3060511, "netOperatingIncome": 4713085, "debtService": 2535312, "cashFlow": 1603399},
            {"year": 7, "grossOperatingIncome": 8020837, "expenses": 3145739, "netOperatingIncome": 4875098, "debtService": 2506527, "cashFlow": 1765412, "netSalesProceeds": 47538282}
        ],
        "investmentDetails": {
            "preferredReturnRate": 14.0,
            "distributionStructures": [{"preferredEquityDistributions": "50%", "originalEquityDistributions": "50%"}]
        },
        "sourceAndUseOfFunds": {
            "sources": {"loanAmount": 52002000, "preferredEquity": 19327767, "lenderReserves": 581583, "availableCash": 89000},
            "uses": {"outstandingPrincipal": 66105000, "closingCosts": 1252819, "existingMemberLoanAndInterest": 3102491, "interestRateBuydown": 1040040, "capitalReserves": 500000}
        },
        "growthAssumptions": {
            "revenue": {"grossPotentialRent": 3.0, "utilityReimbursement": 2.5, "otherIncome": 3.0},
            "expenses": {"taxes": 1.0, "insurance": 5.0, "utilities": 2.5, "trash": 2.5, "payroll": 2.5, "maintenance": 2.5, "services": 2.5, "advertising": 2.5}
        }
    }
    
    # Initialize analyzers
    deal_analyzer = DetailedDealAnalyzer(deal_data)
    
    # Get market analysis
    market_analyzer = MarketAnalyzer()
    try:
        market_data = market_analyzer.analyze_market(
            zip_code="32413",
            months_of_history=24
        )
        
        # Add market analysis to results
        analysis_results = {
            "investment_returns": deal_analyzer.calculate_investment_returns(),
            "capital_structure": deal_analyzer.analyze_capital_structure(),
            "operating_metrics": deal_analyzer.analyze_operating_metrics(),
            "cash_flow_analysis": deal_analyzer.analyze_cash_flows(),
            "market_analysis": {
                "location": "Panama City Beach, FL",
                "zip_code": "32413",
                "market_metrics": market_data
            }
        }
    except Exception as e:
        print(f"Note: Market analysis limited due to: {str(e)}")
        analysis_results = {
            "investment_returns": deal_analyzer.calculate_investment_returns(),
            "capital_structure": deal_analyzer.analyze_capital_structure(),
            "operating_metrics": deal_analyzer.analyze_operating_metrics(),
            "cash_flow_analysis": deal_analyzer.analyze_cash_flows()
        }
    
    # Add property-specific analysis
    property_analysis = analyze_property(
        purchase_price=66105000,  # Using outstanding principal as purchase price
        monthly_rent=6404019/12,  # Using first year gross operating income / 12
        down_payment_percent=0.20,
        interest_rate=0.065,
        zip_code="32413"  # Panama City Beach
    )
    
    if property_analysis.get("Market Analysis"):
        analysis_results["property_analysis"] = property_analysis["Market Analysis"]
    
    # Create visualizations
    output_path = create_visualizations(deal_data, analysis_results)
    print(f"\nCreated visualizations at: {output_path}")
    
    # Print comprehensive analysis
    print_detailed_analysis(analysis_results)

if __name__ == "__main__":
    main()
