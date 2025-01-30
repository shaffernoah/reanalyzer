import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import os

class StateMarketVisualizer:
    def __init__(self, zillow_data_path: str):
        """Initialize with Zillow dataset"""
        self.data = pd.read_csv(zillow_data_path)
        self.states = sorted(self.data['StateName'].unique())
        self.current_state = None
        self.processed_data = {}
        self.output_dir = None
        self._preprocess_data()

    def _preprocess_data(self) -> None:
        """Preprocess and structure the data for analysis"""
        # Convert date columns
        date_columns = [col for col in self.data.columns if col.startswith('20')]
        self.data_melted = self.data.melt(
            id_vars=['RegionName', 'StateName', 'Metro', 'CountyName', 'SizeRank'],
            value_vars=date_columns,
            var_name='Date',
            value_name='Value'
        )
        self.data_melted['Date'] = pd.to_datetime(self.data_melted['Date'])
        
        # Calculate key metrics by state
        self._calculate_state_metrics()

    def _calculate_state_metrics(self) -> None:
        """Calculate comprehensive metrics for each state"""
        for state in self.states:
            state_data = self.data_melted[self.data_melted['StateName'] == state]
            
            self.processed_data[state] = {
                'price_trends': self._analyze_price_trends(state_data),
                'market_health': self._analyze_market_health(state_data),
                'growth_metrics': self._calculate_growth_metrics(state_data),
                'volatility': self._calculate_volatility(state_data),
                'seasonal_patterns': self._analyze_seasonality(state_data),
                'metro_analysis': self._analyze_metro_areas(state_data)
            }

    def set_state(self, state: str) -> None:
        """Set the current state for analysis"""
        if state not in self.states:
            raise ValueError(f"State '{state}' not found in dataset")
        self.current_state = state

    def set_output_directory(self, directory: str) -> None:
        """Set the output directory for saved files"""
        self.output_dir = directory
        os.makedirs(directory, exist_ok=True)

    def generate_state_dashboard(self, state: Optional[str] = None, save: bool = False) -> None:
        """Generate comprehensive dashboard for a state"""
        if state:
            self.set_state(state)
        if not self.current_state:
            raise ValueError("Please set a state first")

        # Create dashboard layout
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Price Trends Over Time',
                'YoY Growth Rates',
                'Market Health Index',
                'Price Distribution',
                'Seasonal Patterns',
                'Metro Area Comparison'
            ),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                  [{'type': 'heatmap'}, {'type': 'box'}],
                  [{'type': 'scatter'}, {'type': 'bar'}]]
        )

        # Add plots
        self._add_price_trend_plot(fig, 1, 1)
        self._add_growth_rate_plot(fig, 1, 2)
        self._add_market_health_plot(fig, 2, 1)
        self._add_price_distribution_plot(fig, 2, 2)
        self._add_seasonal_pattern_plot(fig, 3, 1)
        self._add_metro_comparison_plot(fig, 3, 2)

        # Update layout
        fig.update_layout(
            height=1200,
            width=1600,
            title_text=f"Real Estate Market Analysis - {self.current_state}",
            showlegend=True
        )

        if save and self.output_dir:
            filename = os.path.join(self.output_dir, f"{state}_dashboard.html")
            fig.write_html(filename)
            print(f"Dashboard saved to: {filename}")
        else:
            fig.show()

    def generate_comparative_analysis(self, states: List[str], save: bool = False) -> None:
        """Generate comparative analysis between multiple states"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Median Price Comparison',
                'Growth Rate Comparison',
                'Market Health Comparison',
                'Volatility Comparison'
            )
        )

        # Implementation of comparative plots
        self._add_comparative_price_plot(fig, states, 1, 1)
        self._add_comparative_growth_plot(fig, states, 1, 2)
        self._add_comparative_health_plot(fig, states, 2, 1)
        self._add_comparative_volatility_plot(fig, states, 2, 2)

        fig.update_layout(
            height=1000,
            width=1600,
            title_text="Multi-State Comparative Analysis"
        )

        if save and self.output_dir:
            filename = os.path.join(self.output_dir, "comparative_analysis.html")
            fig.write_html(filename)
            print(f"Comparative analysis saved to: {filename}")
        else:
            fig.show()

    def export_state_report(self, state: str, format: str = 'pdf') -> None:
        """Export comprehensive state analysis report"""
        if state not in self.states:
            raise ValueError(f"State '{state}' not found in dataset")

        if not self.output_dir:
            raise ValueError("Output directory not set. Call set_output_directory() first.")
            
        filename = os.path.join(self.output_dir, f"{state}_report.{format}")
        
        # Generate report content
        report_data = self.processed_data[state]
        
        if format == 'pdf':
            self._generate_pdf_report(state, report_data)
        elif format == 'excel':
            self._generate_excel_report(state, report_data)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Report saved to: {filename}")

    def _analyze_price_trends(self, data: pd.DataFrame) -> Dict:
        """Analyze historical price trends"""
        return {
            'median_prices': data.groupby('Date')['Value'].median(),
            'price_momentum': self._calculate_momentum(data),
            'trend_strength': self._calculate_trend_strength(data)
        }

    def _analyze_market_health(self, data: pd.DataFrame) -> Dict:
        """Calculate market health indicators"""
        return {
            'price_stability': self._calculate_stability(data),
            'growth_sustainability': self._calculate_growth_sustainability(data),
            'market_strength': self._calculate_market_strength(data)
        }

    def _calculate_growth_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate various growth metrics"""
        return {
            'yoy_growth': self._calculate_yoy_growth(data),
            'cagr': self._calculate_cagr(data),
            'growth_acceleration': self._calculate_growth_acceleration(data)
        }

    def _analyze_seasonality(self, data: pd.DataFrame) -> Dict:
        """Analyze seasonal patterns in the data"""
        return {
            'seasonal_indices': self._calculate_seasonal_indices(data),
            'seasonal_strength': self._calculate_seasonal_strength(data)
        }

    def _analyze_metro_areas(self, data: pd.DataFrame) -> Dict:
        """Analyze metropolitan areas within the state"""
        return {
            'metro_rankings': self._rank_metro_areas(data),
            'metro_growth_rates': self._calculate_metro_growth(data),
            'metro_price_levels': self._calculate_metro_prices(data)
        }

    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        """Calculate price momentum using recent price changes"""
        recent_prices = data.groupby('Date')['Value'].median().tail(12)  # Last 12 months
        return (recent_prices.pct_change().mean() * 100)  # Average monthly change as percentage

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate the strength of the price trend"""
        prices = data.groupby('Date')['Value'].median()
        return np.corrcoef(prices.index.astype(int), prices.values)[0, 1]

    def _calculate_stability(self, data: pd.DataFrame) -> float:
        """Calculate price stability metric"""
        return 1 / (data.groupby('Date')['Value'].std().mean() / data.groupby('Date')['Value'].mean().mean())

    def _calculate_growth_sustainability(self, data: pd.DataFrame) -> float:
        """Calculate growth sustainability score"""
        growth_rates = data.groupby('Date')['Value'].median().pct_change()
        return 1 - abs(growth_rates.std())  # Higher score means more sustainable growth

    def _calculate_market_strength(self, data: pd.DataFrame) -> float:
        """Calculate overall market strength"""
        recent_growth = self._calculate_momentum(data)
        stability = self._calculate_stability(data)
        return (recent_growth + stability) / 2

    def _calculate_yoy_growth(self, data: pd.DataFrame) -> pd.Series:
        """Calculate year-over-year growth rates"""
        monthly_prices = data.groupby('Date')['Value'].median()
        return monthly_prices.pct_change(12) * 100

    def _calculate_cagr(self, data: pd.DataFrame) -> float:
        """Calculate Compound Annual Growth Rate"""
        monthly_prices = data.groupby('Date')['Value'].median()
        total_years = (monthly_prices.index[-1] - monthly_prices.index[0]).days / 365.25
        return ((monthly_prices.iloc[-1] / monthly_prices.iloc[0]) ** (1/total_years) - 1) * 100

    def _calculate_growth_acceleration(self, data: pd.DataFrame) -> pd.Series:
        """Calculate growth acceleration"""
        monthly_prices = data.groupby('Date')['Value'].median()
        return monthly_prices.pct_change().diff()

    def _calculate_seasonal_indices(self, data: pd.DataFrame) -> pd.Series:
        """Calculate seasonal indices"""
        monthly_prices = data.groupby('Date')['Value'].median()
        return monthly_prices.groupby(monthly_prices.index.month).mean()

    def _calculate_seasonal_strength(self, data: pd.DataFrame) -> float:
        """Calculate strength of seasonality"""
        seasonal_indices = self._calculate_seasonal_indices(data)
        return seasonal_indices.std() / seasonal_indices.mean()

    def _rank_metro_areas(self, data: pd.DataFrame) -> pd.DataFrame:
        """Rank metropolitan areas by median price"""
        return data.groupby('Metro')['Value'].median().sort_values(ascending=False)

    def _calculate_metro_growth(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate growth rates for each metro area"""
        metro_prices = data.groupby(['Metro', 'Date'])['Value'].median().unstack()
        return (metro_prices.iloc[:, -1] / metro_prices.iloc[:, 0] - 1) * 100

    def _calculate_metro_prices(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate current price levels for each metro area"""
        return data[data['Date'] == data['Date'].max()].groupby('Metro')['Value'].median()

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate price volatility"""
        monthly_returns = data.groupby('Date')['Value'].median().pct_change()
        return monthly_returns.std() * np.sqrt(12)  # Annualized volatility

    def _generate_pdf_report(self, state: str, report_data: Dict) -> None:
        """Generate PDF report for a state"""
        if not self.output_dir:
            return
            
        filename = os.path.join(self.output_dir, f"{state}_report.pdf")
        
        # Create a temporary HTML file for the report
        temp_html = os.path.join(self.output_dir, f"{state}_temp_report.html")
        
        # Generate HTML content
        html_content = f"""
        <html>
        <head>
            <title>Real Estate Market Analysis - {state}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .metric {{ margin: 20px 0; padding: 15px; background: #f7f9fc; border-radius: 5px; }}
                .metric h3 {{ color: #7f8c8d; margin: 0 0 10px 0; }}
                .value {{ font-size: 24px; color: #2c3e50; }}
            </style>
        </head>
        <body>
            <h1>Real Estate Market Analysis Report - {state}</h1>
            <div class="metric">
                <h2>Price Trends</h2>
                <div class="value">
                    <p>Current Median Price: ${report_data['price_trends']['median_prices'][-1]:,.2f}</p>
                    <p>Price Momentum: {report_data['price_trends']['price_momentum']:.1f}%</p>
                    <p>Trend Strength: {report_data['price_trends']['trend_strength']:.2f}</p>
                </div>
            </div>
            
            <div class="metric">
                <h2>Market Health</h2>
                <div class="value">
                    <p>Price Stability: {report_data['market_health']['price_stability']:.2f}</p>
                    <p>Growth Sustainability: {report_data['market_health']['growth_sustainability']:.2f}</p>
                    <p>Market Strength: {report_data['market_health']['market_strength']:.2f}</p>
                </div>
            </div>
            
            <div class="metric">
                <h2>Growth Metrics</h2>
                <div class="value">
                    <p>Latest YoY Growth: {report_data['growth_metrics']['yoy_growth'][-1]:.1f}%</p>
                    <p>CAGR: {report_data['growth_metrics']['cagr']:.1f}%</p>
                </div>
            </div>
            
            <div class="metric">
                <h2>Market Volatility</h2>
                <div class="value">
                    <p>Volatility Index: {report_data['volatility']:.2f}</p>
                </div>
            </div>
            
            <div class="metric">
                <h2>Top Metro Areas</h2>
                <div class="value">
                    {''.join(f"<p>{metro}: ${price:,.2f}</p>" for metro, price in report_data['metro_analysis']['metro_price_levels'].nlargest(5).items())}
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write HTML content to temporary file
        with open(temp_html, 'w') as f:
            f.write(html_content)
        
        # Convert HTML to PDF using wkhtmltopdf
        os.system(f'wkhtmltopdf "{temp_html}" "{filename}"')
        
        # Clean up temporary file
        try:
            os.remove(temp_html)
        except:
            pass

    def _generate_excel_report(self, state: str, report_data: Dict) -> None:
        """Generate Excel report for a state"""
        # Placeholder for Excel report generation
        # This would typically use pandas to_excel
        pass

    def _add_price_trend_plot(self, fig, row, col):
        """Add price trend plot to the dashboard"""
        if not self.current_state:
            return
            
        state_data = self.data_melted[self.data_melted['StateName'] == self.current_state]
        monthly_prices = state_data.groupby('Date')['Value'].median()
        
        fig.add_trace(
            go.Scatter(
                x=monthly_prices.index,
                y=monthly_prices.values,
                name='Median Price',
                mode='lines'
            ),
            row=row, col=col
        )
        fig.update_xaxes(title_text='Date', row=row, col=col)
        fig.update_yaxes(title_text='Price ($)', row=row, col=col)

    def _add_growth_rate_plot(self, fig, row, col):
        """Add growth rate plot to the dashboard"""
        if not self.current_state:
            return
            
        state_data = self.processed_data[self.current_state]['growth_metrics']['yoy_growth']
        
        fig.add_trace(
            go.Bar(
                x=state_data.index,
                y=state_data.values,
                name='YoY Growth %'
            ),
            row=row, col=col
        )
        fig.update_xaxes(title_text='Date', row=row, col=col)
        fig.update_yaxes(title_text='Growth Rate (%)', row=row, col=col)

    def _add_market_health_plot(self, fig, row, col):
        """Add market health plot to the dashboard"""
        if not self.current_state:
            return
            
        state_data = self.data_melted[self.data_melted['StateName'] == self.current_state]
        monthly_volatility = state_data.groupby('Date')['Value'].std() / state_data.groupby('Date')['Value'].mean()
        
        fig.add_trace(
            go.Scatter(
                x=monthly_volatility.index,
                y=monthly_volatility.values,
                name='Market Volatility',
                mode='lines'
            ),
            row=row, col=col
        )
        fig.update_xaxes(title_text='Date', row=row, col=col)
        fig.update_yaxes(title_text='Volatility Index', row=row, col=col)

    def _add_price_distribution_plot(self, fig, row, col):
        """Add price distribution plot to the dashboard"""
        if not self.current_state:
            return
            
        state_data = self.data_melted[self.data_melted['StateName'] == self.current_state]
        recent_data = state_data[state_data['Date'] == state_data['Date'].max()]
        
        fig.add_trace(
            go.Box(
                y=recent_data['Value'],
                name='Price Distribution'
            ),
            row=row, col=col
        )
        fig.update_yaxes(title_text='Price ($)', row=row, col=col)

    def _add_seasonal_pattern_plot(self, fig, row, col):
        """Add seasonal pattern plot to the dashboard"""
        if not self.current_state:
            return
            
        state_data = self.data_melted[self.data_melted['StateName'] == self.current_state]
        seasonal_indices = self._calculate_seasonal_indices(state_data)
        
        fig.add_trace(
            go.Scatter(
                x=list(range(1, 13)),
                y=seasonal_indices.values,
                name='Seasonal Pattern',
                mode='lines+markers'
            ),
            row=row, col=col
        )
        fig.update_xaxes(title_text='Month', row=row, col=col)
        fig.update_yaxes(title_text='Price Index', row=row, col=col)

    def _add_metro_comparison_plot(self, fig, row, col):
        """Add metro comparison plot to the dashboard"""
        if not self.current_state:
            return
            
        state_data = self.data_melted[self.data_melted['StateName'] == self.current_state]
        metro_prices = state_data[state_data['Date'] == state_data['Date'].max()].groupby('Metro')['Value'].median()
        top_metros = metro_prices.nlargest(5)
        
        fig.add_trace(
            go.Bar(
                x=top_metros.index,
                y=top_metros.values,
                name='Top Metro Areas'
            ),
            row=row, col=col
        )
        fig.update_xaxes(title_text='Metro Area', row=row, col=col)
        fig.update_yaxes(title_text='Median Price ($)', row=row, col=col)

    def _add_comparative_price_plot(self, fig, states, row, col):
        """Add comparative price plot for multiple states"""
        for state in states:
            state_data = self.data_melted[self.data_melted['StateName'] == state]
            monthly_prices = state_data.groupby('Date')['Value'].median()
            
            fig.add_trace(
                go.Scatter(
                    x=monthly_prices.index,
                    y=monthly_prices.values,
                    name=f'{state} Prices',
                    mode='lines'
                ),
                row=row, col=col
            )
        fig.update_xaxes(title_text='Date', row=row, col=col)
        fig.update_yaxes(title_text='Price ($)', row=row, col=col)

    def _add_comparative_growth_plot(self, fig, states, row, col):
        """Add comparative growth plot for multiple states"""
        for state in states:
            growth_data = self.processed_data[state]['growth_metrics']['yoy_growth']
            
            fig.add_trace(
                go.Scatter(
                    x=growth_data.index,
                    y=growth_data.values,
                    name=f'{state} Growth',
                    mode='lines'
                ),
                row=row, col=col
            )
        fig.update_xaxes(title_text='Date', row=row, col=col)
        fig.update_yaxes(title_text='Growth Rate (%)', row=row, col=col)

    def _add_comparative_health_plot(self, fig, states, row, col):
        """Add comparative market health plot for multiple states"""
        for state in states:
            health_data = self.processed_data[state]['market_health']['market_strength']
            
            fig.add_trace(
                go.Bar(
                    x=[state],
                    y=[health_data],
                    name=f'{state} Health'
                ),
                row=row, col=col
            )
        fig.update_xaxes(title_text='State', row=row, col=col)
        fig.update_yaxes(title_text='Market Health Index', row=row, col=col)

    def _add_comparative_volatility_plot(self, fig, states, row, col):
        """Add comparative volatility plot for multiple states"""
        volatilities = []
        for state in states:
            volatility = self.processed_data[state]['volatility']
            volatilities.append({'State': state, 'Volatility': volatility})
        
        volatilities_df = pd.DataFrame(volatilities)
        
        fig.add_trace(
            go.Bar(
                x=volatilities_df['State'],
                y=volatilities_df['Volatility'],
                name='Market Volatility'
            ),
            row=row, col=col
        )
        fig.update_xaxes(title_text='State', row=row, col=col)
        fig.update_yaxes(title_text='Volatility Index', row=row, col=col)