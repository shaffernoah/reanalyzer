"""
Core analysis categories for real estate development projects.
Each category represents a different aspect of analysis with its own metrics and scoring system.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List
import json

class AnalysisCategory(ABC):
    def __init__(self, name: str):
        self.name = name
        self.data: Dict[str, Any] = {}
        self.score = 0.0
        self.weight = 1.0  # Default weight for scoring

    @abstractmethod
    def input_data(self) -> None:
        """Input data specific to this category."""
        pass

    @abstractmethod
    def analyze(self) -> float:
        """Analyze the data and return a score between 0 and 1."""
        pass

    @abstractmethod
    def generate_report(self) -> Dict[str, Any]:
        """Generate a detailed report of the analysis."""
        pass

    def validate_numeric(self, value: Any, min_val: float = None, max_val: float = None) -> float:
        """Validate numeric input within specified range."""
        try:
            num = float(value)
            if min_val is not None and num < min_val:
                raise ValueError(f"Value must be greater than {min_val}")
            if max_val is not None and num > max_val:
                raise ValueError(f"Value must be less than {max_val}")
            return num
        except ValueError as e:
            raise ValueError(f"Invalid numeric input: {str(e)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert category data to dictionary for serialization."""
        return {
            'name': self.name,
            'data': self.data,
            'score': self.score,
            'weight': self.weight
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisCategory':
        """Create category instance from dictionary data."""
        instance = cls(data['name'])
        instance.data = data['data']
        instance.score = data['score']
        instance.weight = data['weight']
        return instance

class Financial(AnalysisCategory):
    def __init__(self):
        super().__init__("Financial")
        self.required_fields = [
            'total_cost',
            'expected_revenue',
            'development_period',
            'interest_rate',
            'equity_percentage'
        ]

    def input_data(self) -> None:
        try:
            self.data['total_cost'] = self.validate_numeric(
                input("Total Development Cost ($): "), min_val=0)
            self.data['expected_revenue'] = self.validate_numeric(
                input("Expected Revenue ($): "), min_val=0)
            self.data['development_period'] = self.validate_numeric(
                input("Development Period (months): "), min_val=1)
            self.data['interest_rate'] = self.validate_numeric(
                input("Interest Rate (%): "), min_val=0, max_val=100)
            self.data['equity_percentage'] = self.validate_numeric(
                input("Equity Percentage (%): "), min_val=0, max_val=100)
        except ValueError as e:
            raise ValueError(f"Financial data input error: {str(e)}")

    def analyze(self) -> float:
        # Calculate ROI and other financial metrics
        roi = (self.data['expected_revenue'] - self.data['total_cost']) / self.data['total_cost']
        monthly_rate = self.data['interest_rate'] / 12 / 100
        debt_amount = self.data['total_cost'] * (1 - self.data['equity_percentage'] / 100)
        
        # Calculate debt service coverage ratio
        monthly_payment = (debt_amount * monthly_rate * (1 + monthly_rate) ** 
                         self.data['development_period']) / ((1 + monthly_rate) ** 
                         self.data['development_period'] - 1)
        dscr = (self.data['expected_revenue'] / 12) / monthly_payment

        # Calculate score based on multiple factors
        roi_score = min(max(roi / 0.3, 0), 1)  # Normalize ROI with 30% as benchmark
        dscr_score = min(max((dscr - 1.2) / 0.8, 0), 1)  # DSCR score (1.2 to 2.0 range)
        
        self.score = (roi_score * 0.6 + dscr_score * 0.4)  # Weighted average
        return self.score

    def generate_report(self) -> Dict[str, Any]:
        roi = (self.data['expected_revenue'] - self.data['total_cost']) / self.data['total_cost']
        return {
            'category': 'Financial Analysis',
            'metrics': {
                'Total Cost': f"${self.data['total_cost']:,.2f}",
                'Expected Revenue': f"${self.data['expected_revenue']:,.2f}",
                'ROI': f"{roi*100:.2f}%",
                'Development Period': f"{self.data['development_period']} months",
                'Interest Rate': f"{self.data['interest_rate']}%",
                'Equity Percentage': f"{self.data['equity_percentage']}%"
            },
            'score': self.score,
            'summary': f"The project shows a {roi*100:.1f}% ROI with a financial viability score of {self.score:.2f}"
        }

class Market(AnalysisCategory):
    def __init__(self):
        super().__init__("Market")
        self.required_fields = [
            'market_size',
            'growth_rate',
            'competition_level',
            'demand_score',
            'absorption_rate'
        ]

    def input_data(self) -> None:
        try:
            self.data['market_size'] = self.validate_numeric(
                input("Market Size (millions $): "), min_val=0)
            self.data['growth_rate'] = self.validate_numeric(
                input("Annual Market Growth Rate (%): "), min_val=-100, max_val=100)
            self.data['competition_level'] = self.validate_numeric(
                input("Competition Level (1-10): "), min_val=1, max_val=10)
            self.data['demand_score'] = self.validate_numeric(
                input("Demand Score (1-10): "), min_val=1, max_val=10)
            self.data['absorption_rate'] = self.validate_numeric(
                input("Monthly Absorption Rate (%): "), min_val=0, max_val=100)
        except ValueError as e:
            raise ValueError(f"Market data input error: {str(e)}")

    def analyze(self) -> float:
        # Calculate market score based on various factors
        growth_score = (self.data['growth_rate'] + 20) / 40  # Normalize growth rate (-20% to +20%)
        competition_score = (11 - self.data['competition_level']) / 10  # Inverse of competition level
        demand_score = self.data['demand_score'] / 10
        absorption_score = self.data['absorption_rate'] / 100

        # Weighted average of scores
        self.score = (growth_score * 0.3 + 
                     competition_score * 0.2 + 
                     demand_score * 0.3 + 
                     absorption_score * 0.2)
        return self.score

    def generate_report(self) -> Dict[str, Any]:
        return {
            'category': 'Market Analysis',
            'metrics': {
                'Market Size': f"${self.data['market_size']}M",
                'Growth Rate': f"{self.data['growth_rate']}%",
                'Competition Level': f"{self.data['competition_level']}/10",
                'Demand Score': f"{self.data['demand_score']}/10",
                'Absorption Rate': f"{self.data['absorption_rate']}%"
            },
            'score': self.score,
            'summary': (f"Market conditions show {self.data['growth_rate']}% growth "
                       f"with a market viability score of {self.score:.2f}")
        }

# Additional categories will be implemented similarly
