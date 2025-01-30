"""
Main project analyzer class that coordinates analysis across all categories
and generates comprehensive reports.
"""

import json
import os
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from analysis_categories import AnalysisCategory, Financial, Market

class ProjectAnalyzer:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.categories: Dict[str, AnalysisCategory] = {
            'financial': Financial(),
            'market': Market(),
            # Additional categories will be added here
        }
        self.overall_score = 0.0

    def collect_data(self) -> None:
        """Collect data for all analysis categories."""
        print(f"\nAnalyzing project: {self.project_name}")
        for category in self.categories.values():
            print(f"\n{category.name} Analysis:")
            category.input_data()

    def analyze_project(self) -> float:
        """Analyze all categories and calculate overall project score."""
        total_weight = sum(cat.weight for cat in self.categories.values())
        weighted_scores = sum(cat.analyze() * cat.weight for cat in self.categories.values())
        self.overall_score = weighted_scores / total_weight
        return self.overall_score

    def generate_report(self, output_dir: str = "reports") -> str:
        """Generate a comprehensive project analysis report."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"{self.project_name}_{timestamp}_report.txt")

        with open(report_file, 'w') as f:
            # Write executive summary
            f.write(f"{'='*80}\n")
            f.write(f"REAL ESTATE DEVELOPMENT PROJECT ANALYSIS REPORT\n")
            f.write(f"Project: {self.project_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")

            f.write("EXECUTIVE SUMMARY\n")
            f.write(f"Overall Project Score: {self.overall_score:.2f}/1.00\n\n")

            # Write detailed category analysis
            f.write("DETAILED ANALYSIS\n")
            for category in self.categories.values():
                report = category.generate_report()
                f.write(f"\n{report['category']}\n")
                f.write("-" * len(report['category']) + "\n")
                for metric, value in report['metrics'].items():
                    f.write(f"{metric}: {value}\n")
                f.write(f"Category Score: {category.score:.2f}\n")
                f.write(f"Summary: {report['summary']}\n")

        self._generate_visualizations(output_dir, timestamp)
        return report_file

    def _generate_visualizations(self, output_dir: str, timestamp: str) -> None:
        """Generate visual representations of the analysis."""
        # Category scores comparison
        plt.figure(figsize=(10, 6))
        categories = list(self.categories.keys())
        scores = [cat.score for cat in self.categories.values()]
        
        plt.bar(categories, scores)
        plt.title("Category Scores Comparison")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        for i, score in enumerate(scores):
            plt.text(i, score + 0.01, f'{score:.2f}', ha='center')
        
        plt.savefig(os.path.join(output_dir, f"{self.project_name}_{timestamp}_scores.png"))
        plt.close()

    def save_project(self, filename: str) -> None:
        """Save project data to a file."""
        data = {
            'project_name': self.project_name,
            'overall_score': self.overall_score,
            'categories': {
                name: category.to_dict()
                for name, category in self.categories.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load_project(cls, filename: str) -> 'ProjectAnalyzer':
        """Load project data from a file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        analyzer = cls(data['project_name'])
        analyzer.overall_score = data['overall_score']
        
        for name, category_data in data['categories'].items():
            if name in analyzer.categories:
                analyzer.categories[name] = analyzer.categories[name].__class__.from_dict(category_data)
        
        return analyzer

    def perform_sensitivity_analysis(self, category_name: str, variable: str, 
                                  range_percent: float = 20, steps: int = 5) -> Dict[str, List[float]]:
        """Perform sensitivity analysis on a specific variable."""
        if category_name not in self.categories:
            raise ValueError(f"Category {category_name} not found")
        
        category = self.categories[category_name]
        if variable not in category.data:
            raise ValueError(f"Variable {variable} not found in {category_name}")

        base_value = category.data[variable]
        results = {
            'variations': [],
            'scores': []
        }

        for i in range(-steps, steps + 1):
            variation = 1 + (i * range_percent / 100 / steps)
            category.data[variable] = base_value * variation
            score = category.analyze()
            results['variations'].append(variation)
            results['scores'].append(score)

        # Reset to original value
        category.data[variable] = base_value
        category.analyze()
        
        return results

    def compare_projects(self, other_project: 'ProjectAnalyzer') -> Dict[str, Any]:
        """Compare this project with another project."""
        comparison = {
            'overall_comparison': {
                self.project_name: self.overall_score,
                other_project.project_name: other_project.overall_score
            },
            'category_comparison': {}
        }

        for category_name in self.categories:
            if category_name in other_project.categories:
                comparison['category_comparison'][category_name] = {
                    self.project_name: self.categories[category_name].score,
                    other_project.project_name: other_project.categories[category_name].score
                }

        return comparison
