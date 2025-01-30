"""
Main CLI interface for the Real Estate Development Analysis application.
"""

import os
import sys
from typing import Optional
import argparse
from colorama import init, Fore, Style
from project_analyzer import ProjectAnalyzer

def print_colored(text: str, color: str = Fore.WHITE, bold: bool = False) -> None:
    """Print colored text to console."""
    style = Style.BRIGHT if bold else ""
    print(f"{style}{color}{text}{Style.RESET_ALL}")

def create_new_project() -> ProjectAnalyzer:
    """Create a new project analysis."""
    print_colored("\nNew Project Analysis", Fore.CYAN, bold=True)
    project_name = input("Enter project name: ").strip()
    
    analyzer = ProjectAnalyzer(project_name)
    analyzer.collect_data()
    analyzer.analyze_project()
    
    report_file = analyzer.generate_report()
    print_colored(f"\nAnalysis complete! Report saved to: {report_file}", Fore.GREEN)
    
    save = input("\nWould you like to save this project for later? (y/n): ").lower().strip()
    if save == 'y':
        filename = os.path.join("projects", f"{project_name}.json")
        os.makedirs("projects", exist_ok=True)
        analyzer.save_project(filename)
        print_colored(f"Project saved to: {filename}", Fore.GREEN)
    
    return analyzer

def load_existing_project() -> Optional[ProjectAnalyzer]:
    """Load an existing project analysis."""
    if not os.path.exists("projects"):
        print_colored("No saved projects found.", Fore.YELLOW)
        return None
    
    projects = [f for f in os.listdir("projects") if f.endswith('.json')]
    if not projects:
        print_colored("No saved projects found.", Fore.YELLOW)
        return None
    
    print_colored("\nSaved Projects:", Fore.CYAN, bold=True)
    for i, project in enumerate(projects, 1):
        print(f"{i}. {project[:-5]}")
    
    while True:
        try:
            choice = int(input("\nSelect a project number (0 to cancel): "))
            if choice == 0:
                return None
            if 1 <= choice <= len(projects):
                filename = os.path.join("projects", projects[choice-1])
                analyzer = ProjectAnalyzer.load_project(filename)
                print_colored(f"Project '{analyzer.project_name}' loaded successfully!", Fore.GREEN)
                return analyzer
            print_colored("Invalid choice. Please try again.", Fore.RED)
        except ValueError:
            print_colored("Invalid input. Please enter a number.", Fore.RED)

def perform_sensitivity_analysis(analyzer: ProjectAnalyzer) -> None:
    """Perform sensitivity analysis on a project variable."""
    print_colored("\nSensitivity Analysis", Fore.CYAN, bold=True)
    
    # Select category
    categories = list(analyzer.categories.keys())
    print("\nAvailable categories:")
    for i, category in enumerate(categories, 1):
        print(f"{i}. {category}")
    
    while True:
        try:
            cat_choice = int(input("\nSelect category number: ")) - 1
            if 0 <= cat_choice < len(categories):
                category_name = categories[cat_choice]
                break
            print_colored("Invalid choice. Please try again.", Fore.RED)
        except ValueError:
            print_colored("Invalid input. Please enter a number.", Fore.RED)
    
    # Select variable
    category = analyzer.categories[category_name]
    variables = list(category.data.keys())
    print("\nAvailable variables:")
    for i, var in enumerate(variables, 1):
        print(f"{i}. {var}")
    
    while True:
        try:
            var_choice = int(input("\nSelect variable number: ")) - 1
            if 0 <= var_choice < len(variables):
                variable = variables[var_choice]
                break
            print_colored("Invalid choice. Please try again.", Fore.RED)
        except ValueError:
            print_colored("Invalid input. Please enter a number.", Fore.RED)
    
    # Perform analysis
    results = analyzer.perform_sensitivity_analysis(category_name, variable)
    
    print_colored(f"\nSensitivity Analysis Results for {variable}:", Fore.CYAN)
    print("\nVariation\tScore")
    print("-" * 30)
    for var, score in zip(results['variations'], results['scores']):
        print(f"{var:+.2%}\t\t{score:.3f}")

def compare_projects(analyzer1: ProjectAnalyzer) -> None:
    """Compare two projects."""
    print_colored("\nProject Comparison", Fore.CYAN, bold=True)
    
    analyzer2 = load_existing_project()
    if not analyzer2:
        return
    
    comparison = analyzer1.compare_projects(analyzer2)
    
    print_colored("\nOverall Comparison:", Fore.CYAN)
    print(f"{analyzer1.project_name}: {comparison['overall_comparison'][analyzer1.project_name]:.3f}")
    print(f"{analyzer2.project_name}: {comparison['overall_comparison'][analyzer2.project_name]:.3f}")
    
    print_colored("\nCategory Comparison:", Fore.CYAN)
    for category, scores in comparison['category_comparison'].items():
        print(f"\n{category.title()}:")
        print(f"{analyzer1.project_name}: {scores[analyzer1.project_name]:.3f}")
        print(f"{analyzer2.project_name}: {scores[analyzer2.project_name]:.3f}")

def main() -> None:
    """Main CLI interface."""
    init()  # Initialize colorama
    
    print_colored("""
    ╔═══════════════════════════════════════════╗
    ║  Real Estate Development Analysis Tool    ║
    ╚═══════════════════════════════════════════╝
    """, Fore.CYAN, bold=True)
    
    current_analyzer = None
    
    while True:
        print_colored("\nMain Menu:", Fore.CYAN, bold=True)
        print("1. Create New Project Analysis")
        print("2. Load Existing Project")
        print("3. Perform Sensitivity Analysis")
        print("4. Compare Projects")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        try:
            if choice == '1':
                current_analyzer = create_new_project()
            elif choice == '2':
                current_analyzer = load_existing_project()
            elif choice == '3':
                if current_analyzer:
                    perform_sensitivity_analysis(current_analyzer)
                else:
                    print_colored("Please create or load a project first.", Fore.YELLOW)
            elif choice == '4':
                if current_analyzer:
                    compare_projects(current_analyzer)
                else:
                    print_colored("Please create or load a project first.", Fore.YELLOW)
            elif choice == '5':
                print_colored("\nThank you for using the Real Estate Development Analysis Tool!", 
                            Fore.GREEN, bold=True)
                sys.exit(0)
            else:
                print_colored("Invalid choice. Please try again.", Fore.RED)
        except Exception as e:
            print_colored(f"An error occurred: {str(e)}", Fore.RED)

if __name__ == "__main__":
    main()
