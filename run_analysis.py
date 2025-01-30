from state_market_visualizer import StateMarketVisualizer
import os
from datetime import datetime

def main():
    # Get the desktop path and construct file paths
    desktop = os.path.expanduser("~/Desktop")
    zillow_data_path = "zillow_data.csv"  # File is in the current directory
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = os.path.join(desktop, f"market_analysis_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nOutput will be saved to: {output_dir}")
    
    # Initialize visualizer with output directory
    print("Loading Zillow data...")
    visualizer = StateMarketVisualizer(zillow_data_path)
    visualizer.set_output_directory(output_dir)
    
    # Generate analyses
    print("\nGenerating state-level dashboards...")
    major_states = ['CA', 'TX', 'FL', 'NY', 'IL']
    
    # Individual state analysis
    for state in major_states:
        print(f"\nAnalyzing {state}...")
        visualizer.generate_state_dashboard(state, save=True)  # Now saves to file
    
    # Comparative analysis
    print("\nGenerating comparative analysis for major markets...")
    visualizer.generate_comparative_analysis(major_states[:3], save=True)
    
    # Export reports
    print("\nExporting detailed reports...")
    for state in major_states:
        print(f"Exporting {state} report...")
        visualizer.export_state_report(state, format='pdf')
    
    print(f"\nAnalysis complete! All files have been saved to:\n{output_dir}")
    print("\nYou'll find:")
    print("1. Individual state dashboards (HTML files)")
    print("2. Comparative analysis dashboard (HTML file)")
    print("3. Detailed PDF reports for each state")

if __name__ == "__main__":
    main() 