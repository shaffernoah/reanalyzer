# Real Estate Development Analysis Tool

A comprehensive Python application for analyzing real estate development projects across multiple dimensions including financial viability, market conditions, location analysis, legal considerations, design aspects, political factors, and exit strategies.

## Features

- **Multi-dimensional Analysis**: Evaluate projects across various critical aspects
- **Scoring System**: Get objective scores for each category and overall project viability
- **Data Visualization**: Visual representations of key metrics and comparisons
- **Sensitivity Analysis**: Understand how changes in key variables affect project viability
- **Project Comparison**: Compare multiple projects side-by-side
- **Data Persistence**: Save and load project data using Supabase
- **Web Interface**: Easy-to-use Streamlit web interface
- **Comprehensive Reporting**: Generate detailed analysis reports with recommendations

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd real-estate-analyzer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.template .env
```
Then edit `.env` with your Supabase credentials.

## Usage

### Command Line Interface
Run the analysis tool:
```bash
python main.py
```

The interactive CLI will guide you through:
1. Creating a new project analysis
2. Loading existing projects
3. Performing sensitivity analysis
4. Comparing multiple projects

### Web Interface
Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

The web interface will guide you through:
1. Entering property details
2. Viewing market analysis results
3. Exploring historical trends
4. Accessing saved analyses

## Project Structure

- `main.py`: Main CLI interface
- `project_analyzer.py`: Core project analysis coordination
- `analysis_categories.py`: Individual analysis category implementations
- `database.py`: Supabase database operations
- `streamlit_app.py`: Web interface
- `reports/`: Generated analysis reports
- `projects/`: Saved project data
- `requirements.txt`: Project dependencies

## Analysis Categories

### 1. Financial Analysis
- Total development cost
- Expected revenue
- Development period
- Interest rates
- Equity requirements
- ROI and other financial metrics

### 2. Market Analysis
- Market size
- Growth rate
- Competition level
- Demand analysis
- Absorption rate

Additional categories (Location, Legal, Design, Political, Exit Strategy) will be implemented in future updates.

## Database Setup

1. Create a Supabase account at https://supabase.com
2. Create a new project
3. Get your project URL and API key
4. Add them to your `.env` file

## Deployment

1. Create a GitHub repository and push your code
2. Sign up for Streamlit Cloud (https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy your app

## Output

The tool generates:
1. Comprehensive text reports with:
   - Executive summary
   - Detailed category analysis
   - Recommendations
2. Visual representations:
   - Category score comparisons
   - Sensitivity analysis charts
   - Project comparison graphs

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
