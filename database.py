from supabase import create_client
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_KEY')
)

def upload_market_data(df: pd.DataFrame, table_name: str = 'market_data'):
    """
    Upload market data to Supabase
    """
    # Convert DataFrame to records
    records = df.to_dict('records')
    
    # Upload to Supabase
    try:
        data = supabase.table(table_name).insert(records).execute()
        print(f"Successfully uploaded {len(records)} records to {table_name}")
        return data
    except Exception as e:
        print(f"Error uploading data: {str(e)}")
        raise
