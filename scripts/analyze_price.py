"""
Script for running the trend analysis scanner engine locally
"""
import os
import dotenv
import sys
dotenv.load_dotenv()
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

neon_db_url = os.environ.get("NEON_DB_CONSTR")
neon_db_url = os.environ.get("NEON_DB_CONSTR")

engine = create_engine(neon_db_url)

# Get the directory of the script and add the project root to the Python path
script_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_directory, '..'))
sys.path.append(project_root)

import src.trend_analysis.scanner_engine
from src.models.models import StockRepository

if __name__ == '__main__':
    src.trend_analysis.scanner_engine.main(multiprocess=True)
    with Session(engine) as _session:
        entry_table = StockRepository(_session).get_entry_table(risk=500)
        entry_table.to_sql('entry', engine, if_exists='replace', index=False, chunksize=10000)
