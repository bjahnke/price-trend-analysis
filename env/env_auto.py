"""
Desc:
env_auto.py is generated from .env by the `invoke buildenvpy` task.
it's purpose is to provide IDE support for environment variables.
"""

import os
from dotenv import load_dotenv
load_dotenv()


PROJECT_NAME = os.environ.get('PROJECT_NAME')
PACKAGE_NAME = os.environ.get('PACKAGE_NAME')
NEON_DB_CONSTR = os.environ.get('NEON_DB_CONSTR')
