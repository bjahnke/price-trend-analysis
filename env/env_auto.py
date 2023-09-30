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
IMAGE_NAME = os.environ.get('IMAGE_NAME')
DOCKER_TOKEN = os.environ.get('DOCKER_TOKEN')
DOCKER_USERNAME = os.environ.get('DOCKER_USERNAME')
GITHUB_USERNAME = os.environ.get('GITHUB_USERNAME')
GITHUB_PASSWORD = os.environ.get('GITHUB_PASSWORD')
GCR_PROJECT_ID = os.environ.get('GCR_PROJECT_ID')
MONGO_URL = os.environ.get('MONGO_URL')
MONGO_API_KEY = os.environ.get('MONGO_API_KEY')
