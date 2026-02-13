import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
LOG_INTERVAL_SECONDS = int(os.getenv("LOG_INTERVAL_SECONDS", 5))

# Validate that critical config is present
if not DATABASE_URL:
    raise ValueError("❌ DATABASE_URL is not set. Please check your .env file.")