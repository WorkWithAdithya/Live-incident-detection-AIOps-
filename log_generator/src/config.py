import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
LOG_INTERVAL_SECONDS = int(os.getenv("LOG_INTERVAL_SECONDS", 5))
LOG_RETENTION_HOURS = int(os.getenv("LOG_RETENTION_HOURS", 24))
CLEANUP_INTERVAL_SECONDS = int(os.getenv("CLEANUP_INTERVAL_SECONDS", 3600))

# Validate that critical config is present
if not DATABASE_URL:
    raise ValueError("❌ DATABASE_URL is not set. Please check your .env file.")