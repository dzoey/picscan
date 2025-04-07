import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database configuration
DB_PATH = os.getenv("DB_PATH", "/home/dzoey/projects/picscan/ragdb")
DB_COLLECTION_NAME = os.getenv("DB_COLLECTION_NAME", "image_rag")

# Model configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "60000"))

# Flask settings
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "1") == "1"
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))

# Computed paths
TEMP_UPLOADS_DIR = Path("temp_uploads")
TEMP_UPLOADS_DIR.mkdir(exist_ok=True)
