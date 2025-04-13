import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory configuration
BASE_DIR = os.environ.get("PICSCAN_BASE_DIR", str(Path(__file__).parent))

TEMP_UPLOADS_DIR=os.environ.get("PICSCAN_TEMP_UPLOADS", BASE_DIR+'/temp_uploads')

# Database configuration - using your specific paths
DB_PATH = os.environ.get("PICSCAN_DB_PATH", "/home/dzoey/projects/picscan/ragdb")
DB_COLLECTION_NAME = os.environ.get("PICSCAN_DB_COLLECTION", "image_rag")
# Collection names for text and image embeddings
TEXT_COLLECTION_NAME = os.environ.get("PICSCAN_TEXT_COLLECTION", f"{DB_COLLECTION_NAME}_text")
IMAGE_COLLECTION_NAME = os.environ.get("PICSCAN_IMAGE_COLLECTION", f"{DB_COLLECTION_NAME}_image")

# Model Configuration
DEFAULT_LLM_MODEL = os.environ.get("PICSCAN_DEFAULT_LLM", "llama3.2:1b")
SMALL_LLM_MODEL = os.environ.get("PICSCAN_SMALL_LLM", "llama3.2:1b")
VISION_LLM_MODEL = os.environ.get("PICSCAN_VISION_LLM", "granite3.2-vision:latest")
EMBEDDING_MODEL = os.environ.get("PICSCAN_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# LLM parameters
DEFAULT_TEMPERATURE = float(os.environ.get("PICSCAN_DEFAULT_TEMPERATURE", "0.1"))
DEFAULT_REPEAT_PENALTY = float(os.environ.get("PICSCAN_DEFAULT_REPEAT_PENALTY", "1.2"))
DEFAULT_NUM_PREDICT = int(os.environ.get("PICSCAN_DEFAULT_NUM_PREDICT", "2000"))
DEFAULT_TIMEOUT = int(os.environ.get("PICSCAN_DEFAULT_TIMEOUT", "60"))

# Classification parameters for use with ragapp
CLASSIFICATION_TEMPERATURE = float(os.environ.get("PICSCAN_CLASSIFICATION_TEMPERATURE", "0.1"))
CLASSIFICATION_REPEAT_PENALTY = float(os.environ.get("PICSCAN_CLASSIFICATION_REPEAT_PENALTY", "1.0"))
CLASSIFICATION_NUM_PREDICT = int(os.environ.get("PICSCAN_CLASSIFICATION_NUM_PREDICT", "100"))
CLASSIFICATION_TIMEOUT = int(os.environ.get("PICSCAN_CLASSIFICATION_TIMEOUT", "10"))

# Fallback models (in order of preference)
# For environment variable, use comma-separated list: "llama3.2:1b,llama3:8b,mistral:7b"
default_fallbacks = "llama3.2:1b,gemma3:1b,phi4-mini"
FALLBACK_MODELS = os.environ.get("PICSCAN_FALLBACK_MODELS", default_fallbacks).split(',')

# Logging configuration
LOG_LEVEL = os.environ.get("PICSCAN_LOG_LEVEL", "INFO")
LOG_FILE = os.environ.get("PICSCAN_LOG_FILE", os.path.join(BASE_DIR, "logs", "picscan.log"))

# Server configuration
SERVER_HOST = os.environ.get("PICSCAN_SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("PICSCAN_SERVER_PORT", "5000"))
DEBUG_MODE = os.environ.get("PICSCAN_DEBUG_MODE", "False").lower() in ("true", "1", "yes")

# Feature flags
ENABLE_LLM_CLASSIFICATION = os.environ.get("PICSCAN_ENABLE_LLM_CLASSIFICATION", "True").lower() in ("true", "1", "yes")
ENABLE_VISION_FEATURES = os.environ.get("PICSCAN_ENABLE_VISION", "True").lower() in ("true", "1", "yes")

# Vision model parameters for use with rag_builder
VISION_TEMPERATURE = float(os.environ.get("PICSCAN_VISION_TEMPERATURE", "0.1"))
VISION_REPEAT_PENALTY = float(os.environ.get("PICSCAN_VISION_REPEAT_PENALTY", "1.2"))
VISION_NUM_PREDICT = int(os.environ.get("PICSCAN_VISION_NUM_PREDICT", "2000"))
VISION_TIMEOUT = int(os.environ.get("PICSCAN_VISION_TIMEOUT", "60"))

# Geocoding configuration
GEOCODING_URL = os.environ.get("PICSCAN_GEOCODING_URL", "https://nominatim.openstreetmap.org/reverse")
GEOCODING_USER_AGENT = os.environ.get("PICSCAN_GEOCODING_USER_AGENT", "PicScan/1.0")
GEOCODING_RATE_LIMIT = float(os.environ.get("PICSCAN_GEOCODING_RATE_LIMIT", "1.0"))

LLM_CONTEXT_SIZE = int(os.environ.get("PICSCAN_LLM_CONTEXT_SIZE", "8192"))

# Flask settings
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "1") == "1"
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))

# Type conversion helper function for complex settings
def parse_list(env_value, default):
    """Parse a comma-separated environment variable into a list, with fallback to default"""
    value = os.environ.get(env_value)
    if value:
        return [item.strip() for item in value.split(',')]
    return default

# Example of a more complex configuration with the helper
SUPPORTED_IMAGE_TYPES = parse_list(
    "PICSCAN_IMAGE_TYPES", 
    [".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic"]
)
