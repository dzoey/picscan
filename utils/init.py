# Utils package initialization
from utils.logging_config import logger
from utils.metadata import (
    format_image_metadata,
    analyze_query_for_metadata_focus,
    is_rag_appropriate
)
from utils.image import extract_image_path, encode_images_to_base64


from .exif_utils import detect_exif_query_type, extract_exif_metadata, format_exif_context

# Export commonly used functions and objects
__all__ = [
    'logger',
    'format_image_metadata',
    'analyze_query_for_metadata_focus', 
    'is_rag_appropriate',
    'extract_image_path',
    'encode_images_to_base64',
    'extract_exif_metadata',
    'format_exif_context',
    'detect_exif_query_type',
]