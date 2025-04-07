import os
import re
import base64
from io import BytesIO
from typing import Optional, List
from pathlib import Path
from PIL import Image

from utils.logging_config import logger

def extract_image_path(text: str) -> Optional[str]:
    """
    Extract a file path from text that might contain an image reference
    
    Args:
        text: Text line that might contain an image reference
        
    Returns:
        Extracted file path if found, None otherwise
    """
    # Look for file paths with image extensions
    file_path_patterns = [
        r'(?:^|\s)(/[\w\-./]+\.(?:jpg|jpeg|png|gif))(?:$|\s|[,.\'"\)])',  # Unix absolute path
        r'(?:^|\s)([A-Za-z]:\\[\w\-\\. ]+\.(?:jpg|jpeg|png|gif))(?:$|\s|[,.\'"\)])',  # Windows absolute path
        r'(?:^|\s)([\w\-./]+\.(?:jpg|jpeg|png|gif))(?:$|\s|[,.\'"\)])'  # Relative path
    ]
    
    for pattern in file_path_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):  # Handle multiple capture groups
                for m in match:
                    if m and os.path.exists(m):
                        logger.debug(f"Found valid image path: {m}")
                        return m
            elif os.path.exists(match):
                logger.debug(f"Found valid image path: {match}")
                return match
    
    return None

def encode_images_to_base64(image_paths: List[str]):
    """
    Convert image file paths to base64 representations for web display
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        List of dictionaries with image data in base64
    """
    images_base64 = []
    
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                buffer = BytesIO()
                img_format = img.format or 'JPEG'
                img.save(buffer, format=img_format)
                img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                images_base64.append({
                    "path": img_path,
                    "data": img_b64,
                    "format": img_format.lower()
                })
                logger.debug(f"Encoded image to base64: {img_path}")
        except Exception as e:
            logger.error(f"Error encoding image {img_path}: {e}")
    
    return images_base64