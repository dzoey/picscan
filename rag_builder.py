import os
import glob
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import requests
import json
from PIL import Image
from PIL.ExifTags import TAGS
from PIL.TiffImagePlugin import IFDRational
from sentence_transformers import SentenceTransformer
import chromadb
import argparse
import ollama
import torch
import torchvision
import torchvision.transforms as transforms
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag_builder")

# Default configuration
DEFAULT_CONFIG = {
    "geocoding_url": "https://nominatim.openstreetmap.org/reverse",
    "geocoding_user_agent": "RAGBuilder/1.0",
    "geocoding_rate_limit_seconds": 1.0,
    "vlm_model": "granite3.2-vision",
    "embedding_model": "all-MiniLM-L6-v2",
    "db_collection_name": "image_rag",
    "image_extensions": [".jpg", ".jpeg", ".png", ".gif"]
}

@dataclass
class GeoLocation:
    """Structure for geographic location data"""
    latitude: float = 0.0
    longitude: float = 0.0
    country: str = ""
    state: str = ""
    city: str = ""
    address: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'country': self.country,
            'state': self.state,
            'city': self.city,
            'address': self.address
        }
    
    def is_valid(self) -> bool:
        """Check if location has valid coordinates"""
        return self.latitude != 0.0 or self.longitude != 0.0

class ExifJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle EXIF data types"""
    def default(self, obj):
        if isinstance(obj, IFDRational):
            # Convert IFDRational to a float or string representation
            return float(obj)
        elif isinstance(obj, (Fraction, complex)):
            return str(obj)
        elif isinstance(obj, bytes):
            try:
                return obj.decode('utf-8', errors='replace')
            except:
                return str(obj)
        elif isinstance(obj, (np.ndarray, np.number)):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

def encode_exif_to_json(exif_data: Dict) -> str:
    """Encode EXIF data to JSON string using custom encoder"""
    # Convert all keys to strings to ensure consistent types for sorting
    string_keyed_data = {str(k): v for k, v in exif_data.items()}
    return json.dumps(string_keyed_data, cls=ExifJSONEncoder, sort_keys=True)

def convert_gps_to_decimal(gps_coords: Tuple[Any, Any, Any], gps_ref: str) -> float:
    """
    Converts GPS coordinates from the EXIF format (degrees, minutes, seconds) to decimal format.
    
    Args:
        gps_coords: The GPS coordinates tuple (degrees, minutes, seconds)
        gps_ref: The GPS reference (N, S, E, W)
    
    Returns:
        Decimal coordinate value
    """
    try:
        # Handle IFDRational values by converting to float
        degrees = float(gps_coords[0])
        minutes = float(gps_coords[1])
        seconds = float(gps_coords[2])
        
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        
        if gps_ref in ['S', 'W']:
            decimal = -decimal
        
        return decimal
    except (ValueError, TypeError, IndexError) as e:
        logger.warning(f"Error converting GPS coordinates: {e}")
        return 0.0

def get_location_from_gps(latitude: float, longitude: float, config: Dict[str, Any] = None) -> GeoLocation:
    """
    Gets location details from latitude and longitude using geocoding.
    
    Args:
        latitude: The latitude in decimal format
        longitude: The longitude in decimal format
        config: Configuration dictionary with options
    
    Returns:
        GeoLocation object with location details
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    if not latitude or not longitude:
        return GeoLocation()
    
    location = GeoLocation(latitude=latitude, longitude=longitude)
    
    try:
        # Using OpenStreetMap's Nominatim service
        params = {
            "format": "json",
            "lat": str(latitude),
            "lon": str(longitude),
            "zoom": "18",
            "addressdetails": "1"
        }
        
        headers = {
            "User-Agent": config["geocoding_user_agent"]
        }
        
        response = requests.get(
            config["geocoding_url"], 
            params=params,
            headers=headers,
            timeout=10
        )
        
        # Add a delay to respect service's usage policy
        time.sleep(config["geocoding_rate_limit_seconds"])
        
        if response.status_code != 200:
            logger.warning(f"Geocoding service returned status code {response.status_code}")
            return location
        
        data = response.json()
        
        # Extract relevant information
        address = data.get('display_name', '')
        address_details = data.get('address', {})
        country = address_details.get('country', '')
        state = address_details.get('state', '')
        
        # City could be in different fields depending on location type
        city = address_details.get('city', 
               address_details.get('town',
               address_details.get('village',
               address_details.get('hamlet', ''))))
        
        location.address = address
        location.country = country
        location.state = state
        location.city = city
        
        return location
    except requests.RequestException as e:
        logger.warning(f"Network error during geocoding: {e}")
        return location
    except (ValueError, KeyError) as e:
        logger.warning(f"Error parsing geocoding response: {e}")
        return location
    except Exception as e:
        logger.warning(f"Unexpected error during geocoding: {e}")
        return location

def decode_bytes_safely(value: Any) -> Any:
    """Safely decode bytes to string or return original value"""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return str(value)
    elif isinstance(value, IFDRational):
        # Handle IFDRational values
        return float(value)
    elif isinstance(value, dict):
        return {k: decode_bytes_safely(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [decode_bytes_safely(item) for item in value]
    else:
        return value

def extract_gps_info(gps_data: Dict[int, Any]) -> Optional[GeoLocation]:
    """Extract GPS information from EXIF GPS data"""
    try:
        # Extract GPS data
        lat_ref = gps_data.get(1)  # 1 = GPSLatitudeRef
        lat_ref = decode_bytes_safely(lat_ref) if lat_ref else 'N'
        
        lat = gps_data.get(2)  # 2 = GPSLatitude
        if not lat:
            return None
        
        lon_ref = gps_data.get(3)  # 3 = GPSLongitudeRef
        lon_ref = decode_bytes_safely(lon_ref) if lon_ref else 'E'
        
        lon = gps_data.get(4)  # 4 = GPSLongitude
        if not lon:
            return None
        
        # Convert to decimal format
        latitude = convert_gps_to_decimal(lat, lat_ref)
        longitude = convert_gps_to_decimal(lon, lon_ref)
        
        if latitude == 0.0 and longitude == 0.0:
            return None
            
        # Get location details from geocoding
        return get_location_from_gps(latitude, longitude)
    except Exception as e:
        logger.warning(f"Error extracting GPS info: {e}")
        return None

def get_exif_data(image_path: str) -> Optional[Dict[str, Any]]:
    """
    Extracts the EXIF data from an image file and returns it as a dictionary.

    Args:
        image_path: The path to the image file.

    Returns:
        A dictionary containing the EXIF data or None if no data found
    """
    try:
        # Open the image file
        with Image.open(image_path) as image:
            # Get the EXIF data
            exif_data = image._getexif()
            
            if exif_data is None:
                logger.info(f"No EXIF data found in {image_path}")
                return None
            
            # Convert EXIF data to a readable dictionary
            exif_dict = {}
            for tag_id, value in exif_data.items():
                # Get the tag name, if possible
                tag = TAGS.get(tag_id, tag_id)
                
                # Special handling for GPSInfo tag
                if tag == 'GPSInfo':
                    location = extract_gps_info(value)
                    if location and location.is_valid():
                        exif_dict[tag] = location.to_dict()
                        logger.info(f"Processed GPS data for {image_path}: {exif_dict[tag]}")
                    else:
                        # If GPS extraction fails, just clean the bytes
                        exif_dict[tag] = decode_bytes_safely(value)
                else:
                    # Process other tags by cleaning bytes
                    exif_dict[tag] = decode_bytes_safely(value)
            
            return exif_dict
    
    except FileNotFoundError:
        logger.warning(f"File not found: {image_path}")
        return None
    except PermissionError:
        logger.warning(f"Permission denied accessing file: {image_path}")
        return None
    except Exception as e:
        logger.warning(f"Error extracting EXIF data from {image_path}: {e}")
        return None

def generate_image_description(image_path: str, model_name: str) -> Optional[str]:
    """
    Generates a description for an image using a vision model.
    
    Args:
        image_path: Path to the image file.
        model_name: The name of the Ollama model to use.
        
    Returns:
        A string containing the image description, or None if an error occurred.
    """
    try:
        filename = os.path.basename(image_path)
        logger.info(f"Generating description for {image_path}")
        
        prompt = f"""Provide a comprehensive description of this image that would help with later retrieval through text search.

Please include:
1. A concise summary of the main subject(s) and scene (1-2 sentences)
2. Key visual elements (objects, setting, activities)
3. Notable characteristics (colors, lighting, composition, style)
4. Any visible text, signs, or distinctive features
5. Scene type (indoor/outdoor, urban/rural, natural/artificial)
6. Environmental context (weather, time of day, season if apparent)
7. Visual indicators of location or setting if present

For context, the filename is: {filename}

Format your response as a cohesive, detailed paragraph that would be useful for image retrieval through semantic search."""
        
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user", 
                    "content": prompt,
                    "images": [image_path]
                }
            ]
        )
        
        return response.message.content
    except ollama.ResponseError as e:
        logger.error(f"Ollama response error for {image_path}: {e}")
        return None
    except (KeyError, IndexError) as e:
        logger.error(f"Error parsing VLM response for {image_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error generating description for {image_path}: {e}")
        return None
    
def get_image_features(image_path: str) -> Optional[np.ndarray]:
    """
    Extract image features using a pre-trained CNN model
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Numpy array of image features
    """
    try:
        # Use a more modern approach for loading pretrained models
        image_model = torchvision.models.densenet121(weights="IMAGENET1K_V1")
        image_model.eval()  # Set to evaluation mode
        
        image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        with Image.open(image_path) as image:
            image_tensor = image_transform(image)
        
        with torch.no_grad():  # Disable gradient calculation for inference
            image_features = image_model(image_tensor.unsqueeze(0)).squeeze(0).numpy()
            
        return image_features
    except Exception as e:
        logger.error(f"Error extracting image features from {image_path}: {e}")
        return None

def find_images(directory: str, extensions: List[str]) -> List[str]:
    """Find all images with specified extensions in directory"""
    image_paths = []
    directory_path = Path(directory)
    
    for ext in extensions:
        if not ext.startswith("."):
            ext = f".{ext}"
            
        # Use Path for more reliable path handling
        pattern = f"**/*{ext}"
        image_paths.extend([str(p) for p in directory_path.glob(pattern)])
    
    logger.info(f"Found {len(image_paths)} images in {directory}")
    return image_paths

import hashlib

def get_image_id(image_path: str) -> str:
    """Generate a consistent ID for an image based on its path"""
    return f"img_{hashlib.md5(image_path.encode('utf-8')).hexdigest()}"

def build_rag_data(directory: str, config: Dict[str, Any] = None):
    """
    Build RAG data from images in a directory, skipping already processed images.
    
    Args:
        directory: Directory containing images
        config: Configuration dictionary with options
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    # Find all images in directory
    image_paths = find_images(
        directory, 
        [ext.lstrip(".") for ext in config["image_extensions"]]
    )
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=config.get("db_path", "ragdb"))
    
    # Create separate collections for text and image embeddings
    text_collection = client.get_or_create_collection(f"{config['db_collection_name']}_text")
    image_collection = client.get_or_create_collection(f"{config['db_collection_name']}_image")
    
    # Get existing file paths from the database
    existing_paths = set()
    try:
        # Query the collection to get all metadata
        results = text_collection.get(include=["metadatas"])
        if results and 'metadatas' in results and results['metadatas']:
            for metadata in results['metadatas']:
                if metadata and 'path' in metadata:
                    existing_paths.add(metadata['path'])
        
        logger.info(f"Found {len(existing_paths)} already processed images in the database")
    except Exception as e:
        logger.warning(f"Error retrieving existing items: {e}")
    
    # Initialize sentence transformer
    sentence_model = SentenceTransformer(config["embedding_model"])
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for image_path in image_paths:
        try:
            # Check if this image has already been processed
            if image_path in existing_paths:
                logger.info(f"Skipping already processed image: {image_path}")
                skipped_count += 1
                continue
            
            # Generate a consistent ID for this image
            image_id = get_image_id(image_path)
                
            # Get image description
            description = generate_image_description(
                image_path, 
                config["vlm_model"]
            )
            
            if not description:
                continue
                
            # Get EXIF data
            exif = get_exif_data(image_path)
            
            # Prepare document text - use only the description
            text = description
            
            # Prepare document metadata
            document = {
                "path": image_path, 
                "description": description
            }
            
            # Add individual EXIF fields as separate metadata entries
            if exif:
                # Store a flattened version of key EXIF fields in metadata
                for key, value in exif.items():
                    # Convert complex types to strings to ensure they can be stored in metadata
                    if isinstance(value, dict):
                        # Special handling for nested structures like GPSInfo
                        for subkey, subvalue in value.items():
                            safe_key = f"exif_{key}_{subkey}".replace(" ", "_")
                            document[safe_key] = str(subvalue)
                    else:
                        # Make sure the key is a valid string (no spaces) and prefixed for clarity
                        safe_key = f"exif_{key}".replace(" ", "_")
                        document[safe_key] = str(value)
                
                # Also store the full EXIF as JSON for reference/completeness
                document["exif_json"] = encode_exif_to_json(exif)
            
            # Generate text embedding only from the description
            sentence_embedding = sentence_model.encode([text])[0]
            
            # Add text embedding to text collection
            text_collection.add(
                embeddings=[sentence_embedding.tolist()],
                documents=[text],  # Just the description
                metadatas=[document],  # Metadata now includes EXIF data as individual fields
                ids=[image_id]
            )
            
            # Extract image features
            image_features = get_image_features(image_path)
            if image_features is not None:
                # Use the same metadata for consistency
                image_collection.add(
                    embeddings=[image_features.tolist()],
                    documents=[description],  # Just the description without EXIF
                    metadatas=[document],  # Same metadata as text collection
                    ids=[image_id]
                )
            
            processed_count += 1
            logger.info(f"Processed {processed_count}, skipped {skipped_count}: {image_path}")
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}", exc_info=True)
            error_count += 1
    
    total = processed_count + skipped_count + error_count
    logger.info(f"Completed: {processed_count} processed, {skipped_count} skipped, {error_count} errors out of {total} images.")
    
def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="Build RAG data from images.")
    
    parser.add_argument(
        "image_dir", 
        help="The directory containing the images."
    )
    
    parser.add_argument(
        "--db_path", 
        default="ragdb",
        help="The database path to store vectors (default: ragdb)"
    )
    
    parser.add_argument(
        "--vlm", 
        default=DEFAULT_CONFIG["vlm_model"],
        help=f"The VLM model to use (default: {DEFAULT_CONFIG['vlm_model']})"
    )
    
    parser.add_argument(
        "--log", 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    logger.setLevel(getattr(logging, args.log))
    
    # Set up configuration
    config = DEFAULT_CONFIG.copy()
    config["vlm_model"] = args.vlm
    config["db_path"] = args.db_path
    
    # Build RAG data
    build_rag_data(args.image_dir, config)

if __name__ == "__main__":
    main()