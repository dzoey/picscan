# app.py
import os
import base64
import logging
import json
import re
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from io import BytesIO

from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import ollama
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_app")

app = Flask(__name__)

# Configuration
CONFIG = {
    "db_path": "ragdb",
    "db_collection_name": "image_rag",
    "embedding_model": "all-MiniLM-L6-v2",
    "ollama_base_url": "http://localhost:11434",
    "max_context_chars": 60000,  # ~15k tokens, adjust based on model
}

# Initialize global variables
sentence_model = None
client = None
text_collection = None
image_collection = None

def initialize_app():
    """Initialize the application components"""
    global sentence_model, client, text_collection, image_collection
    
    logger.debug("Initializing sentence transformer and ChromaDB collections")
    try:
        sentence_model = SentenceTransformer(CONFIG["embedding_model"])
        
        # Initialize ChromaDB
        client = chromadb.PersistentClient(path=CONFIG["db_path"], settings=Settings(anonymized_telemetry=False))
        text_collection = client.get_or_create_collection(f"{CONFIG['db_collection_name']}_text")
        image_collection = client.get_or_create_collection(f"{CONFIG['db_collection_name']}_image")
        logger.debug("Initialization complete")
    except Exception as e:
        logger.error(f"Error during initialization: {e}", exc_info=True)
        raise

# Initialize at startup
initialize_app()

def format_image_metadata(metadata: Dict) -> str:
    """
    Format image metadata into a useful text representation
    
    Args:
        metadata: Dictionary containing image metadata
        
    Returns:
        Formatted metadata as a string
    """
    formatted_text = "Image Metadata:\n"
    
    # Extract location information if available
    gps_info = metadata.get('GPSInfo', {})
    if gps_info and isinstance(gps_info, dict):
        formatted_text += "Location:\n"
        
        if 'address' in gps_info:
            formatted_text += f"  Address: {gps_info['address']}\n"
        if 'city' in gps_info:
            formatted_text += f"  City: {gps_info['city']}\n"
        if 'state' in gps_info:
            formatted_text += f"  State: {gps_info['state']}\n"
        if 'country' in gps_info:
            formatted_text += f"  Country: {gps_info['country']}\n"
        
        # Add coordinates if available
        if 'latitude' in gps_info and 'longitude' in gps_info:
            formatted_text += f"  GPS Coordinates: {gps_info['latitude']}, {gps_info['longitude']}\n"
    
    # Extract date/time information
    datetime_original = metadata.get('DateTimeOriginal')
    if datetime_original:
        formatted_text += f"Date/Time: {datetime_original}\n"
    elif metadata.get('DateTime'):
        formatted_text += f"Date/Time: {metadata.get('DateTime')}\n"
    
    # Camera information
    make = metadata.get('Make')
    model = metadata.get('Model')
    if make or model:
        camera_info = "Camera: "
        if make:
            camera_info += make
        if model:
            camera_info += f" {model}"
        formatted_text += camera_info + "\n"
    
    # Technical details (selective)
    technical_details = []
    
    aperture = metadata.get('FNumber')
    if aperture:
        technical_details.append(f"Aperture: f/{aperture}")
    
    iso = metadata.get('ISOSpeedRatings')
    if iso:
        technical_details.append(f"ISO: {iso}")
    
    exposure = metadata.get('ExposureTime')
    if exposure:
        # Format as fraction if needed
        if exposure < 1:
            denominator = round(1/exposure)
            technical_details.append(f"Exposure: 1/{denominator} sec")
        else:
            technical_details.append(f"Exposure: {exposure} sec")
    
    focal_length = metadata.get('FocalLength')
    if focal_length:
        technical_details.append(f"Focal Length: {focal_length}mm")
    
    if technical_details:
        formatted_text += "Technical Details: " + ", ".join(technical_details) + "\n"
    
    return formatted_text

def get_all_states_from_metadata() -> List[str]:
    """
    Extract all unique state values from the entire metadata collection
    
    Returns:
        List of unique state names
    """
    try:
        # Get ALL entries from the collection - not just semantically relevant ones
        all_entries = text_collection.get(include=["metadatas"])
        
        if not all_entries or "metadatas" not in all_entries or not all_entries["metadatas"]:
            logger.warning("No metadata found in collection")
            return []
            
        all_metadatas = all_entries["metadatas"]
        logger.debug(f"Retrieved {len(all_metadatas)} metadata entries")
        
        # Extract states
        states = set()
        for metadata in all_metadatas:
            # Check if metadata exists and has GPSInfo
            if metadata and "GPSInfo" in metadata and isinstance(metadata["GPSInfo"], dict):
                gps_info = metadata["GPSInfo"]
                # Check if state exists in GPSInfo
                if "state" in gps_info and gps_info["state"]:
                    states.add(gps_info["state"])
        
        states_list = sorted(list(states))
        logger.debug(f"Found {len(states_list)} unique states: {states_list}")
        return states_list
    except Exception as e:
        logger.error(f"Error extracting states from metadata: {e}", exc_info=True)
        return []

 # Add this new function to the codebase
def get_all_metadata_values(field_path: str) -> List[str]:
    """
    Extract all unique values for a specific metadata field across the entire database
    
    Args:
        field_path: Dot-separated path to metadata field (e.g. 'GPSInfo.state')
        
    Returns:
        List of unique values for that field
    """
    try:
        # Get all entries from the collection
        all_entries = text_collection.get(include=["metadatas"])
        all_metadatas = all_entries.get("metadatas", [])
        
        unique_values = set()
        field_parts = field_path.split('.')
        
        for metadata in all_metadatas:
            # Skip if metadata is None
            if not metadata:
                continue
                
            # Navigate nested dictionary
            current_dict = metadata
            valid = True
            
            for i, part in enumerate(field_parts):
                # Handle the case where field might be missing
                if part not in current_dict:
                    valid = False
                    break
                    
                # If we're at the last part, add the value
                if i == len(field_parts) - 1:
                    value = current_dict[part]
                    # Only add string values
                    if isinstance(value, str) and value.strip():
                        unique_values.add(value.strip())
                # Otherwise, navigate deeper
                else:
                    # Ensure the next level is a dictionary
                    if not isinstance(current_dict[part], dict):
                        valid = False
                        break
                    current_dict = current_dict[part]
            
        return sorted(list(unique_values))
    except Exception as e:
        logger.error(f"Error getting metadata values: {e}", exc_info=True)
        return []   

def get_unique_metadata_values(field_name: str) -> List[str]:
    """
    Extract all unique values for any metadata field
    
    Args:
        field_name: The exact field name in the metadata
        
    Returns:
        List of unique values for that field
    """
    try:
        # Get ALL entries from the collection
        all_entries = text_collection.get(include=["metadatas"])
        
        if not all_entries or "metadatas" not in all_entries:
            logger.warning("No metadata found in collection")
            return []
            
        all_metadatas = all_entries["metadatas"]
        logger.debug(f"Retrieved {len(all_metadatas)} metadata entries")
        
        # Extract values
        values = set()
        value_count = 0
        for metadata in all_metadatas:
            # Skip empty metadata
            if not metadata:
                continue
                
            # Look for the specified field
            if field_name in metadata and metadata[field_name]:
                value = str(metadata[field_name]).strip()
                if value:  # Only add non-empty values
                    values.add(value)
                    value_count += 1
        
        logger.debug(f"Found {value_count} entries for field '{field_name}' across {len(all_metadatas)} metadata records")
        values_list = sorted(list(values))
        return values_list
    except Exception as e:
        logger.error(f"Error extracting metadata values: {e}", exc_info=True)
        return []

def analyze_query_for_metadata_focus(query_text: str) -> Optional[str]:
    """
    Analyze the query to detect if it's focused on specific metadata
    
    Args:
        query_text: Original user query
        
    Returns:
        Type of metadata to focus on, or None if no specific focus
    """
    query_lower = query_text.lower()
    
    # Check for location-related queries
    location_terms = ['where', 'location', 'place', 'city', 'country', 'address', 'gps']
    if any(term in query_lower for term in location_terms):
        return "location"
    
    # Check for time-related queries
    time_terms = ['when', 'date', 'time', 'year', 'month', 'day']
    if any(term in query_lower for term in time_terms):
        return "date/time"
    
    # Check for camera/technical queries
    camera_terms = ['camera', 'lens', 'make', 'model', 'iso', 'aperture', 'f-stop', 
                    'exposure', 'focal length', 'resolution', 'megapixel']
    if any(term in query_lower for term in camera_terms):
        return "camera settings"
    
    # No specific metadata focus detected
    return None

def get_available_models() -> List[str]:
    """Get list of available models from Ollama"""
    try:
        response = ollama.list()
        logger.debug(f"Raw Ollama response structure: {response}")
        
        # Handle the ListResponse object from Ollama 0.6
        model_names = []
        
        # Check if models attribute exists (expected in Ollama 0.6)
        if hasattr(response, 'models'):
            models = response.models
            logger.debug(f"Found {len(models)} models in response.models")
            
            # Extract model names from each model object
            for model in models:
                # Try to access the model attribute directly
                if hasattr(model, 'model'):
                    model_names.append(model.model)
                    logger.debug(f"Added model: {model.model}")
                elif hasattr(model, 'name'):
                    model_names.append(model.name)
                    logger.debug(f"Added model: {model.name}")
                else:
                    # Log the model object's attributes for debugging
                    logger.warning(f"Could not find model name. Available attributes: {dir(model)}")
        
        # If no models were found, provide a fallback
        if not model_names:
            logger.warning("No models extracted from response")
            return ["default"]
            
        return model_names
    except Exception as e:
        logger.error(f"Error fetching models from Ollama: {e}", exc_info=True)
        return ["default"]

def get_rag_context(query_text: str, top_k: int = 5, max_chars: int = 60000) -> str:
    """
    Retrieve relevant context from RAG database with size limits
    
    Args:
        query_text: User's query text
        top_k: Maximum number of results to retrieve
        max_chars: Maximum characters to include in context (roughly 15k tokens)
        
    Returns:
        Relevant context as a string, limited to max_chars
    """
    try:
        logger.debug(f"Generating embedding for query: {query_text}")
        # Generate embedding for the query
        query_embedding = sentence_model.encode([query_text])[0]
        
        logger.debug(f"Querying text collection with embedding")
        # Query the text collection
        results = text_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas"]
        )
        
        if not results or 'documents' not in results or not results['documents'] or not results['documents'][0]:
            logger.warning("No relevant documents found in RAG database")
            return ""
        
        # Format the context with size limits
        context = "Relevant context from knowledge base:\n\n"
        current_size = len(context)
        
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            # Extract image path
            image_path = metadata.get('path', 'Unknown path')
            
            # Format the metadata in a structured way
            formatted_metadata = format_image_metadata(metadata)
            
            # Create document header with metadata
            doc_header = f"Document {i+1}:\nSource: {image_path}\n{formatted_metadata}\nContent:\n"
            doc_footer = "\n\n"
            doc_size = len(doc_header) + len(doc) + len(doc_footer)
            
            # Check if adding this document would exceed our limit
            if current_size + doc_size > max_chars:
                # Calculate how much of the document we can include
                available_space = max_chars - current_size - len(doc_header) - len(doc_footer)
                
                if available_space > 100:  # Only include if we can add at least 100 chars
                    # Truncate the document to fit
                    truncated_doc = doc[:available_space]
                    if len(truncated_doc) < len(doc):
                        truncated_doc += "... [truncated]"
                    
                    context += doc_header + truncated_doc + doc_footer
                    logger.debug(f"Added truncated document {i+1} ({len(truncated_doc)} chars)")
                
                # We've reached our size limit
                break
            
            # Add the full document if it fits
            context += doc_header + doc + doc_footer
            current_size += doc_size
            logger.debug(f"Added document {i+1} ({doc_size} chars)")
        
        logger.debug(f"Generated context with {len(context)} characters (~{len(context)//4} tokens)")
        return context
    except Exception as e:
        logger.error(f"Error retrieving RAG context: {e}", exc_info=True)
        return ""

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

def query_exif_field(field_path: str) -> List[str]:
    """
    Query the database for unique values of any EXIF field
    
    Args:
        field_path: The field path (e.g., 'GPSInfo_state' or 'Make')
        
    Returns:
        List of unique values for that field
    """
    try:
        # Add the exif_ prefix to match database structure
        full_field_name = f"exif_{field_path}"
        
        # Get all entries with metadata - we'll filter in Python code
        results = text_collection.get(include=["metadatas"])
        
        if not results or "metadatas" not in results or not results["metadatas"]:
            logger.debug(f"No entries with metadata found")
            return []
            
        # Extract unique values for the specific field
        values = set()
        field_count = 0
        total_entries = len(results["metadatas"])
        
        for metadata in results["metadatas"]:
            if metadata and full_field_name in metadata and metadata[full_field_name]:
                value = str(metadata[full_field_name]).strip()
                if value:
                    values.add(value)
                    field_count += 1
                    
        values_list = sorted(list(values))
        logger.debug(f"Found {field_count}/{total_entries} entries with {field_path}")
        logger.debug(f"Found {len(values_list)} unique values for {field_path}: {values_list}")
        return values_list
    except Exception as e:
        logger.error(f"Error querying {field_path}: {e}", exc_info=True)
        return []

def process_query(user_text: str, image_path: Optional[str], model_name: str) -> Dict[str, Any]:
    try:
        logger.debug(f"Processing query with text: '{user_text}', image: {image_path}, model: {model_name}")
        
        query_lower = user_text.lower()
        
        # Define mapping of query terms to EXIF fields
        exif_field_mappings = {
            # Location fields
            "states": {
                "field": "GPSInfo_state",
                "patterns": ["states", "which states", "what states", "list states"]
            },
            "cities": {
                "field": "GPSInfo_city",
                "patterns": ["cities", "which cities", "what cities", "list cities"]
            },
            "countries": {
                "field": "GPSInfo_country",
                "patterns": ["countries", "which countries", "what countries", "list countries"]
            },
            "addresses": {
                "field": "GPSInfo_address",
                "patterns": ["addresses", "locations", "where were", "list addresses"]
            },
            
            # Camera fields
            "camera brands": {
                "field": "Make",
                "patterns": ["camera brands", "camera makes", "manufacturers"]
            },
            "camera models": {
                "field": "Model",
                "patterns": ["camera models", "what cameras", "which cameras", "list cameras"]
            },
            
            # Technical fields
            "focal lengths": {
                "field": "FocalLength",
                "patterns": ["focal lengths", "focal length", "lens focal"],
                "format": lambda v: f"{v}mm"
            },
            "apertures": {
                "field": "FNumber",
                "patterns": ["apertures", "f-stops", "fnumber"],
                "format": lambda v: f"f/{v}"
            },
            "ISO values": {
                "field": "ISOSpeedRatings",
                "patterns": ["iso values", "iso settings", "list iso"]
            },
            "exposure times": {
                "field": "ExposureTime",
                "patterns": ["shutter speeds", "exposure times", "list exposure"],
                "format": lambda v: f"1/{int(1/float(v))} sec" if float(v) < 1 else f"{v} sec"
            },
            
            # Time fields
            "dates": {
                "field": "DateTime",
                "patterns": ["dates", "when", "list dates"]
            },
            
            # Software fields
            "software": {
                "field": "Software",
                "patterns": ["software", "apps", "applications", "list software"]
            }
        }
        
        # Check for "list all fields" type query
        if any(term in query_lower for term in ["list all exif", "all exif fields", "what exif"]):
            # Get a sample of entries to analyze
            results = text_collection.get(limit=5, include=["metadatas"])
            
            if results and "metadatas" in results and results["metadatas"]:
                all_fields = set()
                for metadata in results["metadatas"]:
                    if metadata:
                        for field in metadata.keys():
                            if field.startswith("exif_") and field != "exif_json":
                                clean_field = field[5:]  # Remove exif_ prefix
                                all_fields.add(clean_field)
                
                if all_fields:
                    grouped_fields = {}
                    for field in all_fields:
                        if "_" in field:
                            prefix, rest = field.split("_", 1)
                            if prefix not in grouped_fields:
                                grouped_fields[prefix] = []
                            grouped_fields[prefix].append(rest)
                        else:
                            if "General" not in grouped_fields:
                                grouped_fields["General"] = []
                            grouped_fields["General"].append(field)
                    
                    response = "Available EXIF fields in the database:\n\n"
                    for group, fields in sorted(grouped_fields.items()):
                        response += f"**{group}**\n"
                        for field in sorted(fields):
                            response += f"- {field}\n"
                        response += "\n"
                    
                    return {
                        "text": response,
                        "image_references": []
                    }
            
            return {
                "text": "I couldn't find any EXIF fields in the database.",
                "image_references": []
            }
        
        # Check for specific field queries
        for field_name, config in exif_field_mappings.items():
            if any(pattern in query_lower for pattern in config["patterns"]):
                values = query_exif_field(config["field"])
                
                if values:
                    # Apply formatting if specified
                    if "format" in config and callable(config["format"]):
                        try:
                            formatted_values = [config["format"](v) for v in values]
                        except Exception as e:
                            logger.warning(f"Error formatting values: {e}")
                            formatted_values = values
                    else:
                        formatted_values = values
                    
                    response = f"The images in the database have the following {field_name}:\n\n"
                    response += "\n".join([f"- {value}" for value in formatted_values])
                    return {
                        "text": response,
                        "image_references": []
                    }
                else:
                    return {
                        "text": f"I couldn't find any {field_name} information in the database.",
                        "image_references": []
                    }
        
        # Continue with existing semantic search code...
        
        # Analyze the query for metadata focus
        metadata_focus = analyze_query_for_metadata_focus(user_text)
        logger.debug(f"Detected metadata focus: {metadata_focus}")
        
        # Get RAG context
        context = get_rag_context(user_text, max_chars=CONFIG["max_context_chars"])
        
        # Create the complete prompt
        if context:
            # Base prompt
            prompt = f"""The following is relevant context that may help answer the query. 
Each document includes both content and detailed image metadata:

{context}

User query: {user_text}

Please provide a detailed and helpful response based on this information."""

            # Add specific instructions based on metadata focus if detected
            if metadata_focus:
                prompt += f"\n\nPay special attention to the {metadata_focus} information in the metadata when answering."
        else:
            prompt = user_text
            
        logger.debug(f"Generated prompt with {len(context)} characters of context")
        
        # Prepare the message
        message = {
            "role": "user",
            "content": prompt
        }
        
        # Add image if provided
        if image_path:
            message["images"] = [image_path]
            logger.debug(f"Added image to message: {image_path}")
        
        # Query the model
        logger.debug(f"Sending query to Ollama model: {model_name}")
        response = ollama.chat(
            model=model_name,
            options={'num_ctx': 20000},
            messages=[message]
        )
        
        response_text = response["message"]["content"]
        logger.debug(f"Received response with {len(response_text)} characters")
        
        # Look for image file references in the response
        image_references = []
        for line in response_text.split('\n'):
            # Check for image extensions
            if any(ext in line.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                potential_path = extract_image_path(line)
                if potential_path and os.path.exists(potential_path):
                    logger.debug(f"Found image reference in response: {potential_path}")
                    image_references.append(potential_path)
        
        return {
            "text": response_text,
            "image_references": image_references
        }
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return {
            "text": f"Error processing your query: {str(e)}",
            "image_references": []
        }

@app.route('/')
def index():
    """Render the main page"""
    try:
        models = get_available_models()
        logger.debug(f"Rendering index page with {len(models)} models")
        return render_template('index.html', models=models)
    except Exception as e:
        logger.error(f"Error rendering index page: {e}", exc_info=True)
        return "Error loading the application. Please check the logs."

@app.route('/query', methods=['POST'])
def query():
    """Handle user query submission"""
    try:
        user_text = request.form.get('text', '')
        selected_model = request.form.get('model', '')
        
        logger.debug(f"Received query request with text: '{user_text}', model: {selected_model}")
        
        # Handle image upload if present
        image_path = None
        if 'image' in request.files and request.files['image'].filename:
            image_file = request.files['image']
            # Save the uploaded image to a temporary location
            temp_dir = Path('temp_uploads')
            temp_dir.mkdir(exist_ok=True)
            image_path = temp_dir / image_file.filename
            image_file.save(image_path)
            image_path = str(image_path)
            logger.debug(f"Saved uploaded image to {image_path}")
        
        # Process the query
        result = process_query(user_text, image_path, selected_model)
        
        # Convert any image references to base64 for display
        images_base64 = []
        for img_path in result.get("image_references", []):
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
        
        # Clean up temporary uploaded image if it exists
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
                logger.debug(f"Removed temporary image: {image_path}")
            except Exception as e:
                logger.warning(f"Could not remove temporary image {image_path}: {e}")
        
        return jsonify({
            "text": result["text"],
            "images": images_base64
        })
    except Exception as e:
        logger.error(f"Error processing query request: {e}", exc_info=True)
        return jsonify({
            "text": f"Server error: {str(e)}",
            "images": []
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)