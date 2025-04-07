from typing import Dict, List, Optional
from utils.logging_config import logger

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

def is_rag_appropriate(query_text: str) -> bool:
    """
    Determine if using RAG is appropriate for this query
    
    Args:
        query_text: User's query text
        
    Returns:
        True if RAG should be used, False if direct query is better
    """
    # Keywords that indicate factual information needs
    factual_keywords = [
        'where', 'when', 'what', 'which', 'who', 'how many', 'list', 'show me',
        'search', 'find', 'locate', 'identify', 'look for', 'tell me about',
        'describe', 'explain', 'summarize', 'information about', 'details'
    ]
    
    # Keywords that suggest general chat or opinion-based questions
    chat_keywords = [
        'hey', 'hi', 'hello', 'think', 'believe', 'feel', 'opinion', 'view',
        'preference', 'suggest', 'recommend', 'consider', 'what if',
        'hypothetically', 'imagine', 'pretend', 'write', 'create', 'generate'
    ]
    
    query_lower = query_text.lower()
    
    # Check if query contains factual keywords
    has_factual_intent = any(keyword in query_lower for keyword in factual_keywords)
    
    # Check if query is primarily chat-oriented
    is_chat_intent = any(query_lower.startswith(keyword) for keyword in chat_keywords)
    
    # If it has factual elements and isn't just a casual chat, use RAG
    return has_factual_intent and not is_chat_intent


