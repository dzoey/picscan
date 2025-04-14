import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger("rag_app")

# Field name normalization mapping
FIELD_MAPPING = {
    "flash": "exif_Flash",
    "iso": "exif_ISOSpeedRatings",
    "aperture": "exif_FNumber",
    "f-number": "exif_FNumber",
    "focal length": "exif_FocalLength",
    "exposure": "exif_ExposureTime",
    "camera": ["exif_Make", "exif_Model"],
    "model": "exif_Model",
    "make": "exif_Make",
    "software": "exif_Software",
    "date": ["exif_DateTimeOriginal", "exif_DateTimeDigitized"],
    "time": ["exif_DateTimeOriginal", "exif_DateTimeDigitized"],
    "state": "exif_GPSInfo_state",
    "city": "exif_GPSInfo_city",
    "country": "exif_GPSInfo_country",
    "address": "exif_GPSInfo_address",
    "resolution": ["exif_XResolution", "exif_YResolution"],
    "orientation": "exif_Orientation",
    "white balance": "exif_WhiteBalance",
    "people": "person_count"
}

# State/province variations dictionary
STATE_VARIATIONS = {
    # US States
    "maryland": ["Maryland", "MD"],
    "california": ["California", "CA"],
    # Add more states as needed
}

def get_metadata_filters(query_text):
    """Generate metadata filters based on query content for EXIF fields."""
    logger.debug(f"Analyzing query: {query_text}")
    
    filters = None
    query_lower = query_text.lower()
    
    # Check for people/faces query
    if _is_people_query(query_lower):
        filters = _create_people_filters(query_lower)
        if filters:
            return filters
    
    # Check for location query
    if _is_location_query(query_lower):
        filters = _create_location_filters(query_lower)
        if filters:
            return filters
    
    # Check for date query
    if _is_date_query(query_lower):
        filters = _create_date_filters(query_lower)
        if filters:
            return filters
    
    # Check for camera query
    if _is_camera_query(query_lower):
        filters = _create_camera_filters(query_lower)
        if filters:
            return filters
    
    # Check for technical details query
    if _is_tech_query(query_lower):
        filters = _create_tech_filters(query_lower)
        if filters:
            return filters
    
    # Check for specific field mentions
    field_filters = _create_field_specific_filters(query_lower)
    if field_filters:
        return field_filters
    
    logger.debug(f"Generated filters: {filters}")
    return filters

def _is_people_query(query_lower):
    """Check if query is about people or faces."""
    people_terms = ["people", "person", "face", "faces", "group", "crowd", "multiple people"]
    return any(term in query_lower for term in people_terms)

def _create_people_filters(query_lower):
    """Create filters for people queries."""
    if any(term in query_lower for term in ["multiple", "group", "several", "many"]):
        return {'person_count': {'$gt': 1}}  # More than 1 person
    else:
        return {'person_count': {'$gt': 0}}  # Any pictures with people

def _is_location_query(query_lower):
    """Check if query is about locations."""
    location_terms = ["state", "city", "country", "location", "where", "place", "taken in"]
    return any(term in query_lower for term in location_terms)

def _create_location_filters(query_lower):
    """Create filters for location queries."""
    # Extract location name
    location_pattern = r"(?:in|from|at)\s+(?:the\s+(?:state|city|country)\s+of\s+)?((?:[A-Z][a-zA-Z]+)(?:\s+[A-Z][a-zA-Z]+)*)"
    location_match = re.search(location_pattern, query_lower, re.IGNORECASE)
    
    if location_match:
        location_name = location_match.group(1).strip()
        
        # Check if it matches a known state/province
        for state_key, variations in STATE_VARIATIONS.items():
            if location_name.lower() == state_key or location_name.lower() in [v.lower() for v in variations]:
                return {'exif_GPSInfo_state': {'$in': variations}}
        
        # Generic location search
        return {
            '$or': [
                {'exif_GPSInfo_state': {'$eq': location_name}},
                {'exif_GPSInfo_city': {'$eq': location_name}},
                {'exif_GPSInfo_country': {'$eq': location_name}}
            ]
        }
    
    # Generic location query with any location data
    return {
        '$or': [
            {'exif_GPSInfo_state': {'$ne': ""}},
            {'exif_GPSInfo_city': {'$ne': ""}},
            {'exif_GPSInfo_country': {'$ne': ""}}
        ]
    }

# Additional helper methods for date, camera, and tech queries follow similar patterns
# I'll include simplified versions of these

def _is_date_query(query_lower):
    date_terms = ["date", "when", "time", "year", "month", "day", "taken on"]
    return any(term in query_lower for term in date_terms)

def _create_date_filters(query_lower):
    # Year pattern (2000-2025)
    year_pattern = r"\b(19\d{2}|20[0-2]\d)\b"
    year_match = re.search(year_pattern, query_lower)
    
    if year_match:
        year_str = year_match.group(1)
        # Match all month prefixes for this year
        year_prefixes = [f"{year_str}:{m:02d}" for m in range(1, 13)]
        
        return {
            '$or': [
                {'exif_DateTimeOriginal': {'$in': year_prefixes}},
                {'exif_DateTimeDigitized': {'$in': year_prefixes}}
            ]
        }
    
    # Generic date query - any date information
    return {
        '$or': [
            {'exif_DateTimeOriginal': {'$ne': ""}},
            {'exif_DateTimeDigitized': {'$ne': ""}}
        ]
    }

def _create_field_specific_filters(query_lower):
    """Create filters for specific field mentions."""
    field_pattern = r"\b(flash|iso|aperture|f-number|focal length|exposure|camera|model|make|software|date|time|state|city|country|address|resolution|orientation|white balance)\b"
    field_matches = re.findall(field_pattern, query_lower, re.IGNORECASE)
    
    if field_matches:
        field_filters = []
        for field in field_matches:
            field = field.lower()
            if field in FIELD_MAPPING:
                mapped_fields = FIELD_MAPPING[field]
                if isinstance(mapped_fields, list):
                    for mapped_field in mapped_fields:
                        field_filters.append({mapped_field: {'$ne': ""}})
                else:
                    field_filters.append({mapped_fields: {'$ne': ""}})
        
        if field_filters:
            return {'$or': field_filters}
    
    return None

