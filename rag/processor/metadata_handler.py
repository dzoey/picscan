import re
import json
import logging
from typing import Dict, Any, List, Union
from rag.embeddings import EmbeddingManager
import config

logger = logging.getLogger("rag_app")

class MetadataHandler:
    """Handles EXIF metadata extraction, formatting, and filtering"""
    
    def __init__(self):
        """Initialize the metadata handler"""
        self.field_mapping = self._create_exif_field_mapping()
    
    def _create_exif_field_mapping(self):
        """Create a mapping from user-friendly terms to EXIF field names."""
        return {
            # Location fields
            "state": "exif_GPSInfo_state",
            "city": "exif_GPSInfo_city", 
            "country": "exif_GPSInfo_country",
            "address": "exif_GPSInfo_address",
            "location": ["exif_GPSInfo_state", "exif_GPSInfo_city", "exif_GPSInfo_country"],
            "latitude": "exif_GPSInfo_latitude",
            "longitude": "exif_GPSInfo_longitude",
            "gps": ["exif_GPSInfo_latitude", "exif_GPSInfo_longitude"],
            
            # Date/Time fields
            "date": ["exif_DateTimeOriginal", "exif_DateTimeDigitized"],
            "time": ["exif_DateTimeOriginal", "exif_DateTimeDigitized"],
            "when": ["exif_DateTimeOriginal", "exif_DateTimeDigitized"],
            "year": ["exif_DateTimeOriginal", "exif_DateTimeDigitized"],
            
            # Camera fields
            "camera": ["exif_Make", "exif_Model"],
            "make": "exif_Make",
            "model": "exif_Model",
            "brand": "exif_Make",
            
            # Technical fields
            "iso": "exif_ISOSpeedRatings",
            "aperture": "exif_FNumber",
            "f-stop": "exif_FNumber",
            "exposure": "exif_ExposureTime",
            "shutter": "exif_ExposureTime",
            "focal length": "exif_FocalLength",
            "flash": "exif_Flash",
            
            # Processing fields
            "software": "exif_Software",
            "edited with": "exif_Software",
            "processed": "exif_Software",
            
            # Image attributes
            "resolution": ["exif_XResolution", "exif_YResolution"],
            "width": "exif_ExifImageWidth",
            "height": "exif_ExifImageHeight",
            "orientation": "exif_Orientation",
            "white balance": "exif_WhiteBalance",
            "color space": "exif_ColorSpace"
        }
        
    def extract_metadata_query(self, query_text: str, llm=None):
        """
        Extract structured metadata query parameters using plain prompting 
        instead of function calling for compatibility with Ollama.
        """
        logger.debug(f"Extracting metadata from query: {query_text}")
        
        # Simple rule-based extraction for location if LLM is not available
        if llm is None:
            return self._rule_based_extraction(query_text)
        
        # Use a simple prompt template instead of function calling
        extraction_prompt = """Extract the information requested based on this query:
"{query}"

Please extract the following information as a JSON object:
- location: The location mentioned in the query (state, city, or country)
- date: Any date or time information mentioned
- camera: Any camera make or model information mentioned
- technical_details: Any technical settings like ISO, aperture, flash, etc.

If a field is not mentioned in the query, leave it as null.
Format your response as a valid JSON object with these fields.
ONLY RESPOND WITH THE JSON OBJECT, nothing else.""".replace("{query}", query_text)
        
        try:
            # Extract the information using the provided LLM
            result_str = llm.invoke(extraction_prompt)
            logger.debug(f"Raw extraction result: {result_str}")
            
            # Parse the JSON result
            try:
                extracted_info = json.loads(result_str)
                logger.debug(f"Parsed extracted metadata: {extracted_info}")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse extraction result as JSON: {e}")
                # Try to extract JSON from the text with regex as fallback
                json_pattern = r"\{[\s\S]*\}"
                match = re.search(json_pattern, result_str)
                if match:
                    try:
                        extracted_info = json.loads(match.group(0))
                        logger.debug(f"Extracted JSON using regex: {extracted_info}")
                    except json.JSONDecodeError:
                        logger.error("Failed to parse extracted JSON even with regex")
                        return self._rule_based_extraction(query_text)
                else:
                    logger.error("No JSON structure found in the extraction result")
                    return self._rule_based_extraction(query_text)
            
            # Convert extracted info to EXIF field format
            structured_query = {}
            for key, value in extracted_info.items():
                if not value:  # Skip empty or null values
                    continue
                    
                # Map to corresponding EXIF field(s)
                if key in self.field_mapping:
                    exif_fields = self.field_mapping[key]
                    if isinstance(exif_fields, list):
                        for field in exif_fields:
                            structured_query[field] = value
                    else:
                        structured_query[self.field_mapping[key]] = value
            
            # Special handling for state of Maryland query
            if "location" in extracted_info and extracted_info["location"]:
                location = extracted_info["location"].lower()
                if "maryland" in location or " md" in location or location.endswith("md"):
                    structured_query["exif_GPSInfo_state"] = "Maryland"
            
            return structured_query
            
        except Exception as e:
            logger.error(f"Error in metadata extraction: {e}")
            return self._rule_based_extraction(query_text)
    
    def _rule_based_extraction(self, query_text: str) -> Dict[str, Any]:
        """
        Simple rule-based extraction when LLM extraction fails or is unavailable.
        """
        query_lower = query_text.lower()
        structured_query = {}
        
        # Handle location patterns
        if "maryland" in query_lower or " md " in query_lower or query_lower.endswith(" md"):
            structured_query["exif_GPSInfo_state"] = "Maryland"
        elif "state of " in query_lower:
            # Try to extract state name
            match = re.search(r"state of ([a-zA-Z\s]+)(?:\?|\.|\s|$)", query_lower)
            if match:
                state_name = match.group(1).strip().title()
                structured_query["exif_GPSInfo_state"] = state_name
        
        # Handle date patterns
        date_match = re.search(r"in (\d{4})", query_lower)
        if date_match:
            year = date_match.group(1)
            structured_query["exif_DateTimeOriginal"] = year
        
        # Handle camera patterns
        camera_match = re.search(r"with (canon|nikon|sony|fuji|iphone|samsung)", query_lower)
        if camera_match:
            make = camera_match.group(1).title()
            structured_query["exif_Make"] = make
        
        logger.debug(f"Rule-based extraction result: {structured_query}")
        return structured_query
    
    def create_metadata_filters(self, structured_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a structured query to metadata filters for retrieval.
        """
        filters = {}
        if structured_query:
            # Convert to ChromaDB filter format
            for field, value in structured_query.items():
                if isinstance(value, list):
                    filters[field] = {"$in": value}
                else:
                    filters[field] = {"$eq": value}
        return filters
    
    def format_exif_for_prompt(self, doc):
        """Format EXIF metadata for inclusion in a prompt."""
        metadata = doc.metadata
        exif_info = ""
        
        for key, value in metadata.items():
            if key.startswith('exif_') and key not in ['exif_MakerNote', 'exif_json', 'exif_UserComment']:
                nice_key = key.replace('exif_', '').replace('_', ' ')
                exif_info += f"  - {nice_key}: {value}\n"
                
        return exif_info

