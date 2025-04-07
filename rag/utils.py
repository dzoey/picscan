"""Utility functions for RAG application"""
import sys
import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

logger = logging.getLogger("rag_app")

def format_documents_batch(docs, batch_size=10):
    """Format documents in batches for better LLM digestion."""
    formatted_batches = []
    total_docs = len(docs)
    
    for i in range(0, total_docs, batch_size):
        end_idx = min(i + batch_size, total_docs)
        batch = docs[i:end_idx]
        
        formatted_batch = f"BATCH {i//batch_size + 1} OF {(total_docs + batch_size - 1) // batch_size}:\n\n"
        
        for j, doc in enumerate(batch):
            doc_idx = i + j + 1  # Global document index (1-based)
            
            # Extract key metadata
            metadata = doc.metadata
            path = metadata.get("path", "Unknown")
            filename = path.split("/")[-1] if "/" in path else path
            
            # Extract date information
            date = metadata.get("exif_DateTimeOriginal", metadata.get("exif_DateTimeDigitized", "Unknown date"))
            
            # Extract location information
            location_parts = []
            if "exif_GPSInfo_city" in metadata and metadata["exif_GPSInfo_city"]:
                location_parts.append(metadata["exif_GPSInfo_city"])
            if "exif_GPSInfo_state" in metadata and metadata["exif_GPSInfo_state"]:
                location_parts.append(metadata["exif_GPSInfo_state"])
            if "exif_GPSInfo_country" in metadata and metadata["exif_GPSInfo_country"]:
                location_parts.append(metadata["exif_GPSInfo_country"])
            
            location = ", ".join(location_parts) if location_parts else "Unknown location"
            
            # Format the document entry with clear numbering and key metadata
            formatted_batch += f"PHOTO {doc_idx} - {filename}\n"
            formatted_batch += f"DATE: {date}\n"
            formatted_batch += f"LOCATION: {location}\n"
            formatted_batch += f"DESCRIPTION: {doc.page_content.strip()}\n\n"
        
        formatted_batches.append(formatted_batch)
    
    return formatted_batches

def format_documents(doc):
    """Format a document with comprehensive metadata organization for better LLM understanding."""
    metadata = doc.metadata
    content = doc.page_content
    
    # Check if metadata has already been pruned, if not, prune it
    if any(field in metadata for field in ['exif_MakerNote', 'exif_UserComment', 'exif_json']):
        logger.warning("Document metadata contains bulky fields that should have been pruned")
        # Import the pruning function from retriever if not already pruned
        try:
            from .retriever import prune_metadata_for_context
            metadata = prune_metadata_for_context(metadata)
        except ImportError:
            # If we can't import, just create a simple filter here
            fields_to_remove = ['exif_MakerNote', 'exif_UserComment', 'exif_json']
            metadata = {k: v for k, v in metadata.items() if k not in fields_to_remove}
    
    # Start with the photo description (shorter version for context efficiency)
    # Truncate very long descriptions to save context
    max_description_length = 500
    if len(content) > max_description_length:
        short_content = content[:max_description_length] + "... [description truncated]"
        formatted = f"PHOTO DESCRIPTION:\n{short_content}\n\n"
    else:
        formatted = f"PHOTO DESCRIPTION:\n{content}\n\n"
    
    # Location information (if available)
    location_parts = []
    if "exif_GPSInfo_city" in metadata and metadata["exif_GPSInfo_city"]:
        location_parts.append(f"City: {metadata['exif_GPSInfo_city']}")
    if "exif_GPSInfo_state" in metadata and metadata["exif_GPSInfo_state"]:
        location_parts.append(f"State: {metadata['exif_GPSInfo_state']}")
    if "exif_GPSInfo_country" in metadata and metadata["exif_GPSInfo_country"]:
        location_parts.append(f"Country: {metadata['exif_GPSInfo_country']}")
    
    # Only include address if it's not too long
    if "exif_GPSInfo_address" in metadata and metadata["exif_GPSInfo_address"]:
        if len(metadata["exif_GPSInfo_address"]) < 100:  # Exclude very long addresses
            location_parts.append(f"Address: {metadata['exif_GPSInfo_address']}")
        else:
            # Extract just the essential parts of the address
            parts = metadata["exif_GPSInfo_address"].split(',')
            if len(parts) > 2:
                short_address = ', '.join(parts[-3:])  # Last 3 components
                location_parts.append(f"Address: {short_address}")
    
    if location_parts:
        formatted += "LOCATION INFORMATION:\n" + "\n".join(location_parts) + "\n\n"
    
    # Date and time information - Simplify to just one date field
    date_info = []
    if "exif_DateTimeOriginal" in metadata and metadata["exif_DateTimeOriginal"]:
        date_info.append(f"Date Taken: {metadata['exif_DateTimeOriginal']}")
    elif "exif_DateTimeDigitized" in metadata and metadata["exif_DateTimeDigitized"]:
        date_info.append(f"Date Digitized: {metadata['exif_DateTimeDigitized']}")
    
    if date_info:
        formatted += "DATE INFORMATION:\n" + "\n".join(date_info) + "\n\n"
    
    # Camera information
    camera_info = []
    if "exif_Make" in metadata and metadata["exif_Make"]:
        camera_info.append(f"Camera Make: {metadata['exif_Make']}")
    if "exif_Model" in metadata and metadata["exif_Model"]:
        camera_info.append(f"Camera Model: {metadata['exif_Model']}")
    
    # Only include software if not too verbose
    if "exif_Software" in metadata and metadata["exif_Software"]:
        if len(metadata["exif_Software"]) < 30:  # Skip overly detailed software strings
            camera_info.append(f"Software: {metadata['exif_Software']}")
    
    if camera_info:
        formatted += "CAMERA INFORMATION:\n" + "\n".join(camera_info) + "\n\n"
    
    # Photo technical details - Select only the most important ones
    tech_info = []
    technical_fields = {
        "exif_ISOSpeedRatings": "ISO",
        "exif_FNumber": "F-Number", 
        "exif_FocalLength": "Focal Length"
    }
    
    for field, label in technical_fields.items():
        if field in metadata and metadata[field]:
            tech_info.append(f"{label}: {metadata[field]}")
    
    if tech_info:
        formatted += "TECHNICAL DETAILS:\n" + "\n".join(tech_info) + "\n\n"
    
    # File information - Just the filename, not the full path
    if "path" in metadata and metadata["path"]:
        # Extract filename from path
        filename = metadata["path"].split("/")[-1]
        formatted += f"FILE INFORMATION:\nFilename: {filename}"
    
    # Check final size for debugging
    formatted_size = sys.getsizeof(formatted)
    if formatted_size > 10000:  # If over ~10KB, log a warning
        logger.warning(f"Large formatted document: {formatted_size} bytes")
    
    return formatted

def extract_metadata_highlights(metadata: Dict[str, Any]) -> str:
    """Extract important metadata highlights in a very concise format"""
    # Check if metadata has already been pruned, if not, prune it
    if any(field in metadata for field in ['exif_MakerNote', 'exif_UserComment', 'exif_json']):
        # Import the pruning function from retriever if not already pruned
        try:
            from .retriever import prune_metadata_for_context
            metadata = prune_metadata_for_context(metadata)
        except ImportError:
            # If we can't import, just create a simple filter here
            fields_to_remove = ['exif_MakerNote', 'exif_UserComment', 'exif_json']
            metadata = {k: v for k, v in metadata.items() if k not in fields_to_remove}
    
    highlights = []
    
    # Location data - Simplified to just city/state
    location_parts = []
    if "exif_GPSInfo_state" in metadata and metadata["exif_GPSInfo_state"]:
        location_parts.append(metadata["exif_GPSInfo_state"])
    elif "exif_GPSInfo_city" in metadata and metadata["exif_GPSInfo_city"]:
        location_parts.append(metadata["exif_GPSInfo_city"])
        
    if location_parts:
        highlights.append(f"Location: {', '.join(location_parts)}")
        
    # Date - Just year if possible
    if "exif_DateTimeOriginal" in metadata and metadata["exif_DateTimeOriginal"]:
        date_str = metadata["exif_DateTimeOriginal"]
        if ":" in date_str:
            year = date_str.split(":")[0]
            highlights.append(f"Year: {year}")
        else:
            highlights.append(f"Date: {date_str}")
        
    # Camera - Just make, not model (to save space)
    if "exif_Make" in metadata and metadata["exif_Make"]:
        highlights.append(f"Camera: {metadata['exif_Make']}")
        
    # Format the highlights
    if highlights:
        return "METADATA: " + "; ".join(highlights)
    return ""