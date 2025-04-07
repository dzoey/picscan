"""Utilities for handling EXIF metadata queries and extraction"""

def detect_exif_query_type(query_text):
    """
    Detect if the query is about EXIF metadata and what type.
    Returns None if not an EXIF query, otherwise returns the category.
    """
    query_lower = query_text.lower()
    
    exif_categories = {
        "location": ["location", "where", "place", "city", "country", "state", "province", "taken", "shot in"],
        "camera": ["camera", "lens", "device", "equipment", "phone", "model", "make"],
        "settings": ["settings", "aperture", "iso", "f-stop", "shutter", "exposure"],
        "date": ["date", "when", "time", "year", "month", "day", "old"],
        "dimensions": ["resolution", "size", "megapixel", "dimensions", "pixels"]
    }
    
    # Check for general metadata queries
    if any(word in query_lower for word in ["exif", "metadata", "data", "information", "details"]):
        # Try to determine more specific category
        for category, keywords in exif_categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        return "general"  # General metadata query
    
    # Check for specific categories
    for category, keywords in exif_categories.items():
        if any(keyword in query_lower for keyword in keywords):
            return category
            
    return None  # Not an EXIF query

def extract_exif_metadata(docs, category):
    """Extract and group EXIF metadata from documents based on category"""
    exif_data = {}
    
    for doc in docs:
        metadata = doc.metadata
        
        # Extract all EXIF fields
        exif_fields = {k: v for k, v in metadata.items() if k.startswith('exif_')}
        
        # Process each EXIF field
        for field, value in exif_fields.items():
            # Skip the full JSON field to avoid duplication
            if field == 'exif_json':
                continue
                
            # Add to appropriate category
            clean_field = field.replace('exif_', '')
            
            if clean_field not in exif_data:
                exif_data[clean_field] = []
                
            if value and value not in exif_data[clean_field]:
                exif_data[clean_field].append(value)
    
    return exif_data

def format_exif_context(exif_data, category):
    """Format EXIF data into a context string based on category"""
    context = []
    
    # Handle location specific data
    if category == "location" or category == "general":
        location_fields = [
            "GPSInfo_city", "GPSInfo_state", "GPSInfo_country", 
            "GPSInfo_address", "GPSInfo_latitude", "GPSInfo_longitude"
        ]
        
        location_data = {}
        for field in location_fields:
            if field in exif_data and exif_data[field]:
                clean_name = field.replace("GPSInfo_", "")
                values = sorted(set(exif_data[field]))
                location_data[clean_name] = values
        
        if "state" in location_data:
            context.append(f"States in the photo collection: {', '.join(location_data['state'])}.")
            
        if "country" in location_data:
            context.append(f"Countries in the photo collection: {', '.join(location_data['country'])}.")
            
        if "city" in location_data:
            context.append(f"Cities in the photo collection: {', '.join(location_data['city'])}.")
    
    # Handle camera info
    if category == "camera" or category == "general":
        camera_fields = ["Make", "Model", "Software"]
        
        for field in camera_fields:
            if field in exif_data and exif_data[field]:
                values = sorted(set(exif_data[field]))
                context.append(f"Camera {field.lower()}: {', '.join(values)}.")
    
    # Handle date info
    if category == "date" or category == "general":
        date_fields = ["DateTimeOriginal", "DateTimeDigitized"]
        
        for field in date_fields:
            if field in exif_data and exif_data[field]:
                # Extract years and summarize
                years = set()
                for date_str in exif_data[field]:
                    if date_str and ":" in date_str:
                        year = date_str.split(":")[0]
                        if year.isdigit():
                            years.add(year)
                
                if years:
                    years_list = sorted(years)
                    context.append(f"Years photos were taken: {', '.join(years_list)}.")
                    context.append(f"Date range: {min(years_list)} to {max(years_list)}.")
    
    # Handle camera settings
    if category == "settings" or category == "general":
        settings_fields = ["ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength"]
        
        for field in settings_fields:
            if field in exif_data and exif_data[field]:
                values = sorted(set(exif_data[field]))[:5]  # Limit to 5 examples
                context.append(f"Camera {field}: Examples include {', '.join(str(v) for v in values)}.")
    
    # Add general fallback for any category
    for field, values in exif_data.items():
        # Skip fields we've already processed
        if field in ["GPSInfo_city", "GPSInfo_state", "GPSInfo_country", 
                    "GPSInfo_address", "GPSInfo_latitude", "GPSInfo_longitude", 
                    "Make", "Model", "Software", "DateTimeOriginal", "DateTimeDigitized",
                    "ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength"]:
            continue
            
        # Add any field with relatively few unique values (less than 5)
        unique_values = sorted(set(values))
        if 1 <= len(unique_values) <= 5:
            context.append(f"{field}: {', '.join(str(v) for v in unique_values)}.")
    
    if not context:
        return "No relevant EXIF metadata was found in the collection."
        
    return "\n".join(context)