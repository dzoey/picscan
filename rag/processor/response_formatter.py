import re
import logging
from typing import Dict, List, Any

logger = logging.getLogger("rag_app")

class ResponseFormatter:
    """Handles formatting responses and generating HTML"""
    
    def __init__(self):
        """Initialize the response formatter"""
        pass
    
    def format_retrieved_documents(self, docs) -> str:
        """Format retrieved documents into a context string."""
        if not docs:
            return "No relevant information found."
            
        formatted_docs = []
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            content = doc.page_content
            
            # Format basic information
            doc_text = f"Document {i+1}:\n"
            
            if 'filename' in metadata:
                doc_text += f"Image: {metadata['filename']}\n"
                
            if content:
                doc_text += f"Description: {content}\n"
                
            # Add metadata that might be interesting
            exif_fields = {}
            other_fields = {}
            
            for key, value in metadata.items():
                if not value:
                    continue
                    
                if key == 'filename' or key == 'description':
                    continue
                    
                # Group EXIF fields separately
                if key.startswith('exif_'):
                    # Format the EXIF key in a more readable format
                    readable_key = key[5:].replace('_', ' ').title()
                    exif_fields[readable_key] = value
                else:
                    # Format other metadata in a more readable format
                    readable_key = key.replace('_', ' ').title()
                    other_fields[readable_key] = value
            
            # Add EXIF fields first (more important)
            if exif_fields:
                doc_text += "\nEXIF Metadata:\n"
                for key, value in exif_fields.items():
                    doc_text += f"- {key}: {value}\n"
                    
            # Add other metadata
            if other_fields:
                doc_text += "\nOther Metadata:\n"
                for key, value in other_fields.items():
                    doc_text += f"- {key}: {value}\n"
                    
            formatted_docs.append(doc_text)
            
        return "\n\n".join(formatted_docs)
    
    def process_response_with_images(self, response_text, doc_mapping):
        """Replace photo references with actual images in the response."""
        
        # Pattern to match "Photo X - filename.jpg" format
        pattern = r"Photo\s+(\d+)(?:\s+-\s+([^,\.\n\)]+))?(?:\.jpg|\.png|\.jpeg)?"
        
        # First find all matches to get a complete set
        matched_photos = set()
        for match in re.finditer(pattern, response_text):
            photo_num = int(match.group(1))
            if photo_num in doc_mapping:
                matched_photos.add(photo_num)
        
        # Add a section for any matched photos not explicitly mentioned in the text
        additional_images_html = ""
        for photo_num in doc_mapping.keys():
            if photo_num not in matched_photos:
                img_data = doc_mapping[photo_num]
                filename = img_data["filename"]
                path = img_data["path"]
                
                # Clean path handling
                if path.startswith('/'):
                    clean_path = path
                else:
                    clean_path = f"/{path}"
                    
                additional_images_html += f'''
                <div class="photo-container">
                    <img src="/image{clean_path}" alt="{filename}" class="photo-result" 
                        onerror="this.onerror=null; this.src='/image/{filename}'; this.setAttribute('data-fallback', 'true');" />
                    <div class="caption">Photo {photo_num} - {filename}</div>
                </div>'''
        
        # Then do the replacements
        def replace_with_image(match):
            photo_num = int(match.group(1))
            if photo_num in doc_mapping:
                img_data = doc_mapping[photo_num]
                filename = img_data["filename"]
                path = img_data["path"]
                
                # Create a URL that will work with our serve_image endpoint
                if path.startswith('/'):
                    clean_path = path
                else:
                    clean_path = f"/{path}"
                    
                # Generate HTML for the image display
                return f'''
                <div class="photo-container">
                    <img src="/image{clean_path}" alt="{filename}" class="photo-result" 
                        onerror="this.onerror=null; this.src='/image/{filename}'; this.setAttribute('data-fallback', 'true');" />
                    <div class="caption">Photo {photo_num} - {filename}</div>
                </div>'''
            return match.group(0)  # Keep original text if no match
        
        # Replace all matches in the text
        processed_text = re.sub(pattern, replace_with_image, response_text)
        
        # Add any additional matched images
        if additional_images_html:
            processed_text += "\n\n<h3>Additional Matched Images:</h3>\n" + additional_images_html
        
        # Add CSS styling and error handling script
        html_response = f"""
        <style>
            .photo-container {{
                margin: 10px 0;
                display: inline-block;
                max-width: 100%;
                text-align: center;
                margin-right: 15px;
            }}
            .photo-result {{
                max-width: 300px;
                max-height: 300px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                object-fit: contain;
            }}
            .caption {{
                margin-top: 5px;
                font-size: 14px;
                color: #555;
            }}
            .error-placeholder {{
                width: 300px;
                height: 200px;
                background-color: #f1f1f1;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 8px;
                font-size: 14px;
                color: #666;
            }}
        </style>
        <script>
            // Add event listener to handle image loading errors
            document.addEventListener('DOMContentLoaded', function() {{
                const images = document.querySelectorAll('.photo-result');
                images.forEach(img => {{
                    img.addEventListener('error', function() {{
                        if (this.getAttribute('data-fallback') === 'true') {{
                            // If even the fallback failed, replace with error placeholder
                            const errorDiv = document.createElement('div');
                            errorDiv.className = 'error-placeholder';
                            errorDiv.textContent = 'Image not available';
                            this.parentNode.replaceChild(errorDiv, this);
                        }}
                    }});
                }});
            }});
        </script>
        {processed_text}
        """
        
        return html_response
    
    def generate_simple_summary(self, docs, metadata_focus):
        """Generate a simple summary of documents when LLM processing fails."""
        if not docs:
            return "No pictures matching your query were found in the collection."
        
        num_docs = len(docs)
        summary_parts = [f"I found {num_docs} {'picture' if num_docs == 1 else 'pictures'} that might match your query."]
        
        # Extract key info based on focus
        if metadata_focus == "location":
            locations = []
            for doc in docs:
                metadata = doc.metadata
                location_parts = []
                
                if "exif_GPSInfo_city" in metadata and metadata["exif_GPSInfo_city"]:
                    location_parts.append(metadata["exif_GPSInfo_city"])
                
                if "exif_GPSInfo_state" in metadata and metadata["exif_GPSInfo_state"]:
                    location_parts.append(metadata["exif_GPSInfo_state"])
                
                if location_parts:
                    locations.append(", ".join(location_parts))
            
            if locations:
                unique_locations = list(set(locations))
                if len(unique_locations) == 1:
                    summary_parts.append(f"The picture was taken in {unique_locations[0]}.")
                else:
                    summary_parts.append(f"These pictures were taken in the following locations: {', '.join(unique_locations)}.")
            
        elif metadata_focus == "date":
            dates = []
            for doc in docs:
                metadata = doc.metadata
                if "exif_DateTimeOriginal" in metadata and metadata["exif_DateTimeOriginal"]:
                    dates.append(metadata["exif_DateTimeOriginal"])
            
            if dates:
                unique_dates = list(set(dates))
                if len(unique_dates) == 1:
                    summary_parts.append(f"The picture was taken on {unique_dates[0]}.")
                else:
                    summary_parts.append(f"These pictures were taken on the following dates: {', '.join(unique_dates[:3])}" + 
                                       (f" and {len(unique_dates) - 3} more." if len(unique_dates) > 3 else "."))
        
        elif metadata_focus == "camera":
            cameras = []
            for doc in docs:
                metadata = doc.metadata
                camera_parts = []
                
                if "exif_Make" in metadata and metadata["exif_Make"]:
                    camera_parts.append(metadata["exif_Make"])
                
                if "exif_Model" in metadata and metadata["exif_Model"]:
                    camera_parts.append(metadata["exif_Model"])
                
                if camera_parts:
                    cameras.append(" ".join(camera_parts))
            
            if cameras:
                unique_cameras = list(set(cameras))
                if len(unique_cameras) == 1:
                    summary_parts.append(f"The picture was taken with a {unique_cameras[0]}.")
                else:
                    summary_parts.append(f"These pictures were taken with the following cameras: {', '.join(unique_cameras)}.")
        
        # Add a description of the first few images
        descriptions = []
        for i, doc in enumerate(docs[:3]):
            if doc.page_content:
                descriptions.append(f"Image {i+1}: {doc.page_content}")
        
        if descriptions:
            summary_parts.append("Here are brief descriptions of some matching images:")
            summary_parts.extend(descriptions)
        
        return "\n\n".join(summary_parts)

