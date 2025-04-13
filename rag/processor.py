import os
import logging
from typing import List, Dict, Optional, Tuple, Any, Union
import json
import re

# Add import for config
import config

# Update import for Ollama
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

from .retriever import get_langchain_retriever, process_documents_in_batches, determine_result_limit
from .embeddings import EmbeddingManager

from utils.exif_utils import extract_exif_metadata, format_exif_context
import logging

# Configure logger
logger = logging.getLogger("rag_app")

# Base prompt templates
RAG_SYSTEM_TEMPLATE = """You are a helpful assistant analyzing photos based on their metadata and descriptions.
    
Below is metadata from photos in a database:

{context}

QUESTION: {question}

When answering:
1. Consider only the information available in the provided metadata
2. If a photo's metadata includes information related to the question, describe that photo
3. If multiple photos match the query, summarize what they have in common
4. If no photos match the query criteria, clearly state that no matching photos were found
5. Be specific about locations, dates, camera equipment, or other details mentioned in the metadata
6. If the question does not reference images or photos or a person, answer the question without the metadata.

ANSWER:"""

EXIF_METADATA_TEMPLATE = """You are an expert at analyzing photos based on their EXIF metadata.

Below is EXIF metadata from photos in a database that match the query:

{context}

Additional document information:
{documents}

QUESTION: {question}

When answering:
1. Focus on the metadata fields specifically requested in the query
2. If the query asks about locations, pay special attention to state, city, and country fields
3. If the query asks about when photos were taken, focus on date information
4. If the query asks about camera equipment, highlight make and model information
5. Be specific about what metadata was found in the matching photos
6. If multiple photos match the criteria, summarize them by their common attributes
7. Remember that fields like "state" refer to "exif_GPSInfo_state" in the metadata

ANSWER:"""

DIRECT_SYSTEM_TEMPLATE = """You are an assistant that helps users search and understand their photo collection.
Answer questions about the photos based on your knowledge.
If you don't know the answer, say so.
"""

# Simple template for extraction without function calling
EXTRACTION_TEMPLATE = """Extract the information requested based on this query:
"{query}"

Please extract the following information as a JSON object:
- location: The location mentioned in the query (state, city, or country)
- date: Any date or time information mentioned
- camera: Any camera make or model information mentioned
- technical_details: Any technical settings like ISO, aperture, flash, etc.

If a field is not mentioned in the query, leave it as null.
Format your response as a valid JSON object with these fields.
ONLY RESPOND WITH THE JSON OBJECT, nothing else."""

class QueryProcessor:
    def __init__(self, embedding_manager: EmbeddingManager = None):
        """Initialize the query processor with embedding manager"""
        self.embedding_manager = embedding_manager or EmbeddingManager.get_instance()
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
    
    def _is_factual_query(self, query_text: str, llm=None) -> bool:
        """Determine if a query is factual using LLM with keyword fallback."""
        # First try using the LLM-based classifier if enabled
        if config.ENABLE_LLM_CLASSIFICATION:
            # If we don't have an LLM instance yet, initialize one specifically for classification
            if llm is None:
                try:
                    classification_llm = OllamaLLM(
                        model=config.SMALL_LLM_MODEL,  # Use small model for classification
                        temperature=config.CLASSIFICATION_TEMPERATURE,
                        repeat_penalty=config.CLASSIFICATION_REPEAT_PENALTY,
                        context_window=config.LLM_CONTEXT_SIZE,
                        num_predict=config.CLASSIFICATION_NUM_PREDICT
                    )
                except Exception as e:
                    logger.error(f"Error initializing classification LLM: {e}")
                    # Fall back to keyword matching if LLM initialization fails
                    return self._is_factual_query_keywords(query_text)
            else:
                classification_llm = llm
                
            classification = self._classify_query_with_llm(query_text, classification_llm)
            if classification is not None:
                # Return True if classification is "FACTUAL", False if "VISUAL"
                return classification == "FACTUAL"
        
        # Fall back to keyword matching if LLM classification fails or is disabled
        return self._is_factual_query_keywords(query_text)

    def _classify_query_with_llm(self, query_text, llm):
        """Use the provided LLM to classify the query as FACTUAL or VISUAL."""
        prompt = f"""Classify the following user query as either FACTUAL or VISUAL.

    DEFINITIONS:
    - FACTUAL: Questions seeking information, statistics, counts, or data extraction from the image collection without necessarily viewing the images themselves.
    - VISUAL: Requests to see, view, show, or display specific types of images. The user primarily wants to SEE the actual images.

    Key indicators of VISUAL queries:
    - Contains verbs like "show", "see", "view", "display", "find", "get"
    - Contains nouns like "pictures", "photos", "images" 
    - The primary intent is to view the images themselves
    - User is asking what something "looks like"
    - Any request about showing/displaying pictures is VISUAL by default

    Examples of VISUAL queries (all of these are VISUAL):
    - "Show me scenic pictures"
    - "Find pictures with vehicles"
    - "Get photos from California"
    - "What pictures have people in them?"
    - "I want to see images of mountains"
    - "Show pictures taken last year"
    - "Find photos with multiple people and trees"

    Examples of FACTUAL queries (these are seeking information, not primarily images):
    - "How many pictures were taken in 2022?"
    - "What camera was used for most photos?"
    - "When was my trip to Paris?"
    - "Count how many photos have people in them"
    - "Which location appears most in my photos?"

    USER QUERY: {query_text}

    CLASSIFICATION RULE:
    If the query is asking to SEE, SHOW, VIEW, FIND or GET PICTURES/PHOTOS/IMAGES, it is VISUAL.
    When in doubt, prefer VISUAL for any query about pictures.

    Your classification (respond with only FACTUAL or VISUAL):"""

        try:
            response = llm.invoke(prompt)
            result = response.strip().upper()
            
            # Make the classification more reliable by accepting partial matches
            if "VISUAL" in result:
                logger.debug(f"LLM classified query as VISUAL: {query_text}")
                return "VISUAL"
            elif "FACTUAL" in result:
                logger.debug(f"LLM classified query as FACTUAL: {query_text}")
                return "FACTUAL"
            else:
                # Default to VISUAL for ambiguous responses
                logger.debug(f"LLM gave ambiguous classification for query, defaulting to VISUAL: {query_text}")
                return "VISUAL"
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            # Default to VISUAL on error to ensure images are displayed
            return "VISUAL"

    def _check_document_relevance(self, query: str, docs: List[Any], threshold: float = 0.4) -> bool:
        """Check if any documents are semantically relevant to the query."""
        try:
            # Get embeddings from the embedding manager
            if not self.embedding_manager or not hasattr(self.embedding_manager, 'embeddings'):
                logger.warning("No embedding manager available for relevance check")
                return True  # Default to assuming relevance
                
            # Get query embedding
            query_embedding = self.embedding_manager.embeddings.embed_query(query)
            
            # Get document embeddings
            relevance_scores = []
            for doc in docs:
                doc_text = doc.page_content
                doc_embedding = self.embedding_manager.embeddings.embed_query(doc_text)
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                relevance_scores.append(similarity)
                
                logger.debug(f"Document relevance score: {similarity:.4f} for path: {doc.metadata.get('path', 'no_path')}")
                
            # Check if any document exceeds the threshold
            max_relevance = max(relevance_scores) if relevance_scores else 0
            logger.debug(f"Maximum document relevance: {max_relevance:.4f} (threshold: {threshold})")
            
            return max_relevance >= threshold
            
        except Exception as e:
            logger.error(f"Error checking document relevance: {str(e)}")
            return True  # Default to showing docs if there's an error
            
    def _cosine_similarity(self, v1, v2):
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        
        v1 = np.array(v1)
        v2 = np.array(v2)
        
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0
            
        return dot_product / (norm_v1 * norm_v2)

    def _get_direct_answer(self, question: str, llm) -> str:
        """Get a direct answer from the LLM without using RAG context."""
        try:
            if llm is None:
                return f"I don't have specific information about {question} in my photo collection."
                
            prompt = f"""You are a helpful assistant answering questions based on your knowledge.
            
    Question: {question}

    Please provide a concise and accurate answer based on factual information:"""

            return llm.invoke(prompt)
        except Exception as e:
            logger.error(f"Error getting direct answer: {str(e)}")
            return f"I'm not able to provide information about {question} at this time."
        
    def _initialize_llm(self, model):
        """Initialize an LLM instance with the specified model."""
        # Check if the model is a text generation model
        if not self.is_text_generation_model(model):
            logger.warning(f"Model {model} appears to be an embedding model, not suitable for text generation")
            return None
            
        # Initialize the model with timeout
        try:
            llm = OllamaLLM(
                model=model,
                context_window=config.LLM_CONTEXT_SIZE,
                temperature=config.DEFAULT_TEMPERATURE,
                repeat_penalty=config.DEFAULT_REPEAT_PENALTY,
                num_predict=config.DEFAULT_NUM_PREDICT,
                timeout=config.DEFAULT_TIMEOUT
            )
            return llm
        except Exception as e:
            logger.error(f"Error initializing Ollama LLM with model {model}: {e}")
            # Try fallback to a default model from the config
            try:
                llm = OllamaLLM(
                    model=config.FALLBACK_MODELS[0],
                    context_window=config.LLM_CONTEXT_SIZE,
                    temperature=config.DEFAULT_TEMPERATURE,
                    repeat_penalty=config.DEFAULT_REPEAT_PENALTY,
                    num_predict=config.DEFAULT_NUM_PREDICT,
                    timeout=config.DEFAULT_TIMEOUT
                )
                return llm
            except Exception as e2:
                logger.error(f"Error initializing fallback model: {e2}")
                return None
            
    def is_text_generation_model(self, model_name):
        """Check if the model is capable of text generation."""
        # List of known embedding-only models
        embedding_models = ["all-minilm", "all-mpnet", "nomic-embed", "bge-"]
        
        # Check if the model name contains any embedding model identifiers
        for embedding_model in embedding_models:
            if embedding_model in model_name.lower():
                return False
                
        return True
    
        
    def extract_metadata_query(self, query_text: str, llm):
        """
        Extract structured metadata query parameters using plain prompting 
        instead of function calling for compatibility with Ollama.
        """
        logger.debug(f"Extracting metadata from query: {query_text}")
        
        # Simple rule-based extraction for location if LLM is not available
        if llm is None:
            return self._rule_based_extraction(query_text)
        
        # Use a simple prompt template instead of function calling
        extraction_prompt = PromptTemplate.from_template(EXTRACTION_TEMPLATE)
        
        # Create a simple extraction chain
        extraction_chain = (
            {"query": lambda x: x}
            | extraction_prompt
            | llm
            | StrOutputParser()
        )
        
        try:
            # Extract the information
            result_str = extraction_chain.invoke(query_text)
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
    
    def retrieve_with_filters(self, query_text, structured_query):
        """
        Retrieve documents with metadata filters based on the structured query.
        """
        logger.debug(f"Retrieving with filters: {structured_query}")
        
        # First, ensure we have a valid vectorstore
        if self.embedding_manager is None:
            logger.error("Embedding manager is None")
            # Try to recreate it
            self.embedding_manager = EmbeddingManager.get_instance()
        
        # Get the vectorstore
        vectorstore = self.embedding_manager.text_vectorstore
        
        # Check if vectorstore is initialized
        if vectorstore is None:
            logger.error("Vectorstore is None, falling back to standard retriever")
            # Try fallback to standard retriever
            try:
                retriever = get_langchain_retriever(query_text)
                return retriever.get_relevant_documents(query_text)
            except Exception as e:
                logger.error(f"Failed to use standard retriever: {e}")
                return []
        
        # Create filters
        filters = {}
        if structured_query:
            # Convert to ChromaDB filter format
            for field, value in structured_query.items():
                if isinstance(value, list):
                    filters[field] = {"$in": value}
                else:
                    filters[field] = {"$eq": value}
        
        # If we have filters, use them
        if filters:
            logger.debug(f"Using filters: {filters}")
            try:
                retriever = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5, "filter": filters}
                )
                docs = retriever.get_relevant_documents(query_text)
                
                # If no results with filter, try without
                if not docs:
                    logger.debug("No results with filters, trying without")
                    retriever = vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 5}
                    )
                    docs = retriever.get_relevant_documents(query_text)
                return docs
            except Exception as e:
                logger.error(f"Error retrieving with filters: {e}")
                # Try without filters
                try:
                    retriever = vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 5}
                    )
                    return retriever.get_relevant_documents(query_text)
                except Exception as e2:
                    logger.error(f"Error retrieving without filters: {e2}")
                    return []
        else:
            # No filters, just do a regular search
            try:
                retriever = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                )
                return retriever.get_relevant_documents(query_text)
            except Exception as e:
                logger.error(f"Error in basic retrieval: {e}")
                return []

    def _format_retrieved_documents(self, docs) -> str:
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
    
    def _process_response_with_images(self, response_text, doc_mapping):
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

    def detect_metadata_focus(self, query_text: str) -> Optional[str]:
        """Detect if the query is focusing on specific metadata."""
        metadata_keywords = {
            "date": ["date", "when", "time", "day", "month", "year"],
            "location": ["location", "where", "place", "city", "country", "state"],
            "people": ["person", "people", "who", "faces", "man", "woman", "child"],
            "objects": ["object", "thing", "contain", "what is in", "what's in"],
            "colors": ["color", "colours", "red", "blue", "green", "yellow"],
            "camera": ["camera", "lens", "shot", "iso", "aperture", "shutter", "focal", "make", "model"]
        }
        
        query_lower = query_text.lower()
        for focus, keywords in metadata_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return focus
                
        return None

    def should_use_rag(self, query_text: str) -> bool:
        """Determine if RAG should be used for this query."""
        # Always use RAG for specific queries about content
        content_indicators = ["show", "find", "which", "where", "when", "what", "who", "how many"]
        query_lower = query_text.lower()
        
        for indicator in content_indicators:
            if query_lower.startswith(indicator) or f" {indicator} " in query_lower:
                return True
                
        # If query is very short, likely a command/general question
        if len(query_text.split()) < 3:
            return False
            
        # Default to using RAG
        return True
    
    def process_exif_metadata_query(self, query_text, llm=None, vectorstore=None, is_factual=False):
        """Process a query specifically about EXIF metadata."""
        logger.debug(f"Processing EXIF metadata query: {query_text}")
        
        # Extract locations or other EXIF parameters from the query
        exif_params = self.extract_metadata_query(query_text, llm)
        logger.debug(f"Extracted EXIF parameters: {exif_params}")
        
        # Construct metadata filters
        filters = {}
        if exif_params.get('location'):
            location = exif_params['location']
            if isinstance(location, dict):
                for key, value in location.items():
                    if value:
                        field_name = f"exif_GPSInfo_{key}"
                        filters[field_name] = {"$eq": value}
            elif isinstance(location, str):
                # Create an OR filter for various location fields
                filters = {
                    "$or": [
                        {"exif_GPSInfo_city": {"$eq": location}},
                        {"exif_GPSInfo_state": {"$eq": location}},
                        {"exif_GPSInfo_country": {"$eq": location}}
                    ]
                }
        
        # Retrieve documents with the metadata filters
        logger.debug(f"Retrieving with filters: {filters}")
        
        # Set a high score threshold for factual queries
        score_threshold = 0.5 if is_factual else 0.25
        
        # Use the retriever with the appropriate threshold
        retriever = get_langchain_retriever(query_text, top_k=5, metadata_filters=filters, score_threshold=score_threshold)
        docs = retriever.get_relevant_documents(query_text)
        logger.debug(f"Retrieved {len(docs)} documents")
        
        # For factual queries with no relevant documents, return direct answer with no images
        if is_factual and (not docs or len(docs) == 0):
            direct_answer = self._get_direct_answer(query_text, llm)
            return {
                "answer": direct_answer,
                "text": direct_answer,
                "documents": [],
                "query_type": "direct_answer",
                "metadata_focus": self.detect_metadata_focus(query_text),
                "suppress_images": True
            }
        
        # Process with LLM
        if llm:
            # Format documents for context
            formatted_docs = []
            for i, doc in enumerate(docs):
                metadata = doc.metadata
                path = metadata.get('path', 'Unknown')
                
                # Format key EXIF data for the prompt
                exif_info = ""
                for key, value in metadata.items():
                    if key.startswith('exif_') and not key in ['exif_MakerNote', 'exif_json', 'exif_UserComment']:
                        nice_key = key.replace('exif_', '').replace('_', ' ')
                        exif_info += f"  - {nice_key}: {value}\n"
                
                # Add general description
                doc_content = f"Document {i+1}:\n"
                doc_content += f"Description: {doc.page_content}\n\n"
                doc_content += f"EXIF Metadata:\n{exif_info}\n"
                
                formatted_docs.append(doc_content)
            
            # Create prompt
            prompt = f"""You are a helpful assistant answering questions about EXIF metadata in photos.
I have a collection of photos with their descriptions and EXIF metadata.

USER QUESTION: {query_text}

PHOTO INFORMATION:
{'-'*80}
{"".join(formatted_docs)}
{'-'*80}

Based on the EXIF metadata above, please provide a concise answer to the user's question.
Focus on metadata like locations, dates, camera information, and technical details.
If the information is not available in the metadata, please state that clearly."""

            try:
                # Get response from LLM
                response = llm.invoke(prompt)
                
                # For factual queries, check if there's any real EXIF info in the response
                # If the response is just coming from the LLM's knowledge, suppress images
                suppress_images = False
                if is_factual:
                    # Check if the response mentions any EXIF data or is just general knowledge
                    exif_references = ["exif", "metadata", "gps", "photo", "image", "camera", "location"]
                    contains_exif_info = any(term in response.lower() for term in exif_references)
                    
                    # If it's factual and doesn't reference EXIF data, suppress images
                    suppress_images = is_factual and not contains_exif_info
                
                return {
                    "answer": response,
                    "text": response,
                    "documents": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs],
                    "query_type": "exif_metadata",
                    "metadata_focus": self.detect_metadata_focus(query_text),
                    "structured_query": exif_params,
                    "suppress_images": suppress_images
                }
            except Exception as e:
                logger.error(f"Error getting LLM response: {e}")
                # Fall back to basic summary
        
        # Fallback if LLM fails or is not available
        summary = "Here is the EXIF metadata from relevant photos:\n\n"
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            path = metadata.get('path', 'Unknown')
            filename = path.split('/')[-1] if '/' in path else path
            
            summary += f"Photo {i+1}: {filename}\n"
            
            # Add location if available
            location_parts = []
            for field in ['city', 'state', 'country']:
                if f'exif_GPSInfo_{field}' in metadata and metadata[f'exif_GPSInfo_{field}']:
                    location_parts.append(metadata[f'exif_GPSInfo_{field}'])
            
            if location_parts:
                summary += f"Location: {', '.join(location_parts)}\n"
                
            # Add date if available
            if 'exif_DateTimeOriginal' in metadata:
                summary += f"Date: {metadata['exif_DateTimeOriginal']}\n"
                
            # Add camera info if available
            if 'exif_Make' in metadata or 'exif_Model' in metadata:
                camera = []
                if 'exif_Make' in metadata and metadata['exif_Make']:
                    camera.append(metadata['exif_Make'])
                if 'exif_Model' in metadata and metadata['exif_Model']:
                    camera.append(metadata['exif_Model'])
                if camera:
                    summary += f"Camera: {' '.join(camera)}\n"
            
            summary += "\n"
        
        return {
            "answer": summary,
            "text": summary,
            "documents": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs],
            "query_type": "exif_metadata",
            "metadata_focus": self.detect_metadata_focus(query_text),
            "structured_query": exif_params,
            "suppress_images": is_factual  # Always suppress images for factual queries in fallback mode
        }
    
    def _generate_simple_summary(self, docs, metadata_focus):
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
    
    
    def process_query(self, 
            user_text: str, 
            image_path: Optional[str] = None,
            model: str = config.DEFAULT_LLM_MODEL) -> Dict[str, Any]:
        """Process a user query using RAG techniques."""
        try:
            logger.debug(f"Processing query with text: '{user_text}', image: {image_path}, model: {model}")

             # Initialize LLM first since we need it for both factual and visual queries
            llm = self._initialize_llm(model)
            
            # Check if this is a factual query that shouldn't display images
            is_factual_query = self._is_factual_query(user_text, llm)
            logger.debug(f"Query classified as factual: {is_factual_query}")
            
            
            # HANDLE FACTUAL QUERIES IMMEDIATELY - Exit early before retrieval
            if is_factual_query:
                # If this is a factual query, provide direct answer without images
                direct_answer = self._get_direct_answer(user_text, llm)
                logger.debug("Providing direct answer for factual query (no document retrieval)")
                return {
                    "answer": direct_answer,
                    "text": direct_answer,
                    "documents": [],  # Don't include documents for factual queries
                    "query_type": "direct_answer",
                    "suppress_images": True
                }
                
            # Check if the model is a text generation model
            if not self.is_text_generation_model(model):
                logger.warning(f"Model {model} appears to be an embedding model, not suitable for text generation")
                llm = None
        
                
            # Check if this is a query about multiple people in photos
            is_multiple_people_query = any(term in user_text.lower() for term in 
                                        ["multiple people", "more than one person", "several people", 
                                        "group of people", "group photo"])

            # Detect if this is an EXIF metadata query
            metadata_focus = self.detect_metadata_focus(user_text)
            
            # Only treat as EXIF query if it's NOT a factual query about history
            exif_keywords = ["exif", "metadata", "state", "location", "where", "when", "date", 
                            "time", "camera", "make", "model", "iso", "flash", "aperture"]
            
            historical_keywords = ["history", "historical", "ancient", "past", "century", 
                                "war", "king", "queen", "president", "leader", "empire", 
                                "nation", "country", "government", "founding", "constitution"]
            
            is_exif_query = any(term in user_text.lower() for term in exif_keywords)
            is_historical_query = any(term in user_text.lower() for term in historical_keywords)
            
            # Don't treat historical factual queries as EXIF queries even if they contain EXIF keywords
            if is_exif_query and is_factual_query and is_historical_query:
                is_exif_query = False
                logger.debug(f"Detected historical factual query, not treating as EXIF query")
            
            # If it's an EXIF metadata query, use specialized processing
            if is_exif_query:
                logger.debug(f"Using specialized EXIF metadata processing for query with focus: {metadata_focus}")
                # Pass factual flag to EXIF query processing
                result = self.process_exif_metadata_query(
                    user_text, 
                    llm, 
                    self.embedding_manager.text_vectorstore if self.embedding_manager else None,
                    is_factual=is_factual_query
                )
                # For factual queries, always suppress images
                if is_factual_query:
                    result["suppress_images"] = True
                return result
                    
            # For other queries, determine if we should use RAG
            use_rag = self.should_use_rag(user_text)
            logger.debug(f"Using RAG for this query: {use_rag}")
            
            if use_rag:
                # Get a dynamic document limit based on query type
                result_limit = determine_result_limit(user_text)
                logger.debug(f"Using retriever with limit of {result_limit} documents")
                
                # Set higher score threshold for factual queries
                score_threshold = 0.8 if is_factual_query else 0.3
                logger.debug(f"Using similarity score threshold: {score_threshold}")
                
                # Retrieve relevant documents using the dynamic limit and threshold
                retriever = get_langchain_retriever(
                    user_text, 
                    top_k=result_limit, 
                    score_threshold=score_threshold
                )
                
                # Use the updated retriever which handles score filtering internally
                docs = retriever.get_relevant_documents(user_text)
                logger.debug(f"Retrieved {len(docs)} documents")
                
                # For factual queries, we'll suppress images regardless
                if is_factual_query:
                    # If this is a factual query, provide direct answer without images
                    direct_answer = self._get_direct_answer(user_text, llm)
                    return {
                        "answer": direct_answer,
                        "text": direct_answer,
                        "documents": [],  # Don't include documents for factual queries
                        "query_type": "direct_answer",
                        "suppress_images": True
                    }
                
                if not docs:
                    logger.warning(f"No documents retrieved for query: {user_text}")
                    return {
                        "answer": "I couldn't find any photos matching your query in the collection.",
                        "text": "I couldn't find any photos matching your query in the collection.",
                        "documents": [],
                        "query_type": "rag",
                        "suppress_images": True
                    }
                
                # For multiple people queries or any query that needs to analyze many documents
                if is_multiple_people_query or len(docs) > 10:
                    logger.debug(f"Using batched document processing for query: {user_text}")
                    
                    # Determine an appropriate batch size based on the total number of docs
                    batch_size = 5 if len(docs) <= 20 else 10
                    logger.debug(f"Using batch size of {batch_size} for {len(docs)} documents")
                    
                    try:
                        # Use batched processing with the dynamic batch size
                        batch_result = process_documents_in_batches(docs, user_text, model, batch_size=batch_size, llm=llm)
                        
                        # Check if we got a dictionary with response and doc_mapping
                        if isinstance(batch_result, dict) and "response" in batch_result:
                            answer_text = batch_result["response"]
                            doc_mapping = batch_result.get("doc_mapping", {})
                            matched_images = batch_result.get("matched_images", [])
                            
                            # Log the number of matched images for debugging
                            logger.debug(f"Found {len(matched_images)} matched images: {matched_images}")
                            logger.debug(f"doc_mapping contains {len(doc_mapping)} entries")
                            
                            # Process the response to replace photo references with actual images
                            processed_answer = self._process_response_with_images(answer_text, doc_mapping)
                            
                            # Return structured response
                            return {
                                "answer": processed_answer,
                                "text": answer_text,  # Original text without images
                                "documents": [{"page_content": doc.page_content, "metadata": doc.metadata} 
                                            for doc in docs if doc.metadata.get("path") in 
                                            [v.get("path") for v in doc_mapping.values()]],
                                "query_type": "rag_batched",
                                "metadata_focus": self.detect_metadata_focus(user_text),
                                "is_html": True,  # Flag indicating this is HTML content
                                "matched_images": matched_images,
                                "suppress_images": is_factual_query
                            }
                        else:
                            # Handle case where batch_result is just a string
                            answer = str(batch_result)
                            # Fall through to standard RAG
                    except Exception as batch_error:
                        logger.error(f"Error in batched processing: {batch_error}, falling back to standard RAG")
                        
                # Format retrieved documents for standard RAG processing
                context = self._format_retrieved_documents(docs)

                # If we don't have a valid LLM for generation, create a simple summary
                if llm is None:
                    summary = self._generate_simple_summary(docs, metadata_focus or "general")
                    return {
                        "answer": summary,
                        "text": summary,
                        "documents": [
                            {
                                "page_content": doc.page_content,
                                "metadata": doc.metadata
                            } for doc in docs
                        ],
                        "query_type": "rag_fallback",
                        "metadata_focus": metadata_focus,
                        "suppress_images": is_factual_query
                    }

                # Create RAG prompt
                rag_prompt = PromptTemplate.from_template(
                    RAG_SYSTEM_TEMPLATE + "\nQuestion: {question}"
                )

                # Set up RAG chain
                rag_chain = (
                    {"context": lambda x: context, "question": RunnablePassthrough()}
                    | rag_prompt
                    | llm
                    | StrOutputParser()
                )

                # Execute RAG chain
                try:
                    answer = rag_chain.invoke(user_text)
                except Exception as e:
                    logger.error(f"Error invoking RAG chain: {e}")
                    # Provide a fallback answer when the chain fails
                    answer = f"I found some photos that might be relevant to your query about {user_text}, but I encountered an error processing the details. The error was: {e}"

                # Return structured response
                return {
                    "answer": answer,
                    "text": answer,
                    "documents": [
                        {
                            "page_content": doc.page_content,
                            "metadata": doc.metadata
                        } for doc in docs
                    ],
                    "query_type": "rag_standard",
                    "metadata_focus": metadata_focus,
                    "suppress_images": is_factual_query
                }
                    
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                "answer": f"I encountered an error while processing your query: {str(e)}",
                "text": f"I encountered an error while processing your query: {str(e)}",  # Add text key
                "error": str(e),
                "documents": [],
                "query_type": "error",
                "suppress_images": True
            }

def get_available_models():
    """
    Return a list of available Ollama models for inference.
    
    Returns:
        List of model names
    """
    try:
        import requests
        
        # Try to get models from Ollama API
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            # Extract model names
            model_names = [model["name"] for model in models]
            return model_names
        else:
            logger.error(f"Failed to get models: {response.status_code}")
            # Return fallback models from config
            return config.FALLBACK_MODELS
            
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        # Return fallback models from config
        return config.FALLBACK_MODELS

# Optional singleton instance
_processor_instance = None

def get_processor():
    """Get or create the processor singleton instance"""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = QueryProcessor()
    return _processor_instance

# Add a standalone wrapper function for backward compatibility
def process_query(user_text, image_path=None, model=config.DEFAULT_LLM_MODEL):
    """
    Process a query with the QueryProcessor.
    This function exists for backward compatibility.
    """
    processor = get_processor()
    return processor.process_query(user_text=user_text, image_path=image_path, model=model)