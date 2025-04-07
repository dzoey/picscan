# Add these imports to your existing imports
import logging
import re
import sys
from typing import Optional, Dict, Any, Tuple, List
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from .embeddings import EmbeddingManager

# Get your logger
logger = logging.getLogger("rag_app")

# Import from your own modules
from utils.exif_utils import detect_exif_query_type

def prune_metadata_for_context(metadata):
    """Remove bulky fields not needed for answering queries"""
    fields_to_remove = [
        'exif_MakerNote', 'exif_UserComment', 'exif_json'
    ]
    
    # Actually remove the fields
    pruned = {k: v for k, v in metadata.items() if k not in fields_to_remove}
    
    # Calculate size reduction for logging
    original_size = sum(sys.getsizeof(str(v)) for v in metadata.values())
    pruned_size = sum(sys.getsizeof(str(v)) for v in pruned.values())
    reduction = original_size - pruned_size
    
    logger.debug(f"Metadata pruning: removed {len(fields_to_remove)} fields, " 
                 f"reduced size by {reduction} bytes ({original_size} → {pruned_size})")
    
    return pruned

def get_langchain_retriever(query_text, top_k=None, metadata_filters=None, score_threshold=None):
    """
    Enhanced retriever with similarity score threshold filtering and metadata support.
    
    Args:
        query_text: The user's query
        top_k: Maximum number of documents to retrieve (determined dynamically if None)
        metadata_filters: Optional metadata filters for ChromaDB
        score_threshold: Minimum similarity score (0-1) to include documents
    
    Returns:
        A retriever that prunes metadata and filters by similarity score
    """
    from .embeddings import EmbeddingManager
    
    # Set score threshold based on query type if not provided
    if score_threshold is None:
        # Use higher threshold for factual queries to avoid irrelevant results
        if any(keyword in query_text.lower().split() for keyword in 
              ["who", "what", "when", "where", "why", "how", "which", "king", "queen", "president"]):
            score_threshold = 0.6  # Use a much higher threshold (0.6 instead of 0.35)
        else:
            score_threshold = 0.3  # Also increase the default threshold
            
    logger.debug(f"Using similarity score threshold: {score_threshold}")
    
    # Get an instance of the embedding manager
    embedding_manager = EmbeddingManager.get_instance()
    
    # Get the text vectorstore
    vectorstore = embedding_manager.text_vectorstore
    
    logger.debug(f"Vectorstore initialized: {vectorstore is not None}")
    
    if vectorstore is None:
        logger.error("Vectorstore is None - check if your database is properly initialized")
        from langchain_core.retrievers import BaseRetriever
        
        class FallbackRetriever(BaseRetriever):
            def _get_relevant_documents(self, query, **kwargs):
                return []
                
        return FallbackRetriever()
    
    # Dynamically adjust top_k based on query type
    if top_k is None:
        top_k = determine_result_limit(query_text)
        logger.debug(f"Dynamically set top_k to {top_k} based on query")
    
    # Set up search parameters
    search_kwargs = {
        "k": top_k
    }
    
    # Add metadata filters if provided
    if metadata_filters and hasattr(vectorstore, "as_retriever"):
        search_kwargs["filter"] = metadata_filters
        logger.debug(f"Using metadata filters: {metadata_filters}")
    
    # Create a custom retriever that wraps the vectorstore retriever and prunes metadata
    from langchain_core.retrievers import BaseRetriever
    
    # Configure the base retriever - don't use similarity_score_threshold directly
    # We'll handle filtering by score ourselves
    vectorstore_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    class ScoreFilteringRetriever(BaseRetriever):
        def _get_relevant_documents(self, query, **kwargs):
            # Get documents with scores but don't pass return_scores
            try:
                # Use similarity_search_with_score method directly on the vectorstore
                docs_with_scores = vectorstore.similarity_search_with_score(
                    query, k=top_k, filter=metadata_filters
                )
                
                # Log the similarity scores for analysis
                if docs_with_scores:
                    for i, (doc, score) in enumerate(docs_with_scores):
                        logger.debug(f"Document {i+1} score: {score:.4f} - Path: {doc.metadata.get('path', 'no_path')}")
                    
                    # Extract just the documents that meet the threshold
                    docs = [doc for doc, score in docs_with_scores if score >= score_threshold]
                else:
                    docs = []
                    
                logger.debug(f"Retrieved {len(docs)} documents after score filtering")
                
                # No docs found, and this isn't a metadata-filtered search? Try a broader search
                if not docs and not metadata_filters and kwargs.get('fallback_allowed', True):
                    logger.debug(f"No documents met the threshold {score_threshold}, trying with lower threshold")
                    # Try again with a lower threshold
                    lower_threshold = max(0.1, score_threshold - 0.2)
                    kwargs['fallback_allowed'] = False  # Prevent infinite recursion
                    
                    # Get documents with lower threshold
                    fallback_docs_with_scores = vectorstore.similarity_search_with_score(
                        query, k=top_k, filter=metadata_filters
                    )
                    
                    # Log the fallback scores
                    logger.debug(f"Using fallback threshold: {lower_threshold}")
                    for i, (doc, score) in enumerate(fallback_docs_with_scores):
                        logger.debug(f"Fallback document {i+1} score: {score:.4f} - Path: {doc.metadata.get('path', 'no_path')}")
                    
                    # Extract documents with warning about lower relevance
                    docs = [doc for doc, score in fallback_docs_with_scores if score >= lower_threshold]
                    logger.debug(f"Retrieved {len(docs)} documents with fallback threshold")
            except Exception as e:
                logger.error(f"Error retrieving documents with scores: {e}")
                # Try without score filtering if the method isn't supported
                docs = vectorstore_retriever.get_relevant_documents(query)
                logger.debug(f"Retrieved {len(docs)} documents without score filtering")
            
            # Process in batches to avoid context overflow
            max_docs_per_batch = 10
            all_pruned_docs = []
            
            for i in range(0, len(docs), max_docs_per_batch):
                batch = docs[i:i+max_docs_per_batch]
                logger.debug(f"Processing batch {i//max_docs_per_batch + 1}, with {len(batch)} documents")
                
                # Create new documents with pruned metadata
                batch_pruned = []
                for doc in batch:
                    pruned_metadata = prune_metadata_for_context(doc.metadata)
                    pruned_doc = Document(page_content=doc.page_content, metadata=pruned_metadata)
                    batch_pruned.append(pruned_doc)
                
                all_pruned_docs.extend(batch_pruned)
            
            logger.debug(f"Returning {len(all_pruned_docs)} documents with pruned metadata")
            return all_pruned_docs
    
    return ScoreFilteringRetriever()

# In your retriever configuration
def get_retriever(vectorstore, k=10, score_threshold=0.25):
    """Get a retriever with filtering for low-relevance results."""
    base_retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold}
    )
    return base_retriever

# In the retrieval process
def retrieve_relevant_docs(self, query_text, limit=10):
    """Retrieve relevant documents with more detailed logging."""
    retriever = self.get_retriever(self.vectorstore, k=limit, score_threshold=0.25)
    docs_with_scores = retriever.get_relevant_documents(query_text, return_scores=True)
    
    # Log the scores for analysis
    for i, (doc, score) in enumerate(docs_with_scores):
        self.logger.debug(f"Document {i+1} score: {score:.4f} - Path: {doc.metadata.get('path', 'no_path')}")
    
    # Only return documents above threshold
    relevant_docs = [doc for doc, score in docs_with_scores if score >= 0.25]
    self.logger.debug(f"Retrieved {len(relevant_docs)} relevant documents after filtering")
    
    return relevant_docs

def determine_result_limit(query_text):
    """Dynamically determine how many results to return based on query type"""
    query_lower = query_text.lower()
    
    # For listing/enumeration queries, return more results
    if any(term in query_lower for term in ["list all", "all pictures", "how many", "every", "count"]):
        return 50  # Return up to 50 results for listing queries
    
    # For comparison queries, return more results
    if any(term in query_lower for term in ["compare", "most", "least", "more than", "several"]):
        return 20  # Return up to 20 results for comparison
        
    # For people queries, increase limit
    if any(term in query_lower for term in ["people", "person", "multiple people", "group", "faces"]):
        return 30  # Return up to 30 results for people queries
    
    # Default for standard queries
    return 10  # Increased from 5 to 10 for better results

def get_metadata_filters(query_text):
    """
    Generate metadata filters based on query content for any EXIF field.
    Uses only operators supported by ChromaDB.
    """
    
    logger = logging.getLogger(__name__)
    logger.debug(f"Analyzing query: {query_text}")
    
    filters = None
    query_text = query_text.lower()
    
    # ====== FIELD NAME NORMALIZATION ======
    # Extract field names that might be mentioned without "exif_" prefix
    field_pattern = r"\b(flash|iso|aperture|f-number|focal length|exposure|camera|model|make|software|date|time|state|city|country|address|resolution|orientation|white balance)\b"
    field_matches = re.findall(field_pattern, query_text, re.IGNORECASE)
    
    field_mapping = {
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
        "people": "person_count",  # Add person count field if available
    }
    
    # ====== PEOPLE/FACES DETECTION ======
    people_terms = ["people", "person", "face", "faces", "group", "crowd", "multiple people"]
    is_people_query = any(term in query_text for term in people_terms)
    
    if is_people_query:
        # Check for multiple/single person queries
        if any(term in query_text for term in ["multiple", "group", "several", "many", "two", "three", "four", "five"]):
            # Query for multiple people
            if "person_count" in field_mapping:
                filters = {'person_count': {'$gt': 1}}  # More than 1 person
            else:
                # If no person count field, try semantic search with content containing people terms
                logger.debug("No person_count field available, using semantic search for multiple people")
                # Let the semantic search handle it via vectorstore retrieval
        elif "person_count" in field_mapping:
            # Any pictures with people
            filters = {'person_count': {'$gt': 0}}
    
    # ====== LOCATION INFORMATION ======
    location_terms = ["state", "city", "country", "location", "where", "place", "taken in", "shot in", "from"]
    is_location_query = any(term in query_text for term in location_terms)
    
    # Improved pattern to extract just the location name
    location_pattern = r"(?:in|from|at|within|of)\s+(?:the\s+(?:state|city|country|province|territory)\s+of\s+)?((?:[A-Z][a-zA-Z]+)(?:\s+[A-Z][a-zA-Z]+)*)"
    location_match = re.search(location_pattern, query_text, re.IGNORECASE)
    
    # Comprehensive state/province variations
    state_variations = {
        # US States
        "alabama": ["Alabama", "AL"],
        "alaska": ["Alaska", "AK"],
        "arizona": ["Arizona", "AZ"],
        "arkansas": ["Arkansas", "AR"],
        "california": ["California", "CA"],
        "colorado": ["Colorado", "CO"],
        "connecticut": ["Connecticut", "CT"],
        "delaware": ["Delaware", "DE"],
        "florida": ["Florida", "FL"],
        "georgia": ["Georgia", "GA"],
        "hawaii": ["Hawaii", "HI"],
        "idaho": ["Idaho", "ID"],
        "illinois": ["Illinois", "IL"],
        "indiana": ["Indiana", "IN"],
        "iowa": ["Iowa", "IA"],
        "kansas": ["Kansas", "KS"],
        "kentucky": ["Kentucky", "KY"],
        "louisiana": ["Louisiana", "LA"],
        "maine": ["Maine", "ME"],
        "maryland": ["Maryland", "MD"],
        "massachusetts": ["Massachusetts", "MA"],
        "michigan": ["Michigan", "MI"],
        "minnesota": ["Minnesota", "MN"],
        "mississippi": ["Mississippi", "MS"],
        "missouri": ["Missouri", "MO"],
        "montana": ["Montana", "MT"],
        "nebraska": ["Nebraska", "NE"],
        "nevada": ["Nevada", "NV"],
        "new hampshire": ["New Hampshire", "NH"],
        "new jersey": ["New Jersey", "NJ"],
        "new mexico": ["New Mexico", "NM"],
        "new york": ["New York", "NY"],
        "north carolina": ["North Carolina", "NC"],
        "north dakota": ["North Dakota", "ND"],
        "ohio": ["Ohio", "OH"],
        "oklahoma": ["Oklahoma", "OK"],
        "oregon": ["Oregon", "OR"],
        "pennsylvania": ["Pennsylvania", "PA"],
        "rhode island": ["Rhode Island", "RI"],
        "south carolina": ["South Carolina", "SC"],
        "south dakota": ["South Dakota", "SD"],
        "tennessee": ["Tennessee", "TN"],
        "texas": ["Texas", "TX"],
        "utah": ["Utah", "UT"],
        "vermont": ["Vermont", "VT"],
        "virginia": ["Virginia", "VA"],
        "washington": ["Washington", "WA"],
        "west virginia": ["West Virginia", "WV"],
        "wisconsin": ["Wisconsin", "WI"],
        "wyoming": ["Wyoming", "WY"],
        
        # US Territories
        "puerto rico": ["Puerto Rico", "PR"],
        "guam": ["Guam", "GU"],
        "us virgin islands": ["US Virgin Islands", "USVI", "VI"],
        "american samoa": ["American Samoa", "AS"],
        "northern mariana islands": ["Northern Mariana Islands", "MP"],
        "district of columbia": ["District of Columbia", "DC", "Washington DC"],
        
        # Canadian Provinces and Territories
        "alberta": ["Alberta", "AB"],
        "british columbia": ["British Columbia", "BC"],
        "manitoba": ["Manitoba", "MB"],
        "new brunswick": ["New Brunswick", "NB"],
        "newfoundland and labrador": ["Newfoundland and Labrador", "NL"],
        "northwest territories": ["Northwest Territories", "NT"],
        "nova scotia": ["Nova Scotia", "NS"],
        "nunavut": ["Nunavut", "NU"],
        "ontario": ["Ontario", "ON"],
        "prince edward island": ["Prince Edward Island", "PE", "PEI"],
        "quebec": ["Quebec", "QC"],
        "saskatchewan": ["Saskatchewan", "SK"],
        "yukon": ["Yukon", "YT"]
    }
    
    if location_match and is_location_query and not filters:
        location_name = location_match.group(1).strip()
        logger.debug(f"Found location name: {location_name}")
        
        # Check if our location matches any known state/province
        for state_key, variations in state_variations.items():
            if location_name.lower() == state_key or location_name.lower() in [v.lower() for v in variations]:
                # If it's a known state, add variations to the filter using $in
                filters = {
                    'exif_GPSInfo_state': {'$in': variations}
                }
                logger.debug(f"Using state variations: {variations}")
                break
        
        # If no match in state variations, try a generic search
        if not filters:
            filters = {
                '$or': [
                    {'exif_GPSInfo_state': {'$eq': location_name}},
                    {'exif_GPSInfo_city': {'$eq': location_name}},
                    {'exif_GPSInfo_country': {'$eq': location_name}}
                ]
            }
    
    elif is_location_query and not filters:
        # Generic location query
        filters = {
            '$or': [
                {'exif_GPSInfo_state': {'$ne': ""}},
                {'exif_GPSInfo_city': {'$ne': ""}},
                {'exif_GPSInfo_country': {'$ne': ""}}
            ]
        }
    
    # ====== DATE/TIME INFORMATION ======
    date_terms = ["date", "when", "time", "year", "month", "day", "taken on"]
    is_date_query = any(term in query_text for term in date_terms)
    
    # Year pattern (2000-2025)
    year_pattern = r"\b(19\d{2}|20[0-2]\d)\b"
    year_match = re.search(year_pattern, query_text)
    
    if year_match and is_date_query and not filters:
        year_str = year_match.group(1)
        logger.debug(f"Found year: {year_str}")
        
        # Since we can't do substring matching, we'll use multiple patterns for year
        year_prefixes = [f"{year_str}:01", f"{year_str}:02", f"{year_str}:03", 
                       f"{year_str}:04", f"{year_str}:05", f"{year_str}:06",
                       f"{year_str}:07", f"{year_str}:08", f"{year_str}:09",
                       f"{year_str}:10", f"{year_str}:11", f"{year_str}:12"]
        
        filters = {
            '$or': [
                {'exif_DateTimeOriginal': {'$in': year_prefixes}},
                {'exif_DateTimeDigitized': {'$in': year_prefixes}}
            ]
        }
    elif is_date_query and not filters:
        # Generic date query
        filters = {
            '$or': [
                {'exif_DateTimeOriginal': {'$ne': ""}},
                {'exif_DateTimeDigitized': {'$ne': ""}}
            ]
        }
    
    # ====== CAMERA INFORMATION ======
    camera_brands = ["nikon", "canon", "sony", "fuji", "fujifilm", "olympus", "panasonic", 
                    "lg", "samsung", "apple", "iphone", "huawei", "xiaomi", "google", "pixel"]
    
    is_camera_query = any(brand in query_text for brand in camera_brands) or any(term in query_text for term in ["camera", "taken with", "shot with"])
    
    if is_camera_query and not filters:
        # Find which brands are mentioned
        mentioned_brands = []
        for brand in camera_brands:
            if brand in query_text:
                mentioned_brands.append(brand.title())  # Capitalize for exact matching
                
        if mentioned_brands:
            logger.debug(f"Found camera brands: {mentioned_brands}")
            filters = {
                '$or': [
                    {'exif_Make': {'$in': mentioned_brands}},
                    {'exif_Model': {'$in': mentioned_brands}}
                ]
            }
        else:
            # Generic camera query
            filters = {
                '$or': [
                    {'exif_Make': {'$ne': ""}},
                    {'exif_Model': {'$ne': ""}}
                ]
            }
    
    # ====== TECHNICAL PHOTO DETAILS ======
    tech_terms = ["iso", "aperture", "f/", "exposure", "focal length", "flash"]
    is_tech_query = any(term in query_text for term in tech_terms)
    
    # ISO pattern (e.g., ISO 100, ISO 400)
    iso_pattern = r"iso\s+(\d+)"
    iso_match = re.search(iso_pattern, query_text, re.IGNORECASE)
    
    if iso_match and is_tech_query and not filters:
        iso_value = iso_match.group(1)
        logger.debug(f"Found ISO value: {iso_value}")
        
        filters = {'exif_ISOSpeedRatings': {'$eq': iso_value}}
    elif "flash" in query_text and is_tech_query and not filters:
        if "with flash" in query_text:
            logger.debug("Looking for photos with flash")
            filters = {'exif_Flash': {'$ne': "0"}}
        elif "without flash" in query_text:
            logger.debug("Looking for photos without flash")
            filters = {'exif_Flash': {'$eq': "0"}}
        else:
            # Just looking for flash field
            filters = {'exif_Flash': {'$ne': ""}}
    elif is_tech_query and not filters:
        # Generic technical query
        filters = {
            '$or': [
                {'exif_ISOSpeedRatings': {'$ne': ""}},
                {'exif_FNumber': {'$ne': ""}},
                {'exif_ExposureTime': {'$ne': ""}},
                {'exif_FocalLength': {'$ne': ""}}
            ]
        }
    
    # ====== HANDLE SPECIFIC FIELD QUERIES ======
    # Check if any specific fields were mentioned
    if field_matches and not filters:
        field_filters = []
        for field in field_matches:
            field = field.lower()
            if field in field_mapping:
                mapped_fields = field_mapping[field]
                if isinstance(mapped_fields, list):
                    for mapped_field in mapped_fields:
                        field_filters.append({mapped_field: {'$ne': ""}})
                else:
                    field_filters.append({mapped_fields: {'$ne': ""}})
        
        if field_filters:
            filters = {'$or': field_filters}
            logger.debug(f"Created field-specific filter for: {field_matches}")
    
    logger.debug(f"Generated filters: {filters}")
    return filters

def retrieve_with_metadata_awareness(query_text, top_k=None):
    """Enhanced retrieval with metadata awareness and debugging."""
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Dynamically determine result limit based on query
    if top_k is None:
        top_k = determine_result_limit(query_text)
        logger.debug(f"Using dynamic top_k: {top_k} for query type")
    
    # Get the embedding manager instance
    from .embeddings import EmbeddingManager
    embedding_manager = EmbeddingManager.get_instance()
    
    # Generate metadata filters based on query
    metadata_filters = get_metadata_filters(query_text)
    logger.debug(f"Using metadata filters: {metadata_filters}")
    
    # Get the vectorstore
    vectorstore = embedding_manager.text_vectorstore
    
    # Process in batches to avoid context overflow
    max_docs_per_request = 10
    all_docs = []
    total_docs_needed = top_k
    offset = 0
    
    while len(all_docs) < total_docs_needed:
        batch_size = min(max_docs_per_request, total_docs_needed - len(all_docs))
        logger.debug(f"Retrieving batch of {batch_size} documents (offset: {offset})")
        
        # Create batch-specific retriever
        if metadata_filters:
            batch_retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": batch_size,
                    "filter": metadata_filters,
                    "offset": offset
                }
            )
            
            docs = batch_retriever.get_relevant_documents(query_text)
            if len(docs) == 0 and offset == 0:
                # If first attempt with filter returns nothing, try without filter
                logger.debug("No results with metadata filter, trying semantic search")
                batch_retriever = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": batch_size,
                        "offset": offset
                    }
                )
                docs = batch_retriever.get_relevant_documents(query_text)
        else:
            # No metadata filter, just do semantic search
            batch_retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": batch_size,
                    "offset": offset
                }
            )
            docs = batch_retriever.get_relevant_documents(query_text)
        
        # Process this batch of documents
        pruned_batch = []
        for doc in docs:
            # Create a new document with pruned metadata
            pruned_metadata = prune_metadata_for_context(doc.metadata)
            pruned_doc = Document(page_content=doc.page_content, metadata=pruned_metadata)
            pruned_batch.append(pruned_doc)
        
        # Add the pruned batch to our results
        logger.debug(f"Retrieved and pruned {len(pruned_batch)} documents in this batch")
        all_docs.extend(pruned_batch)
        
        # If we got fewer docs than requested, we've reached the end
        if len(docs) < batch_size:
            logger.debug(f"Reached end of available documents after {len(all_docs)} total")
            break
            
        # Move to next batch
        offset += len(docs)
    
    # Log summary of retrieved documents
    logger.debug(f"Total documents retrieved: {len(all_docs)}")
    
    # Log document details for a sample of documents
    for i, doc in enumerate(all_docs[:5]):  # Log first 5 docs only
        logger.debug(f"Document {i+1}:")
        logger.debug(f"Content: {doc.page_content[:100]}...")
        
        # Log key metadata fields
        meta_summary = {}
        for key in ['exif_GPSInfo_state', 'exif_GPSInfo_city', 'exif_DateTimeOriginal', 'exif_Make', 'exif_Model']:
            if key in doc.metadata:
                meta_summary[key] = doc.metadata[key]
        logger.debug(f"Key metadata: {meta_summary}")
    
    # Determine query type for response formatting
    query_type = "general"
    if any(term in query_text.lower() for term in ["location", "where", "state", "city", "country"]):
        query_type = "location"
    elif any(term in query_text.lower() for term in ["when", "date", "time", "year"]):
        query_type = "date"
    elif any(term in query_text.lower() for term in ["camera", "device", "make", "model"]):
        query_type = "camera"
    elif any(term in query_text.lower() for term in ["people", "person", "faces", "group"]):
        query_type = "people"
    
    return all_docs, query_type

def process_documents_in_batches(docs, user_text, model_type="llama3.2:1b", batch_size=10):
    """Process documents in batches to handle larger document sets."""

    
    logger = logging.getLogger("rag_app")
    logger.debug(f"Processing {len(docs)} documents in batches of {batch_size}")
    
    # Store a mapping of document IDs to full paths for later use
    doc_mapping = {}

    # Also keep track of matched images across all batches
    matched_images = set()
    
    # Format documents in batches
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
            
            # Store the mapping for later display
            doc_mapping[doc_idx] = {
                "path": path,
                "filename": filename,
                "content": doc.page_content
            }
            
            # Format the document entry with clear numbering
            formatted_batch += f"PHOTO {doc_idx} - {filename}\n"
            formatted_batch += f"DESCRIPTION: {doc.page_content.strip()}\n\n"
        
        formatted_batches.append(formatted_batch)
    
    # Initialize the LLM directly
    try:
        llm = OllamaLLM(
            model=model_type,
            temperature=0.1,
            repeat_penalty=1.2,
            num_predict=2000,
            timeout=60
        )
        logger.debug(f"Successfully initialized LLM with model {model_type}")
    except Exception as e:
        logger.error(f"Error initializing LLM with model {model_type}: {e}")
        # Try fallback model
        try:
            llm = OllamaLLM(
                model="llama3.2:1b",
                temperature=0.1,
                repeat_penalty=1.2,
                num_predict=2000,
                timeout=60
            )
            logger.debug("Successfully initialized fallback LLM model")
        except Exception as e2:
            logger.error(f"Error initializing fallback LLM: {e2}")
            return f"Error: Could not initialize language model to process your query: {e2}"
    
    # Process each batch and collect results
    results = []
    remaining_docs = len(docs)
    
    for i, batch in enumerate(formatted_batches):
        batch_size = min(batch_size, remaining_docs)
        remaining_docs -= batch_size
        
        prompt_template = f"""You are analyzing a collection of photos to answer the user's question.
Below is BATCH {i+1} of {len(formatted_batches)} with {batch_size} photos from the collection.

USER QUESTION: {user_text}

PHOTO COLLECTION (BATCH {i+1}/{len(formatted_batches)}):
{batch}

INSTRUCTIONS:
1. Carefully analyze each photo in this batch
2. Identify ALL photos that are relevant to the user's question
3. For relevant photos, provide their number and a brief explanation of why they match
4. If no photos in this batch match, state that clearly
5. Be concise but thorough

Your analysis of this batch:"""
        
        # Get batch results
        try:
            batch_result = llm.invoke(prompt_template)
            results.append(batch_result)
            logger.debug(f"Successfully processed batch {i+1}")
            
            # Extract any photo references from this batch result
            photo_pattern = r"Photo\s+(\d+)(?:\s+-\s+([^,\.\n\)]+))?(?:\.jpg|\.png|\.jpeg)?"
            for match in re.finditer(photo_pattern, batch_result):
                photo_num = int(match.group(1))
                if photo_num in doc_mapping:
                    matched_images.add(photo_num)
                    
        except Exception as e:
            logger.error(f"Error processing batch {i+1}: {e}")
            results.append(f"Error processing batch {i+1}: {e}")
    
    # Combine the results
   
    combined_prompt = f"""You are providing a final answer to the user's question based on an analysis of {len(docs)} photos.
The photos were analyzed in {len(formatted_batches)} batches, and below are the results from each batch.

USER QUESTION: {user_text}

BATCH ANALYSIS RESULTS:
{"-"*50}
{"\n\n" + "-"*50 + "\n\n".join(results)}
{"-"*50}

INSTRUCTIONS:
1. Based on the batch analyses above, provide a comprehensive answer to the user's question
2. List ALL relevant photos identified across ALL batches
3. Include EVERY photo that was mentioned as relevant in ANY batch
4. Do not summarize or truncate the list of matches
5. Reference each photo by its full identifier (e.g., "Photo 12 - filename.jpg")
6. Organize your response with a clear section titled "MATCHED PHOTOS:" that lists all matches
7. Be thorough and do not omit any photos that were identified as relevant

Your final answer to the user's question:"""
    
    # ... rest of the function ...
    
    # Get the final response
    try:
        final_response = llm.invoke(combined_prompt)
        
        # Extract any additional photo references from the final response
        photo_pattern = r"Photo\s+(\d+)(?:\s+-\s+([^,\.\n\)]+))?(?:\.jpg|\.png|\.jpeg)?"
        for match in re.finditer(photo_pattern, final_response):
            photo_num = int(match.group(1))
            if photo_num in doc_mapping:
                matched_images.add(photo_num)
                
        # Create a filtered doc_mapping that only includes matched images
        matched_doc_mapping = {k: v for k, v in doc_mapping.items() if k in matched_images}
        
        logger.debug(f"Found {len(matched_images)} matched images: {matched_images}")
        
        return {
            "response": final_response, 
            "doc_mapping": matched_doc_mapping,
            "matched_images": list(matched_images)
        }
    except Exception as e:
        logger.error(f"Error generating final response: {e}")
        # Fallback to returning the individual batch results
        fallback_response = f"I found the following relevant photos for your query:\n\n{'\n\n'.join(results)}"
        return {"response": fallback_response, "doc_mapping": doc_mapping}