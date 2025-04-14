import logging
from langchain_ollama import OllamaLLM
import config

logger = logging.getLogger("rag_app")

class QueryClassifier:
    """Handles classification of queries (FACTUAL vs VISUAL, metadata focus, etc.)"""
    
    def __init__(self):
        """Initialize the query classifier"""
        pass
        
    def is_factual_query(self, query_text: str, llm=None) -> bool:
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
    - FACTUAL: Questions seeking information, statistics, counts, data extraction, or general knowledge. This includes historical facts, definitions, and information that doesn't require viewing images.
    - VISUAL: Requests to see, view, show, or display specific types of images. The user primarily wants to SEE the actual images.

    Key indicators of VISUAL queries:
    - Contains verbs like "show", "see", "view", "display" followed by references to pictures/photos
    - The primary intent is to view the images themselves
    - User is asking what something "looks like" in the context of the image collection

    Examples of VISUAL queries:
    - "Show me scenic pictures"
    - "Find pictures with vehicles"
    - "Get photos from California"
    - "What pictures have people in them?"
    - "I want to see images of mountains"

    Examples of FACTUAL queries:
    - "How many pictures were taken in 2022?"
    - "What camera was used for most photos?"
    - "When was my trip to Paris?"
    - "Who was the 32nd president of the United States?"
    - "What is the capital of France?"
    - "When did World War II end?"
    - "Who discovered penicillin?"
    - "What are the planets in our solar system?"

    USER QUERY: {query_text}

    CLASSIFICATION RULES:
    1. If the query is asking to SEE, SHOW, VIEW, FIND or GET PICTURES/PHOTOS/IMAGES, it is VISUAL.
    2. If the query is about general knowledge, history, facts, or information not related to viewing photos, it is FACTUAL.
    3. Questions about who, what, when, where that don't explicitly ask to see images are FACTUAL.

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
                # Check for knowledge queries before defaulting
                if self._is_general_knowledge_query(query_text):
                    logger.debug(f"Ambiguous classification, detected general knowledge query: {query_text}")
                    return "FACTUAL"
                else:
                    logger.debug(f"LLM gave ambiguous classification for query, defaulting to VISUAL: {query_text}")
                    return "VISUAL"
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            # Check if this is a general knowledge query before defaulting
            if self._is_general_knowledge_query(query_text):
                return "FACTUAL"
            return "VISUAL"

    def _is_general_knowledge_query(self, query_text: str) -> bool:
        """Detect if this is a general knowledge question not related to images."""
        query_lower = query_text.lower()
        
        # Historical and general knowledge indicators
        knowledge_indicators = [
            "who was", "what was", "when was", "where was", "why was",
            "who is", "what is", "when is", "where is", "why is",
            "who are", "what are", "when are", "where are", "why are",
            "president", "war", "history", "capital", "country", "discovered",
            "invented", "founded", "created", "born", "died"
        ]
        
        # Photo/image related words
        photo_related = ["photo", "picture", "image", "camera", "photographer"]
        
        # If it contains knowledge indicators and doesn't mention photos/images
        if any(indicator in query_lower for indicator in knowledge_indicators) and \
        not any(term in query_lower for term in photo_related):
            return True
            
        return False
    
    def _is_factual_query_keywords(self, query_text: str) -> bool:
        """Use keyword matching to determine if a query is factual."""
        # Convert to lowercase for case-insensitive matching
        query_lower = query_text.lower()
        
        # Keywords that usually indicate factual queries
        factual_indicators = [
            "how many", "count", "number of", "total", "statistics", 
            "when was", "when did", "what camera", "which lens", 
            "what settings", "average", "summarize"
        ]
        
        # Keywords that strongly indicate visual queries
        visual_indicators = [
            "show me", "display", "see", "view", "find picture", "find photo", 
            "get picture", "get photo", "get image", "show picture", "show photo"
        ]
        
        # Check for visual indicators first (they have precedence)
        for indicator in visual_indicators:
            if indicator in query_lower:
                return False  # Not factual, it's a visual query
    
        # Then check for factual indicators
        for indicator in factual_indicators:
            if indicator in query_lower:
                return True  # It's a factual query
                
        # Default: queries about photos are considered visual unless proven factual
        return False
    
    def detect_metadata_focus(self, query_text: str) -> str:
        """Detect if the query is focusing on specific metadata."""
        metadata_keywords = {
            "date": ["date", "when", "time", "day", "month", "year"],
            "location": ["location", "where", "place", "city", "country", "state"],
            "people": ["person", "people", "faces", "man", "woman", "child"],
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
    
    def is_multiple_people_query(self, query_text: str) -> bool:
        """Check if this is a query about multiple people in photos."""
        multiple_people_terms = [
            "multiple people", "more than one person", "several people", 
            "group of people", "group photo"
        ]
        return any(term in query_text.lower() for term in multiple_people_terms)
    
    def is_exif_query(self, query_text: str, is_factual: bool = False) -> bool:
        """Detect if this is an EXIF metadata query."""
        exif_keywords = [
            "exif", "metadata", "state", "location", "where", "when", "date", 
            "time", "camera", "make", "model", "iso", "flash", "aperture"
        ]
        
        historical_keywords = [
            "history", "historical", "ancient", "past", "century", 
            "war", "king", "queen", "president", "leader", "empire", 
            "nation", "country", "government", "founding", "constitution"
        ]
        
        query_lower = query_text.lower()
        is_exif_keyword_match = any(term in query_lower for term in exif_keywords)
        is_historical_match = any(term in query_lower for term in historical_keywords)
        
        # Don't treat historical factual queries as EXIF queries even if they contain EXIF keywords
        if is_exif_keyword_match and is_factual and is_historical_match:
            logger.debug(f"Detected historical factual query, not treating as EXIF query")
            return False
            
        return is_exif_keyword_match
