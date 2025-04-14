import logging
import re

logger = logging.getLogger("rag_app")

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
    return 10

