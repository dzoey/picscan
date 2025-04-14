import logging
import numpy as np
from typing import List, Dict, Any, Optional
from rag.embeddings import EmbeddingManager
from rag.retriever.core import get_langchain_retriever
import config

logger = logging.getLogger("rag_app")

class DocumentRetriever:
    """Handles document retrieval and filtering"""
    
    def __init__(self, embedding_manager=None):
        """Initialize the document retriever with an embedding manager"""
        self.embedding_manager = embedding_manager
    
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
                from .retriever import get_langchain_retriever
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
    
    def check_document_relevance(self, query: str, docs: List[Any], threshold: float = 0.4) -> bool:
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
        v1 = np.array(v1)
        v2 = np.array(v2)
        
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0
            
        return dot_product / (norm_v1 * norm_v2)

