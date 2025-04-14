import logging
import sys
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from rag.embeddings import EmbeddingManager
from rag.retriever.query_analyzer import determine_result_limit
import config

logger = logging.getLogger("rag_app")

def prune_metadata_for_context(metadata):
    """Remove bulky fields not needed for answering queries"""
    fields_to_remove = [
        'exif_MakerNote', 'exif_UserComment', 'exif_json'
    ]
    
    # Remove the fields
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
    """
    from ..embeddings import EmbeddingManager
    
    # Set score threshold based on query type if not provided
    if score_threshold is None:
        # Use higher threshold for factual queries
        if any(keyword in query_text.lower().split() for keyword in 
              ["who", "what", "when", "where", "why", "how", "which"]):
            score_threshold = 0.8
        else:
            score_threshold = 0.5
            
    logger.debug(f"Using similarity score threshold: {score_threshold}")
    
    # Get embedding manager instance
    embedding_manager = EmbeddingManager.get_instance()
    vectorstore = embedding_manager.text_vectorstore
    
    logger.debug(f"Vectorstore initialized: {vectorstore is not None}")
    
    if vectorstore is None:
        logger.error("Vectorstore is None - check database initialization")
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
    
    # Create custom retriever for score filtering and metadata pruning
    from langchain_core.retrievers import BaseRetriever
    
    class ScoreFilteringRetriever(BaseRetriever):
        def _get_relevant_documents(self, query, **kwargs):
            try:
                # Get documents with scores
                docs_with_scores = vectorstore.similarity_search_with_score(
                    query, k=top_k, filter=metadata_filters
                )
                
                # Log the similarity scores for analysis
                if docs_with_scores:
                    for i, (doc, score) in enumerate(docs_with_scores):
                        logger.debug(f"Document {i+1} score: {score:.4f} - Path: {doc.metadata.get('path', 'no_path')}")
                    
                    # Filter documents by threshold
                    docs = [doc for doc, score in docs_with_scores if score >= score_threshold]
                else:
                    docs = []
                    
                logger.debug(f"Retrieved {len(docs)} documents after score filtering")
                
                # Try with lower threshold if no results and not already using filters
                if not docs and not metadata_filters and kwargs.get('fallback_allowed', True):
                    logger.debug(f"No documents met threshold {score_threshold}, trying lower threshold")
                    lower_threshold = max(0.1, score_threshold - 0.2)
                    kwargs['fallback_allowed'] = False  # Prevent recursion
                    
                    fallback_docs_with_scores = vectorstore.similarity_search_with_score(
                        query, k=top_k, filter=metadata_filters
                    )
                    
                    for i, (doc, score) in enumerate(fallback_docs_with_scores):
                        logger.debug(f"Fallback document {i+1} score: {score:.4f}")
                    
                    docs = [doc for doc, score in fallback_docs_with_scores if score >= lower_threshold]
                    
            except Exception as e:
                logger.error(f"Error retrieving documents with scores: {e}")
                # Try without score filtering if method not supported
                vectorstore_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
                docs = vectorstore_retriever.get_relevant_documents(query)
                logger.debug(f"Retrieved {len(docs)} documents without score filtering")
            
            # Process in batches to avoid context overflow
            max_docs_per_batch = 10
            all_pruned_docs = []
            
            for i in range(0, len(docs), max_docs_per_batch):
                batch = docs[i:i+max_docs_per_batch]
                
                # Create new documents with pruned metadata
                batch_pruned = []
                for doc in batch:
                    pruned_metadata = prune_metadata_for_context(doc.metadata)
                    pruned_doc = Document(page_content=doc.page_content, metadata=pruned_metadata)
                    batch_pruned.append(pruned_doc)
                
                all_pruned_docs.extend(batch_pruned)
            
            return all_pruned_docs
    
    return ScoreFilteringRetriever()

