import logging
from typing import Dict, Any, Optional

import config
from rag.processor.query_classifier import QueryClassifier
from rag.processor.metadata_handler import MetadataHandler
from rag.processor.llm_manager import LLMManager
from rag.processor.document_retriever import DocumentRetriever
from rag.processor.response_formatter import ResponseFormatter
from rag.processor.specialized_handlers import ExifMetadataProcessor, StandardRagProcessor, BatchedDocumentHandler

# Import retriever components
from rag.retriever.core import get_langchain_retriever
from rag.retriever.query_analyzer import determine_result_limit


# Other imports
from rag.embeddings import EmbeddingManager

# Configure logger
logger = logging.getLogger("rag_app")

class QueryProcessor:
    """Main processor that coordinates the query processing pipeline"""
    
    def __init__(self, embedding_manager: EmbeddingManager = None):
        """Initialize the query processor with required components"""
        # Initialize embedding manager
        self.embedding_manager = embedding_manager or EmbeddingManager.get_instance()
        
        # Initialize component modules
        self.query_classifier = QueryClassifier()
        self.metadata_handler = MetadataHandler()
        self.document_retriever = DocumentRetriever(self.embedding_manager)
        self.llm_manager = LLMManager()
        self.response_formatter = ResponseFormatter()
        
        # Initialize specialized handlers
        self.exif_handler = ExifMetadataProcessor(
            self.metadata_handler, 
            self.document_retriever,
            self.response_formatter,
            self.llm_manager,
            self.query_classifier
        )
        self.rag_handler = StandardRagProcessor(self.response_formatter)
        self.batch_handler = BatchedDocumentHandler(self.response_formatter)
    
    def process_query(self, user_text: str, image_path: str, model: str):
        try:
            logger.debug(f"Processing query with text: '{user_text}', image: {image_path}, model: {model}")

            # Initialize LLM first since we need it for both factual and visual queries
            llm = self.llm_manager.initialize_llm(model)
            
            # Check if this is a factual query that shouldn't display images
            is_factual_query = self.query_classifier.is_factual_query(user_text, llm)
            logger.debug(f"Query classified as factual: {is_factual_query}")
            
            # HANDLE FACTUAL QUERIES IMMEDIATELY - Exit early before retrieval
            if is_factual_query:
                # If this is a factual query, provide direct answer without images
                direct_answer = self.llm_manager.get_direct_answer(user_text, llm)
                logger.debug("Providing direct answer for factual query (no document retrieval)")
                return {
                    "answer": direct_answer,
                    "text": direct_answer,
                    "documents": [],  # Don't include documents for factual queries
                    "query_type": "direct_answer",
                    "suppress_images": True
                }
            
            # Detect query characteristics
            is_multiple_people_query = self.query_classifier.is_multiple_people_query(user_text)
            metadata_focus = self.query_classifier.detect_metadata_focus(user_text)
            is_exif_query = self.query_classifier.is_exif_query(user_text, is_factual_query)
            
            # If it's an EXIF metadata query, use specialized processing
            if is_exif_query:
                logger.debug(f"Using specialized EXIF metadata processing for query with focus: {metadata_focus}")
                # Pass factual flag to EXIF query processing
                result = self.exif_handler.process_exif_metadata_query(
                    user_text, 
                    llm, 
                    is_factual=is_factual_query
                )
                # For factual queries, always suppress images
                if is_factual_query:
                    result["suppress_images"] = True
                return result
                    
            # For other queries, determine if we should use RAG
            use_rag = self.query_classifier.should_use_rag(user_text)
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
                    direct_answer = self.llm_manager.get_direct_answer(user_text, llm)
                    return {
                        "answer": direct_answer,
                        "text": direct_answer,
                        "documents": [],  # Don't include documents for factual queries
                        "query_type": "direct_answer",
                        "suppress_images": True
                    }
                
                # For multiple people queries or any query that needs to analyze many documents
                if is_multiple_people_query or len(docs) > 10:
                    logger.debug(f"Using batched document processing for query: {user_text}")
                    return self.batch_handler.process_batched_query(
                        user_text, 
                        docs, 
                        llm, 
                        model, 
                        is_factual_query,
                        metadata_focus
                    )
                else:
                    # Standard RAG processing for simpler queries
                    return self.rag_handler.process_standard_rag_query(
                        user_text,
                        docs,
                        llm,
                        metadata_focus,
                        is_factual_query
                    )
            else:
                # For queries that don't need RAG, provide a direct answer
                direct_answer = self.llm_manager.get_direct_answer(user_text, llm)
                return {
                    "answer": direct_answer,
                    "text": direct_answer,
                    "documents": [],
                    "query_type": "direct_answer",
                    "suppress_images": True
                }
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                "answer": f"I encountered an error while processing your query: {str(e)}",
                "text": f"I encountered an error while processing your query: {str(e)}",
                "error": str(e),
                "documents": [],
                "query_type": "error",
                "suppress_images": True
            }

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

