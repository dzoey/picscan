# RAG package initialization
from rag.embeddings import embedding_manager
from rag.retriever import get_langchain_retriever
from rag.processor import process_query, get_available_models, process_metadata_query

# Export commonly used functions and objects
__all__ = [
    'embedding_manager',
    'get_langchain_retriever',
    'convert_to_langchain_documents',
    'process_query',
    'get_available_models',
    'process_metadata_query'
]