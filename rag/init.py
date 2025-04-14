# RAG package initialization
from rag.embeddings import embedding_manager
from rag.retriever_facade import get_langchain_retriever
from rag.query_processor import process_query
from rag.processor.llm_manager import get_available_models

# Export commonly used functions and objects
__all__ = [
    'embedding_manager',
    'get_langchain_retriever',
    'process_query',
    'get_available_models'
]