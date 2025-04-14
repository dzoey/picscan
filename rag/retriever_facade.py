# Import from the new modular structure
from rag.retriever.core import get_langchain_retriever, prune_metadata_for_context
from rag.retriever.query_analyzer import determine_result_limit
from rag.retriever.metadata_handler import get_metadata_filters
from rag.retriever.document_processor import process_documents_in_batches
from utils.exif_utils import detect_exif_query_type

# Re-export for backward compatibility
__all__ = [
    'get_langchain_retriever',
    'prune_metadata_for_context',
    'determine_result_limit',
    'get_metadata_filters',
    'process_documents_in_batches',
    'detect_exif_query_type'
]
