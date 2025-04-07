from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings


from utils.logging_config import logger
import config

class EmbeddingManager:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.initialize()
        return cls._instance
        
    def __init__(self):
        self.embeddings = None
        self.chroma_client = None
        self.text_vectorstore = None
        self.image_vectorstore = None
        self.initialized = False
    
    def initialize(self):
        """Initialize embedding models and vector stores"""
        # Skip if already initialized
        if self.initialized:
            logger.debug("Embedding manager already initialized")
            return True
            
        try:
            logger.debug("Initializing LangChain components and ChromaDB collections")
            
            # Initialize HuggingFace embeddings through LangChain
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=config.DB_PATH, 
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Initialize LangChain vector stores for text and images
            self.text_vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=f"{config.DB_COLLECTION_NAME}_text",
                embedding_function=self.embeddings
            )
            
            self.image_vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=f"{config.DB_COLLECTION_NAME}_image",
                embedding_function=self.embeddings
            )
            
            self.initialized = True
            logger.debug("LangChain initialization complete")
            return True
        except Exception as e:
            logger.error(f"Error during initialization: {e}", exc_info=True)
            return False
    
    def query_exif_field(self, field_path: str) -> list:
        """
        Query the database for unique values of any EXIF field
        
        Args:
            field_path: The field path (e.g., 'GPSInfo_state' or 'Make')
            
        Returns:
            List of unique values for that field
        """
        try:
            # Add the exif_ prefix to match database structure
            full_field_name = f"exif_{field_path}"
            
            # Get native ChromaDB client collection for direct access
            collection = self.chroma_client.get_collection(name=f"{config.DB_COLLECTION_NAME}_text")
            results = collection.get(include=["metadatas"])
            
            if not results or "metadatas" not in results or not results["metadatas"]:
                logger.debug(f"No entries with metadata found")
                return []
                
            # Extract unique values for the specific field
            values = set()
            field_count = 0
            total_entries = len(results["metadatas"])
            
            for metadata in results["metadatas"]:
                if metadata and full_field_name in metadata and metadata[full_field_name]:
                    value = str(metadata[full_field_name]).strip()
                    if value:
                        values.add(value)
                        field_count += 1
                        
            values_list = sorted(list(values))
            logger.debug(f"Found {field_count}/{total_entries} entries with {field_path}")
            logger.debug(f"Found {len(values_list)} unique values for {field_path}: {values_list}")
            return values_list
        except Exception as e:
            logger.error(f"Error querying {field_path}: {e}", exc_info=True)
            return []

# Create a singleton instance
embedding_manager = EmbeddingManager()