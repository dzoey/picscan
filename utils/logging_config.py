import logging

def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("rag_app.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create a logger for the application
    logger = logging.getLogger("rag_app")
    return logger

# Create and export the logger instance
logger = setup_logging()