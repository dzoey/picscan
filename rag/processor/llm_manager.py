import logging
from langchain_ollama import OllamaLLM
import config

logger = logging.getLogger("rag_app")

class LLMManager:
    """Manages LLM initialization and interaction"""
    
    def __init__(self):
        """Initialize the LLM manager"""
        pass
    
    def initialize_llm(self, model):
        """Initialize an LLM instance with the specified model."""
        # Check if the model is a text generation model
        if not self.is_text_generation_model(model):
            logger.warning(f"Model {model} appears to be an embedding model, not suitable for text generation")
            return None
            
        # Initialize the model with timeout
        try:
            llm = OllamaLLM(
                model=model,
                context_window=config.LLM_CONTEXT_SIZE,
                temperature=config.DEFAULT_TEMPERATURE,
                repeat_penalty=config.DEFAULT_REPEAT_PENALTY,
                num_predict=config.DEFAULT_NUM_PREDICT,
                timeout=config.DEFAULT_TIMEOUT
            )
            return llm
        except Exception as e:
            logger.error(f"Error initializing Ollama LLM with model {model}: {e}")
            # Try fallback to a default model from the config
            try:
                llm = OllamaLLM(
                    model=config.FALLBACK_MODELS[0],
                    context_window=config.LLM_CONTEXT_SIZE,
                    temperature=config.DEFAULT_TEMPERATURE,
                    repeat_penalty=config.DEFAULT_REPEAT_PENALTY,
                    num_predict=config.DEFAULT_NUM_PREDICT,
                    timeout=config.DEFAULT_TIMEOUT
                )
                return llm
            except Exception as e2:
                logger.error(f"Error initializing fallback model: {e2}")
                return None
    
    def is_text_generation_model(self, model_name):
        """Check if the model is capable of text generation."""
        # List of known embedding-only models
        embedding_models = ["all-minilm", "all-mpnet", "nomic-embed", "bge-"]
        
        # Check if the model name contains any embedding model identifiers
        for embedding_model in embedding_models:
            if embedding_model in model_name.lower():
                return False
                
        return True
    
    def get_direct_answer(self, question: str, llm) -> str:
        """Get a direct answer from the LLM without using RAG context."""
        try:
            if llm is None:
                return f"I don't have specific information about {question} in my photo collection."
                
            prompt = f"""You are a helpful assistant answering questions based on your knowledge.
            
Question: {question}

Please provide a concise and accurate answer based on factual information:"""

            return llm.invoke(prompt)
        except Exception as e:
            logger.error(f"Error getting direct answer: {str(e)}")
            return f"I'm not able to provide information about {question} at this time."

def get_available_models():
    """
    Return a list of available Ollama models for inference.
    
    Returns:
        List of model names
    """
    try:
        import requests
        
        # Try to get models from Ollama API
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            # Extract model names
            model_names = [model["name"] for model in models]
            return model_names
        else:
            logger.error(f"Failed to get models: {response.status_code}")
            # Return fallback models from config
            return config.FALLBACK_MODELS
            
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        # Return fallback models from config
        return config.FALLBACK_MODELS

