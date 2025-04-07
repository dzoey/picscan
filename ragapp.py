from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from pathlib import Path

# Import configuration
import config

# Import utilities
from utils.logging_config import logger
from utils.image import encode_images_to_base64

# Import RAG components
# CHANGE THIS SECTION:
import importlib
import rag.processor
importlib.reload(rag.processor)
# Get a processor instance instead of direct function
from rag.processor import get_processor, get_available_models
processor = get_processor()  # Create an instance to use

# Rest of your imports
from rag.embeddings import embedding_manager
from werkzeug.utils import secure_filename

# Initialize Flask application
app = Flask(__name__)

# Initialize components at startup
logger.debug("Initializing application components")
embedding_manager.initialize()

@app.route('/')
def index():
    """Render the main page"""
    try:
        models = get_available_models()
        logger.debug(f"Rendering index page with {len(models)} models")
        return render_template('index.html', models=models)
    except Exception as e:
        logger.error(f"Error rendering index page: {e}", exc_info=True)
        return "Error loading the application. Please check the logs."


@app.route('/query', methods=['POST'])
def query():
    """Handle query requests from the frontend"""
    try:
        # Get data from form data (multipart/form-data encoding)
        query_text = request.form.get('text', '')
        model = request.form.get('model', 'granite3.2-vision:latest')
        
        # Handle file upload if present
        image_path = None
        if 'image' in request.files and request.files['image'].filename:
            file = request.files['image']
            # Create uploads directory if it doesn't exist
            uploads_dir = os.path.join(app.static_folder, 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            
            # Save the file
            filename = secure_filename(file.filename)
            image_path = os.path.join(uploads_dir, filename)
            file.save(image_path)
            
        logger.debug(f"Received query request with text: '{query_text}', model: {model}, image: {image_path}")
        
        # Call method on the processor instance
        result = processor.process_query(user_text=query_text, image_path=image_path, model=model)
        
        # Add debugging
        logger.debug(f"Raw result from process_query: {result}")
        logger.debug(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")
        
        # Ensure result has required keys
        if not isinstance(result, dict):
            logger.error(f"Result is not a dictionary: {type(result)}")
            result = {"error": "Invalid result format", "answer": "Error processing query"}
        
        # Add text key if missing (backwards compatibility)
        if "text" not in result and "answer" in result:
            result["text"] = result["answer"]
        elif "text" not in result:
            result["text"] = "No response generated"
            
        # Create response object
        response = {
            "text": result.get("text", "No response generated"),
            "documents": result.get("documents", []),
            "query_type": result.get("query_type", "unknown"),
            "metadata_focus": result.get("metadata_focus", None),
            "suppress_images": result.get("suppress_images", False)  # Pass through the suppress flag
        }
        
        # Add HTML content if available
        if "is_html" in result and result["is_html"]:
            response["is_html"] = True
            response["html_content"] = result.get("answer", "")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing query request: {str(e)}", exc_info=True)
        return jsonify({
            "text": f"Error processing query: {str(e)}",
            "error": str(e),
            "documents": [],
            "query_type": "error"
        })

@app.route('/image/<path:image_path>')
def serve_image(image_path):
    """Serve images from absolute paths"""
    try:
        # Remove any leading slashes to standardize the path
        if image_path.startswith('/'):
            image_path = image_path[1:]
        
        logger.debug(f"Requested image: {image_path}")
        
        # Check if the path is an absolute path
        if os.path.isabs(image_path) or image_path.startswith('home/'):
            # Convert to absolute path if not already
            if not os.path.isabs(image_path):
                image_path = f"/{image_path}"
                
            # Check if file exists
            if os.path.isfile(image_path):
                directory = os.path.dirname(image_path)
                filename = os.path.basename(image_path)
                logger.debug(f"Serving absolute path image: {filename} from {directory}")
                return send_from_directory(directory, filename)
            else:
                logger.warning(f"Absolute path image not found: {image_path}")
        
        # Try common directories as fallback
        possible_dirs = [
            os.path.join(app.static_folder, 'uploads'),
            config.MEDIA_DIR if hasattr(config, 'MEDIA_DIR') else None,
            os.path.join(os.getcwd(), "media"),
            os.path.join(os.getcwd(), "data", "images"),
        ]
        
        # Filter out None values
        possible_dirs = [d for d in possible_dirs if d]
        
        # First, check if the image is a filename in one of our directories
        filename = os.path.basename(image_path)
        for directory in possible_dirs:
            if not os.path.isdir(directory):
                continue
                
            full_path = os.path.join(directory, filename)
            if os.path.isfile(full_path):
                logger.debug(f"Serving image by filename: {filename} from {directory}")
                return send_from_directory(directory, filename)
        
        logger.warning(f"Image not found after all attempts: {image_path}")
        return "Image not found", 404
        
    except Exception as e:
        logger.error(f"Error serving image {image_path}: {e}")
        return f"Error serving image: {str(e)}", 500

if __name__ == '__main__':
    # Ensure all required directories exist
    config.TEMP_UPLOADS_DIR.mkdir(exist_ok=True)
    
    # Run the Flask application
    app.run(
        debug=config.FLASK_DEBUG,
        host=config.FLASK_HOST,
        port=config.FLASK_PORT
    )