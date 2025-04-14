import config
from langchain_ollama import OllamaLLM
import re
import logging

logger = logging.getLogger("rag_app")

def process_documents_in_batches(docs, user_text, model_type=config.DEFAULT_LLM_MODEL, batch_size=10, llm=None):
    """Process documents in batches to handle larger document sets."""
    logger.debug(f"Processing {len(docs)} documents in batches of {batch_size}")
    
    # Setup for processing
    doc_mapping = _prepare_document_mapping(docs)
    matched_images = set()
    
    # Format documents in batches
    formatted_batches = _format_document_batches(docs, batch_size)
    
    # Initialize LLM if not provided
    if llm is None:
        llm = _initialize_llm(model_type)
        if isinstance(llm, str):  # Error message returned
            return llm
    
    # Process each batch and collect results
    results = _process_batches(formatted_batches, user_text, llm, doc_mapping, matched_images)
    
    # Generate final response
    return _generate_final_response(results, user_text, llm, doc_mapping, matched_images)

def _prepare_document_mapping(docs):
    """Create a mapping of document IDs to paths and filenames."""
    doc_mapping = {}
    for i, doc in enumerate(docs):
        doc_idx = i + 1  # 1-based indexing
        metadata = doc.metadata
        path = metadata.get("path", "Unknown")
        filename = path.split("/")[-1] if "/" in path else path
        
        doc_mapping[doc_idx] = {
            "path": path,
            "filename": filename,
            "content": doc.page_content
        }
    return doc_mapping

def _format_document_batches(docs, batch_size):
    """Format documents into batches for processing."""
    formatted_batches = []
    total_docs = len(docs)
    
    for i in range(0, total_docs, batch_size):
        end_idx = min(i + batch_size, total_docs)
        batch = docs[i:end_idx]
        
        formatted_batch = f"BATCH {i//batch_size + 1} OF {(total_docs + batch_size - 1) // batch_size}:\n\n"
        
        for j, doc in enumerate(batch):
            doc_idx = i + j + 1  # Global document index (1-based)
            formatted_batch += _format_single_document(doc_idx, doc)
        
        formatted_batches.append(formatted_batch)
    
    return formatted_batches

def _format_single_document(doc_idx, doc):
    """Format a single document with metadata for the prompt."""
    metadata = doc.metadata
    path = metadata.get("path", "Unknown")
    filename = path.split("/")[-1] if "/" in path else path
    
    formatted_doc = f"PHOTO {doc_idx} - {filename}\n"
    formatted_doc += f"DESCRIPTION: {doc.page_content.strip()}\n"
    formatted_doc += "METADATA:\n"
    
    # Add date/time
    if "exif_DateTimeOriginal" in metadata and metadata["exif_DateTimeOriginal"]:
        formatted_doc += f"Time: {metadata.get('exif_DateTimeOriginal')}\n"
    elif "exif_DateTimeDigitized" in metadata and metadata["exif_DateTimeDigitized"]:
        formatted_doc += f"Time: {metadata.get('exif_DateTimeDigitized')}\n"
    
    # Add location
    location_parts = []
    for field in ["exif_GPSInfo_city", "exif_GPSInfo_state", "exif_GPSInfo_country"]:
        if field in metadata and metadata[field]:
            location_parts.append(metadata[field])
    
    if location_parts:
        formatted_doc += f"Location: {', '.join(location_parts)}\n"
    
    formatted_doc += "\n"
    return formatted_doc

def _initialize_llm(model_type):
    """Initialize the LLM with appropriate settings."""
    try:
        llm = OllamaLLM(
            model=model_type,
            context_window=config.LLM_CONTEXT_SIZE,
            temperature=config.DEFAULT_TEMPERATURE,
            repeat_penalty=config.DEFAULT_REPEAT_PENALTY,
            num_predict=config.DEFAULT_NUM_PREDICT,
            timeout=config.DEFAULT_TIMEOUT
        )
        logger.debug(f"Successfully initialized LLM with model {model_type}")
        return llm
    except Exception as e:
        logger.error(f"Error initializing LLM with model {model_type}: {e}")
        # Try fallback model
        try:
            llm = OllamaLLM(
                model=config.FALLBACK_MODELS[0],
                context_window=config.LLM_CONTEXT_SIZE,
                temperature=config.DEFAULT_TEMPERATURE,
                repeat_penalty=config.DEFAULT_REPEAT_PENALTY,
                num_predict=config.DEFAULT_NUM_PREDICT,
                timeout=config.DEFAULT_TIMEOUT
            )
            logger.debug("Successfully initialized fallback LLM model")
            return llm
        except Exception as e2:
            logger.error(f"Error initializing fallback LLM: {e2}")
            return f"Error: Could not initialize language model to process your query: {e2}"

def _process_batches(formatted_batches, user_text, llm, doc_mapping, matched_images):
    """Process each batch and collect results."""
    results = []
    
    for i, batch in enumerate(formatted_batches):
        prompt_template = f"""You are analyzing a collection of photos to answer the user's question.
Below is BATCH {i+1} of {len(formatted_batches)} with photos from the collection.

USER QUESTION: {user_text}

PHOTO COLLECTION (BATCH {i+1}/{len(formatted_batches)}):
{batch}

INSTRUCTIONS:
1. Carefully analyze each photo in this batch
2. Identify ALL photos that are relevant to the user's question
3. For relevant photos, provide their number and a brief explanation of why they match
4. If no photos in this batch match, state that clearly
5. Be concise but thorough

Your analysis of this batch:"""
        
        try:
            batch_result = llm.invoke(prompt_template)
            results.append(batch_result)
            
            # Extract photo references
            _extract_photo_references(batch_result, doc_mapping, matched_images)
                
        except Exception as e:
            logger.error(f"Error processing batch {i+1}: {e}")
            results.append(f"Error processing batch {i+1}: {e}")
    
    return results

def _extract_photo_references(text, doc_mapping, matched_images):
    """Extract photo references from text and update matched_images set."""
    photo_pattern = r"Photo\s+(\d+)(?:\s+-\s+([^,\.\n\)]+))?(?:\.jpg|\.png|\.jpeg)?"
    for match in re.finditer(photo_pattern, text):
        photo_num = int(match.group(1))
        if photo_num in doc_mapping:
            matched_images.add(photo_num)

def _generate_final_response(results, user_text, llm, doc_mapping, matched_images):
    """Generate final combined response."""
    combined_prompt = f"""You are providing a final answer to the user's question based on an analysis of photos.
The photos were analyzed in batches, and below are the results from each batch.

USER QUESTION: {user_text}

BATCH ANALYSIS RESULTS:
{"-"*50}
{"\n\n" + "-"*50 + "\n\n".join(results)}
{"-"*50}

INSTRUCTIONS:
1. Based on the batch analyses above, provide a comprehensive answer to the user's question
2. List ALL relevant photos identified across ALL batches
3. For EACH relevant photo, include:
   - The photo number and filename
   - A brief description specific to that photo
   - Any available time/date information
   - Any available location information
4. Format each photo's information in separate paragraphs
5. Organize your response with a section titled "MATCHED PHOTOS:" that lists all matches
6. Each photo needs its own unique description

Your final answer to the user's question:"""
    
    try:
        final_response = llm.invoke(combined_prompt)
        
        # Extract any additional photo references
        _extract_photo_references(final_response, doc_mapping, matched_images)
            
        # Create filtered doc_mapping with only matched images
        matched_doc_mapping = {k: v for k, v in doc_mapping.items() if k in matched_images}
        
        logger.debug(f"Found {len(matched_images)} matched images")
        
        return {
            "response": final_response, 
            "doc_mapping": matched_doc_mapping,
            "matched_images": list(matched_images)
        }
    except Exception as e:
        logger.error(f"Error generating final response: {e}")
        # Fallback to returning individual batch results
        fallback_response = f"I found the following relevant photos for your query:\n\n{'\n\n'.join(results)}"
        return {"response": fallback_response, "doc_mapping": doc_mapping}

