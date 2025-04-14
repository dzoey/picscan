import logging
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rag.retriever.core import get_langchain_retriever
from rag.retriever.document_processor import process_documents_in_batches

logger = logging.getLogger("rag_app")

# Define prompt templates used by specialized handlers
EXIF_METADATA_TEMPLATE = """You are an expert at analyzing photos based on their EXIF metadata.

Below is EXIF metadata from photos in a database that match the query:

{context}

Additional document information:
{documents}

QUESTION: {question}

When answering:
1. Focus on the metadata fields specifically requested in the query
2. If the query asks about locations, pay special attention to state, city, and country fields
3. If the query asks about when photos were taken, focus on date information
4. If the query asks about camera equipment, highlight make and model information
5. Be specific about what metadata was found in the matching photos
6. If multiple photos match the criteria, summarize them by their common attributes
7. Remember that fields like "state" refer to "exif_GPSInfo_state" in the metadata

ANSWER:"""

class ExifMetadataProcessor:
    """Handles specialized queries about EXIF metadata"""
    
    def __init__(self, metadata_handler=None, document_retriever=None, response_formatter=None, llm_manager=None, query_classifier=None):
        """Initialize with required dependencies"""
        self.metadata_handler = metadata_handler
        self.document_retriever = document_retriever
        self.response_formatter = response_formatter
        self.llm_manager = llm_manager
        self.query_classifier = query_classifier
    
    def process_exif_metadata_query(self, query_text, llm=None, is_factual=False):
        """Process a query specifically about EXIF metadata."""
        logger.debug(f"Processing EXIF metadata query: {query_text}")
        
        # Extract locations or other EXIF parameters from the query
        exif_params = self.metadata_handler.extract_metadata_query(query_text, llm)
        logger.debug(f"Extracted EXIF parameters: {exif_params}")
        
        # Construct metadata filters
        filters = {}
        if exif_params.get('location'):
            location = exif_params['location']
            if isinstance(location, dict):
                for key, value in location.items():
                    if value:
                        field_name = f"exif_GPSInfo_{key}"
                        filters[field_name] = {"$eq": value}
            elif isinstance(location, str):
                # Create an OR filter for various location fields
                filters = {
                    "$or": [
                        {"exif_GPSInfo_city": {"$eq": location}},
                        {"exif_GPSInfo_state": {"$eq": location}},
                        {"exif_GPSInfo_country": {"$eq": location}}
                    ]
                }
        
        # Retrieve documents with the metadata filters
        logger.debug(f"Retrieving with filters: {filters}")
        
        # Note: get_langchain_retriever handles score threshold internally
        # based on query structure - we don't need to pass is_factual
        retriever = get_langchain_retriever(
            query_text, 
            top_k=5, 
            metadata_filters=filters
            # score_threshold is determined by get_langchain_retriever
        )
        
        docs = retriever.get_relevant_documents(query_text)
        logger.debug(f"Retrieved {len(docs)} documents")
        
        # For factual queries with no relevant documents, return direct answer with no images
        if is_factual and (not docs or len(docs) == 0):
            direct_answer = self.llm_manager.get_direct_answer(query_text, llm)
            return {
                "answer": direct_answer,
                "text": direct_answer,
                "documents": [],
                "query_type": "direct_answer",
                "metadata_focus": self.query_classifier.detect_metadata_focus(query_text),
                "suppress_images": True
            }
        
        # Process with LLM
        if llm:
            # Format documents for context
            formatted_docs = []
            for i, doc in enumerate(docs):
                metadata = doc.metadata
                path = metadata.get('path', 'Unknown')
                
                # Format key EXIF data for the prompt
                exif_info = ""
                for key, value in metadata.items():
                    if key.startswith('exif_') and not key in ['exif_MakerNote', 'exif_json', 'exif_UserComment']:
                        nice_key = key.replace('exif_', '').replace('_', ' ')
                        exif_info += f"  - {nice_key}: {value}\n"
                
                # Add general description
                doc_content = f"Document {i+1}:\n"
                doc_content += f"Description: {doc.page_content}\n\n"
                doc_content += f"EXIF Metadata:\n{exif_info}\n"
                
                formatted_docs.append(doc_content)
            
            # Create prompt
            prompt = f"""You are a helpful assistant answering questions about EXIF metadata in photos.
I have a collection of photos with their descriptions and EXIF metadata.

USER QUESTION: {query_text}

PHOTO INFORMATION:
{'-'*80}
{"".join(formatted_docs)}
{'-'*80}

Based on the EXIF metadata above, please provide a concise answer to the user's question.
Focus on metadata like locations, dates, camera information, and technical details.
If the information is not available in the metadata, please state that clearly."""

            try:
                # Get response from LLM
                response = llm.invoke(prompt)
                
                # For factual queries, check if there's any real EXIF info in the response
                # If the response is just coming from the LLM's knowledge, suppress images
                suppress_images = False
                if is_factual:
                    # Check if the response mentions any EXIF data or is just general knowledge
                    exif_references = ["exif", "metadata", "gps", "photo", "image", "camera", "location"]
                    contains_exif_info = any(term in response.lower() for term in exif_references)
                    
                    # If it's factual and doesn't reference EXIF data, suppress images
                    suppress_images = is_factual and not contains_exif_info
                
                return {
                    "answer": response,
                    "text": response,
                    "documents": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs],
                    "query_type": "exif_metadata",
                    "metadata_focus": self.query_classifier.detect_metadata_focus(query_text),
                    "structured_query": exif_params,
                    "suppress_images": suppress_images
                }
            except Exception as e:
                logger.error(f"Error getting LLM response: {e}")
                # Fall back to basic summary
        
        # Fallback if LLM fails or is not available
        summary = "Here is the EXIF metadata from relevant photos:\n\n"
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            path = metadata.get('path', 'Unknown')
            filename = path.split('/')[-1] if '/' in path else path
            
            summary += f"Photo {i+1}: {filename}\n"
            
            # Add location if available
            location_parts = []
            for field in ['city', 'state', 'country']:
                if f'exif_GPSInfo_{field}' in metadata and metadata[f'exif_GPSInfo_{field}']:
                    location_parts.append(metadata[f'exif_GPSInfo_{field}'])
            
            if location_parts:
                summary += f"Location: {', '.join(location_parts)}\n"
                
            # Add date if available
            if 'exif_DateTimeOriginal' in metadata:
                summary += f"Date: {metadata['exif_DateTimeOriginal']}\n"
                
            # Add camera info if available
            if 'exif_Make' in metadata or 'exif_Model' in metadata:
                camera = []
                if 'exif_Make' in metadata and metadata['exif_Make']:
                    camera.append(metadata['exif_Make'])
                if 'exif_Model' in metadata and metadata['exif_Model']:
                    camera.append(metadata['exif_Model'])
                if camera:
                    summary += f"Camera: {' '.join(camera)}\n"
            
            summary += "\n"
        
        return {
            "answer": summary,
            "text": summary,
            "documents": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs],
            "query_type": "exif_metadata",
            "metadata_focus": self.query_classifier.detect_metadata_focus(query_text),
            "structured_query": exif_params,
            "suppress_images": is_factual  # Always suppress images for factual queries in fallback mode
        }


class StandardRagProcessor:
    """Handles standard RAG processing"""
    
    def __init__(self, response_formatter=None):
        """Initialize with response formatter"""
        self.response_formatter = response_formatter
    
    def process_standard_rag_query(self, query_text, docs, llm, metadata_focus=None, is_factual=False):
        """Process a standard RAG query with retrieved documents."""
        if not docs:
            return {
                "answer": "I couldn't find any photos matching your query in the collection.",
                "text": "I couldn't find any photos matching your query in the collection.",
                "documents": [],
                "query_type": "rag",
                "suppress_images": True
            }
            
        # Format retrieved documents for RAG processing
        context = self.response_formatter.format_retrieved_documents(docs)

        # If we don't have a valid LLM for generation, create a simple summary
        if llm is None:
            summary = self.response_formatter.generate_simple_summary(docs, metadata_focus or "general")
            return {
                "answer": summary,
                "text": summary,
                "documents": [
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    } for doc in docs
                ],
                "query_type": "rag_fallback",
                "metadata_focus": metadata_focus,
                "suppress_images": is_factual
            }

        # Create RAG prompt
        rag_prompt_template = """You are a helpful assistant analyzing photos based on their metadata and descriptions.
    
Below is metadata from photos in a database:

{context}

QUESTION: {question}

When answering:
1. Consider only the information available in the provided metadata
2. If a photo's metadata includes information related to the question, describe that photo
3. If multiple photos match the query, summarize what they have in common
4. If no photos match the query criteria, clearly state that no matching photos were found
5. Be specific about locations, dates, camera equipment, or other details mentioned in the metadata
6. If the question does not reference images or photos or a person, answer the question without the metadata.

ANSWER:"""

        rag_prompt = PromptTemplate.from_template(rag_prompt_template)

        # Set up RAG chain
        rag_chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )

        # Execute RAG chain
        try:
            answer = rag_chain.invoke(query_text)
        except Exception as e:
            logger.error(f"Error invoking RAG chain: {e}")
            # Provide a fallback answer when the chain fails
            answer = f"I found some photos that might be relevant to your query about {query_text}, but I encountered an error processing the details. The error was: {e}"

        # Return structured response
        return {
            "answer": answer,
            "text": answer,
            "documents": [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in docs
            ],
            "query_type": "rag_standard",
            "metadata_focus": metadata_focus,
            "suppress_images": is_factual
        }


class BatchedDocumentHandler:
    """Handles processing documents in batches for complex queries"""
    
    def __init__(self, response_formatter=None):
        """Initialize with response formatter"""
        self.response_formatter = response_formatter
    
    def process_batched_query(self, query_text, docs, llm, model=None, is_factual=False, metadata_focus=None):
        """Process a query using document batching for complex queries."""
        # Determine an appropriate batch size based on the total number of docs
        batch_size = 5 if len(docs) <= 20 else 10
        logger.debug(f"Using batch size of {batch_size} for {len(docs)} documents")
        
        try:
            # Use batched processing with the dynamic batch size and pass the llm instance
            batch_result = process_documents_in_batches(docs, query_text, model, batch_size=batch_size, llm=llm)
            
            # Check if we got a dictionary with response and doc_mapping
            if isinstance(batch_result, dict) and "response" in batch_result:
                answer_text = batch_result["response"]
                doc_mapping = batch_result.get("doc_mapping", {})
                matched_images = batch_result.get("matched_images", [])
                
                # Log the number of matched images for debugging
                logger.debug(f"Found {len(matched_images)} matched images: {matched_images}")
                logger.debug(f"doc_mapping contains {len(doc_mapping)} entries")
                
                # Process the response to replace photo references with actual images
                processed_answer = self.response_formatter.process_response_with_images(answer_text, doc_mapping)
                
                # Return structured response
                return {
                    "answer": processed_answer,
                    "text": answer_text,  # Original text without images
                    "documents": [{"page_content": doc.page_content, "metadata": doc.metadata} 
                                for doc in docs if doc.metadata.get("path") in 
                                [v.get("path") for v in doc_mapping.values()]],
                    "query_type": "rag_batched",
                    "metadata_focus": metadata_focus,
                    "is_html": True,  # Flag indicating this is HTML content
                    "matched_images": matched_images,
                    "suppress_images": is_factual
                }
            else:
                # Handle case where batch_result is just a string
                logger.warning("Batch result did not return the expected format")
                answer = str(batch_result)
                
                # Return a simple response
                return {
                    "answer": answer,
                    "text": answer,
                    "documents": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs],
                    "query_type": "rag_batched_fallback",
                    "suppress_images": is_factual
                }
                
        except Exception as batch_error:
            logger.error(f"Error in batched processing: {batch_error}, falling back to standard RAG")
            
            # Return error response
            error_msg = f"I encountered an error processing your query in batches: {str(batch_error)}"
            return {
                "answer": error_msg,
                "text": error_msg,
                "error": str(batch_error),
                "documents": [],
                "query_type": "error",
                "suppress_images": True
            }

