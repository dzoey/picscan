# picscan
Scan a directory of images and create a RAG database of descriptions based on the output of a vision model

This effort consists of two programs, a rag_builder and a rag_retriever.   

The rag_builder scans a directory (and sub-directories) for image files (jpg, png) and sends them to an ollama server.  It takes the name of the model tp request the ollama server run and the location of the image directory and command line paramters.  Use -h or --help to get the exact syntax.   While you can specify any model you want, it is strong suggested that a model that can handle images be invoked, such as LLaVA or granite3.2-vision.   The model will analyze the image and return a description.  That description combined with any EXIF data from the image file is stored in a persistent chromadb vector database.  The EXIF data is stored as a series of document metadata keys rather than part of the description document.   If interrupted, rag_builder will check to see if an image has already been processed and if it has been, it will not re-request its processing.  If you want to re-process all images, remove the chromadb database (ragdb).  In addition to the description and EXIF data being stored for each image, the image itself is converted to vectors and stored so that a similarity comparison can be made with another (user uploaded) image in the rag_retriever program.

To run the retriever application, do `python ragapp.py`   You will also want to edit config.python file and adjust the location of where you want the database to reside.
