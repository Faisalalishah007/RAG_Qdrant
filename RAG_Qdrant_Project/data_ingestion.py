# # data_ingestion_with_logging.py

# # Import necessary modules
# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_qdrant import QdrantVectorStore, RetrievalMode
# from loguru import logger

# # Configure loguru to write logs to a file
# logger.add("logs/ingestion_file.log", rotation="1 MB", retention="10 days", level="INFO")

# # Load and split text documents
# loader = TextLoader("/home/dell/RAG_Qdrant_first_project/required_text.txt")

# # Log document loading start
# logger.info("Loading documents from 'required_text.txt'")
# documents = loader.load()

# # Log information about loaded documents
# logger.info("Documents loaded. Type: {}", type(documents))
# logger.info("Number of documents loaded: {}", len(documents))

# # Split documents into chunks
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)

# # Log information about document chunks
# logger.info("Documents split into chunks. Number of chunks created: {}", len(docs))

# # Create Embedder
# embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # Log embedder initialization
# logger.info("Embedder initialized with model: all-MiniLM-L6-v2")

# # Save embeddings to Qdrant Vector Store in local disk memory
# # Todo: use relative path instead of absolute path
# # Todo: create empty collection and then add documents.
# qdrant = QdrantVectorStore.from_documents(
#     docs,
#     embedding=embedder,
#     path="/home/dell/RAG_Qdrant_first_project/RAG_Qdrant_Project",
#     collection_name="my_documents",
#     retrieval_mode=RetrievalMode.DENSE,
# )

# # Log Qdrant Vector Store initialization and document ingestion
# logger.info("Qdrant Vector Store initialized and documents indexed. Path:local_qdrant")

# # Log data ingestion completion
# logger.info("Data ingestion completed successfully.")


# data_ingestion_with_logging.py

import uuid
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from loguru import logger

# Configure loguru to write logs to a file
logger.add("logs/ingestion_file.log", rotation="1 MB", retention="10 days", level="INFO")

# Load the text document
loader = TextLoader("/home/dell/RAG_Qdrant_first_project/required_text.txt")

# Log document loading start
logger.info("Loading documents from 'required_text.txt'")
documents = loader.load()

# Log information about loaded documents
logger.info("Documents loaded. Type: {}", type(documents))
logger.info("Number of documents loaded: {}", len(documents))

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = text_splitter.split_documents(documents)

# Assign unique IDs to document chunks
for chunk in chunks:
    chunk.metadata['id'] = str(uuid.uuid4())  # Assign a unique ID to each chunk

# Log information about document chunks with IDs
logger.info("Documents split into chunks. Number of chunks created: {}", len(chunks))

# Create Embedder
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Log embedder initialization
logger.info("Embedder initialized with model: all-MiniLM-L6-v2")

# Initialize Qdrant Vector Store
collection_name = "my_documents"
qdrant = QdrantVectorStore.from_documents(
    chunks,
    embedding=embedder,
    collection_name=collection_name,
    retrieval_mode=RetrievalMode.DENSE,
)

# Define a function to check if a document is already in the collection
def is_document_exists(qdrant, document_id):
    try:
        # Check if the document with the given ID exists in the collection
        search_results = qdrant.search(query=document_id, top_k=1, search_type="exact")
        return len(search_results) > 0
    except Exception as e:
        logger.error("Error checking document existence: {}", e)
        return False

# Filter out documents that are already present in the collection
new_docs = [doc for doc in chunks if not is_document_exists(qdrant, doc.metadata['id'])]

if not new_docs:
    logger.info("No new documents to ingest. All documents are already present in the collection.")
else:
    # Add new documents to Qdrant Vector Store
    qdrant.add_documents(new_docs)

    # Log Qdrant Vector Store update and document ingestion
    logger.info("Qdrant Vector Store updated with new documents.")

# Log data ingestion completion
logger.info("Data ingestion completed successfully.")
