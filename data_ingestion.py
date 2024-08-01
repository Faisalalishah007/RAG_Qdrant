# data_ingestion_with_logging.py

# Import necessary modules
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from loguru import logger

# Configure loguru to write logs to a file
logger.add("ingestion_file.log", rotation="1 MB", retention="10 days", level="INFO")

# Load and split text documents
loader = TextLoader("required_text.txt")

# Log document loading start
logger.info("Loading documents from 'required_text.txt'")
documents = loader.load()

# Log information about loaded documents
logger.info("Documents loaded. Type: {}", type(documents))
logger.info("Number of documents loaded: {}", len(documents))

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Log information about document chunks
logger.info("Documents split into chunks. Number of chunks created: {}", len(docs))

# Create Embedder
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Log embedder initialization
logger.info("Embedder initialized with model: all-MiniLM-L6-v2")

# Save embeddings to Qdrant Vector Store in local disk memory
# Todo: use relative path instead of absolute path
# Todo: create empty collection and then add documents.
qdrant = QdrantVectorStore.from_documents(
    docs,
    embedding=embedder,
    path="/home/dell/onboarding/local_qdrant",
    collection_name="my_documents",
    retrieval_mode=RetrievalMode.DENSE,
)

# Log Qdrant Vector Store initialization and document ingestion
logger.info("Qdrant Vector Store initialized and documents indexed. Path: /home/dell/onboarding/local_qdrant")

# Log data ingestion completion
logger.info("Data ingestion completed successfully.")

#