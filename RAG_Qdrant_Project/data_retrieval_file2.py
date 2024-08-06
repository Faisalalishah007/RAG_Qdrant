# data_retrieval_file2.py

# Import necessary modules
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger


# Configure loguru to write logs to a file
logger.add("logs/retrieval_file2.log", rotation="1 MB", retention="10 days", level="INFO")

# Initialize the embedder
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Logging embedder initialization
logger.info("Embedder initialized with model: all-MiniLM-L6-v2")

# Load Qdrant Vector Store from the existing collection
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embedder,
    collection_name="my_documents",
    path="/home/dell/RAG_Qdrant_first_project/RAG_Qdrant_Project",
    retrieval_mode=RetrievalMode.DENSE,
)

# Logging Qdrant Vector Store loading
logger.info("Qdrant Vector Store loaded from collection: my_documents")

# Set up retriever
retriever = qdrant.as_retriever()

# Logging retriever setup
logger.info("Retriever initialized")

# Change retriever to use MMR (Maximal Marginal Relevance)
retriever = qdrant.as_retriever(search_type="mmr")

# Logging retriever search type change
logger.info("Retriever search type set to MMR")

# Define the query
query = "effects of climate change on temperature"

# Logging query definition
logger.info("Query defined: {}", query)

# Invoke the retriever and log the result
result = retriever.invoke(query)[0]

# Logging the result
logger.info("Query result: {}", result)
