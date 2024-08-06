# data_retrieval.py

# Import necessary modules
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

# Configure loguru to write logs to a file
logger.add("logs/retrieval_file1.log", rotation="1 MB", retention="10 days", level="INFO")


# Initialize the embedder
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load Qdrant Vector Store from the existing collection
# todo: can we use from_existing_collection to add new document.
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embedder,
    collection_name="my_documents",
    path="/home/dell/RAG_Qdrant_first_project/RAG_Qdrant_Project",
    retrieval_mode=RetrievalMode.DENSE,
)

# Define the query
query = "effects of climate change on world temperature"

# Logging the start of the search
logger.info("Starting similarity search for query: {}", query)

# Retrieve documents based on similarity search
found_docs = qdrant.similarity_search_with_score(query)

# Logging the results
logger.info("Similarity search completed. Number of documents found: {}", len(found_docs))

# Print information about 'found_docs'
logger.info("Type of 'found_docs': {}", type(found_docs))
logger.info("Length of 'found_docs': {}", len(found_docs))

# Print the content and score of the first found document
if found_docs:
    document, score = found_docs[0]
    logger.info("Top document content: {}", document.page_content)
    logger.info("Cosine score of top document: {}", score)
else:
    logger.warning("No documents found.")

