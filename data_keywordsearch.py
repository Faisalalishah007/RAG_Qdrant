'''from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# Initialize the embedder
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load Qdrant Vector Store from the existing collection
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embedder,
    collection_name="my_documents",
    path="/home/dell/onboarding/local_qdrant",
    retrieval_mode=RetrievalMode.DENSE,
)

# Define the query
query = "Effects of climate change on temperature?"

# Retrieve documents based on similarity search
# found_docs = qdrant.similarity_search_with_score(query)


# # Print the content and score of the first found document
# if found_docs:
#     document, score = found_docs[0]
#     logger.info("Top document content: {}", document.page_content)
#     logger.info("Cosine score of top document: {}", score)
# else:
#     logger.warning("No documents found.")


from qdrant_client.http import models


found_docs = qdrant.similarity_search_with_score(query, filter=models.Filter("climate"))'''

from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client.http import models
from qdrant_client.models import Filter, FieldCondition, Range


# Initialize the embedder
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load Qdrant Vector Store from the existing collection
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embedder,
    collection_name="my_documents",
    path="/home/dell/onboarding/local_qdrant",
    
)

# Define the query
query = "what is crypto currency and block chain?"

# Define the filter criteria using Qdrant's Filter class
filter_criteria = models.Filter(must=[
    models.FieldCondition(
        key='page_content',
        match=models.MatchText(text="blockchain")
    )
])

# Retrieve documents based on similarity search
found_docs = qdrant.similarity_search_with_score(query, filter=filter_criteria)

# Print the content and score of the first found document
if found_docs:
    document, score = found_docs[0]
    print("Top document content:", document.page_content)
    print("Cosine score of top document:", score)
else:
    print("No documents found.")

