# data_retrieval.py

# Import necessary modules
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings

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
found_docs = qdrant.similarity_search_with_score(query)

# Print information about 'found_docs'
print(f"\ntype of 'found_docs': {type(found_docs)}")
print(f"\nLength of found_docs: {len(found_docs)}\n")

# Print the content and score of the first found document
if found_docs:
    document, score = found_docs[0]
    print(document.page_content)
    print(f"\nThis is cosine score: {score}")
else:
    print("No documents found.")
