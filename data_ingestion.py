# data_ingestion.py

# Import necessary modules
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode

# Load and split text documents
loader = TextLoader("required_text.txt")
documents = loader.load()

# Print information about documents
print(f"\ntype of 'documents': {type(documents)}")
print(f"\nNumber of documents loader is getting: {len(documents)}")

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Print information about docs after splitting
print(f"\nNumber of chunks created by text splitter: {len(docs)}\n")

# Create Embedder
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Save embeddings to Qdrant Vector Store in local disk memory
qdrant = QdrantVectorStore.from_documents(
    docs,
    embedding=embedder,
    path="/home/dell/onboarding/local_qdrant",
    collection_name="my_documents",
    retrieval_mode=RetrievalMode.DENSE,
)

print("Data ingestion completed successfully.")
