from qdrant_client import QdrantClient

client = QdrantClient(path="/home/dell/RAG_Qdrant_first_project/RAG_Qdrant_Project")

result=client.get_collection(collection_name="my_documents")
print(result)