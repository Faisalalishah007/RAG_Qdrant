### loading langchain
from langchain_community.document_loaders import TextLoader
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_qdrant import RetrievalMode

### loading Embedder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, RetrievalMode


## loading and splitting text documents
loader = TextLoader("sample.txt")
documents = loader.load()
## printing information about documents 
print(f"\ntype of 'documents': {type(documents)}")
print(f"\nNumber of  documents loader is getting: {len(documents)}")

## splitting documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
## printing information about docs after splitting
print(f"\nNumber of chunks created by text splitter: {len(docs)}\n")

## Createing Embedder
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## qdrant in local disk memory ##########
qdrant = QdrantVectorStore.from_documents(
    docs,
    embedding=embedder,
    path="C:/Users/IceCandyman/Desktop/nboarding/local_qdrant",
    collection_name="my_documents",
    retrieval_mode=RetrievalMode.DENSE,
)

query = "In what ways is generative AI being utilized in creative arts and content creation?"
found_docs = qdrant.similarity_search_with_score(query)

## Printing info about 'found_docs'
print(f"\ntype of 'found_docs': {type(found_docs)}")
print(f"\nLength of found_docs: {len(found_docs)}\n")


document, score = found_docs[0]
print(document.page_content)
print(f"\nthis Score is cosine distance: {score}")
