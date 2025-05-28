import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

# Load environment variables
load_dotenv()

print(os.getenv("QDRANT_URL"))
print(os.getenv("QDRANT_API_KEY"))

# Initialize client
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),  # HTTPS URL without port
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=60
    # https=True,
    # check_compatibility=False  # Temporary workaround
)

# Test connection
# try:
#     collections = client.get_collections()
#     print("Connected successfully. Collections:", collections)
# except Exception as e:
#     print("Connection failed:", str(e))


# Initialize Hugging Face embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load and process documents
docs = []
supply_docs_path = "supply_docs/"
for filename in os.listdir(supply_docs_path):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(supply_docs_path, filename))
        docs.extend(loader.load())

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(docs)


# client.create_collection(
#     collection_name="docs_collection_2",
#     vectors_config=VectorParams(size=384, distance=Distance.COSINE),
# )
# print("Collection created!")

# Initialize the Qdrant vector store
vector_store = QdrantVectorStore(
    client=client,
    collection_name="docs_collection_2",
    embedding=embedding_model,
)

print("Vector store created!")

# Define batch size
print("Total documents: ", len(documents))
batch_size = 50
for i in range(0, len(documents), batch_size):
    print(f"Adding batch_{i+1}")
    batch = documents[i:i + batch_size]
    vector_store.add_documents(batch)
    print(f"batch_{i+1} added!")



# Create Qdrant vector store
# qdrant = QdrantVectorStore.from_texts(
#     texts=documents,
#     embedding=embedding_model,
#     url=os.getenv("QDRANT_URL"),
#     api_key=os.getenv("QDRANT_API_KEY"),
#     collection_name="supply_docs"
# )
