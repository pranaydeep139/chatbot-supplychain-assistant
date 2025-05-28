
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

collection_name = "supply_docs"

# Delete the collection if it exists
if collection_name in [c.name for c in client.get_collections().collections]:
    client.delete_collection(collection_name=collection_name)
    print(f"Deleted collection: {collection_name}")
else:
    print(f"No existing collection named '{collection_name}' found.")
