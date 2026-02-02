import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "manual_chunks"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

class Retriever:
    def __init__(self):
        print("Initializing Retriever...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        if QDRANT_URL.startswith("http://localhost"):
            self.qdrant = QdrantClient(url=QDRANT_URL)
        else:
            self.qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    async def search(self, query: str, limit: int = 5):
        print(f"Searching for: {query}")
        vector = self.encoder.encode(query).tolist()
        
        # Use query_points for compatibility with newer clients
        # Note: query_points returns QueryResponse, we need .points
        try:
             # Try modern API
            response = self.qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                limit=limit
            )
            hits = response.points
        except AttributeError:
             # Fallback if older client or different structure
            hits = self.qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=vector,
                limit=limit
            )
        
        # Extract text from payload
        texts = [hit.payload["text"] for hit in hits]
        return texts
