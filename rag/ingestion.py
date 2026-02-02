import os
import asyncio
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import VectorParams, Distance
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pymupdf
from dotenv import load_dotenv

load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "manual_chunks"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, local, good enough

class IngestionPipeline:
    def __init__(self):
        print("Initializing Ingestion Pipeline...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # Connect to Qdrant
        if QDRANT_URL.startswith("http://localhost"):
            self.qdrant = QdrantClient(url=QDRANT_URL)
        else:
            self.qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            
    def parse_pdf(self, file_path: str):
        print(f"Parsing PDF: {file_path}...")
        doc = pymupdf.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        print(f"Extracted {len(text)} characters.")
        return text

    def chunk_text(self, text: str):
        print("Chunking text...")
        # Voice-optimized chunking strategy:
        # Smaller chunks (approx 2-3 sentences) are better for specific answers.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # ~100-150 words
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_text(text)
        print(f"Created {len(chunks)} chunks.")
        return chunks

    def index_chunks(self, chunks):
        print("Creating embeddings and indexing...")
        
        # Recreate collection to ensure clean slate
        self.qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

        # Batch processing
        batch_size = 100
        total_batches = len(chunks) // batch_size + 1
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            embeddings = self.encoder.encode(batch).tolist()
            
            points = [
                models.PointStruct(
                    id=i + idx,
                    vector=embedding,
                    payload={"text": text}
                )
                for idx, (text, embedding) in enumerate(zip(batch, embeddings))
            ]
            
            self.qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            print(f"Indexed batch {i//batch_size + 1}/{total_batches}")

        print("Ingestion Complete!")

async def main():
    pipeline = IngestionPipeline()
    manual_path = "cis_manual.pdf"
    
    if not os.path.exists(manual_path):
        print(f"Error: {manual_path} not found.")
        return

    raw_text = pipeline.parse_pdf(manual_path)
    chunks = pipeline.chunk_text(raw_text)
    pipeline.index_chunks(chunks)

if __name__ == "__main__":
    asyncio.run(main())
